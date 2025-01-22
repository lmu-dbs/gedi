import os
import random
import re
import xml.etree.ElementTree as ET

from ConfigSpace.configuration_space import Configuration
from gedi.features import compute_features_from_event_data
from gedi.generation.model import create_model
from gedi.generation.simulation import simulate_log
from gedi.utils.io_helpers import get_output_key_value_location, dump_features_json, compute_similarity
from pm4py import write_xes
from xml.dom import minidom

RANDOM_SEED = 10

#TODO: Move to io_helpers
def add_extension_before_traces(xes_file):
    def removeextralines(elem):
        hasWords = re.compile("\\w")
        for element in elem.iter():
            if not re.search(hasWords,str(element.tail)):
                element.tail=""
            if not re.search(hasWords,str(element.text)):
                element.text = ""
    # Register the namespace
    ET.register_namespace('', "http://www.xes-standard.org/")

    # Parse the original XML
    tree = ET.parse(xes_file)
    root = tree.getroot()

    # Add extensions
    extensions = [
        {'name': 'Lifecycle', 'prefix': 'lifecycle', 'uri': 'http://www.xes-standard.org/lifecycle.xesext'},
        {'name': 'Time', 'prefix': 'time', 'uri': 'http://www.xes-standard.org/time.xesext'},
        {'name': 'Concept', 'prefix': 'concept', 'uri': 'http://www.xes-standard.org/concept.xesext'}
    ]

    for ext in extensions:
        extension_elem = ET.Element('extension', ext)
        root.insert(0, extension_elem)

    # Add global variables
    globals = [
        {
            'scope': 'event',
            'attributes': [
                {'key': 'lifecycle:transition', 'value': 'complete'},
                {'key': 'concept:name', 'value': '__INVALID__'},
                {'key': 'time:timestamp', 'value': '1970-01-01T01:00:00.000+01:00'}
            ]
        },
        {
            'scope': 'trace',
            'attributes': [
                {'key': 'concept:name', 'value': '__INVALID__'}
            ]
        }
    ]
    for global_var in globals:
        global_elem = ET.Element('global', {'scope': global_var['scope']})
        for attr in global_var['attributes']:
            string_elem = ET.SubElement(global_elem, 'string', {'key': attr['key'], 'value': attr['value']})
        root.insert(len(extensions), global_elem)


    # Pretty print the Xes
    removeextralines(root)
    xml_str = minidom.parseString(ET.tostring(root)).toprettyxml()
    with open(xes_file, "w") as f:
        f.write(xml_str)

def generate_log(config: Configuration, seed: int = 0):
    random.seed(RANDOM_SEED)
    tree = create_model(config)
    random.seed(RANDOM_SEED)
    log = simulate_log(tree, config)
    #random.seed(RANDOM_SEED)
    #result = self.eval_log(log)
    return log

#TODO: See what parameters we can spare
def generate_optimized_log(config: Configuration, feature_keys, output_path, objectives, task, identifier):
    if isinstance(config, list):
        config = config[0]
    log = generate_log(config)
    random.seed(RANDOM_SEED)
    generated_features = compute_features_from_event_data(feature_keys, log)
    save_path = get_output_key_value_location(task.to_dict(),
                                        output_path, identifier, feature_keys)+".xes"
    write_xes(log, save_path)
    add_extension_before_traces(save_path)
    print("SUCCESS: Saved generated event log in", save_path)
    features_to_dump = generated_features

    features_to_dump['log']= os.path.split(save_path)[1].split(".")[0]
    # calculating the manhattan distance of the generated log to the target features
    #features_to_dump['distance_to_target'] = calculate_manhattan_distance(objectives, features_to_dump)
    features_to_dump['target_similarity'] = compute_similarity(objectives, features_to_dump)
    dump_features_json(features_to_dump, save_path)

    #return generated_features
    return {
    #"configuration": config,
    #"log": log,
    "features": generated_features,
    }
