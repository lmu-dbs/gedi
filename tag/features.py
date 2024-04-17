import json
import multiprocessing
import numpy as np
import pandas as pd
import os

from datetime import datetime as dt
from functools import partial
from feeed.feature_extractor import extract_features
from pathlib import Path, PurePath
from sklearn.impute import SimpleImputer
from utils.param_keys import INPUT_PATH
from utils.param_keys.features import FEATURE_PARAMS, FEATURE_SET
from tag.utils.io_helpers import dump_features_json

def get_sortby_parameter(elem):
    number = int(elem.rsplit(".")[0].rsplit("_", 1)[1])
    return number


class EventLogFile:
    def __init__(self, filename, folder_path):
        self.root_path: Path = Path(folder_path)
        self.filename: str = filename

    @property
    def filepath(self) -> str:
        return str(os.path.join(self.root_path, self.filename))

class EventLogFeatures(EventLogFile):
    def __init__(self, filename, folder_path='data/event_log', params=None, logs=None, ft_params=None):
        super().__init__(filename, folder_path)
        if ft_params == None:
            self.params = None
            self.feat = None
            return
        elif ft_params.get(FEATURE_PARAMS) == None:
            self.params = {FEATURE_SET: None}
        else:
            self.params=ft_params.get(FEATURE_PARAMS)

        # TODO: handle parameters in main, not in features. Move to main.py
        if ft_params[INPUT_PATH]:
            input_path = ft_params[INPUT_PATH]
            if os.path.isfile(input_path):
                self.root_path = Path(os.path.split(input_path)[0])
                self.filename = os.path.split(input_path)[-1]
            else:
                self.root_path = Path(input_path)
                # Check if directory exists, if not, create it
                if not os.path.exists(input_path):
                    os.makedirs(input_path)
                self.filename = os.listdir(input_path)

        try:
            start = dt.now()
            print("=========================== EventLogFeatures Computation===========================")

            print(f"INFO: Running with {ft_params}")

            if str(self.filename).endswith('csv'): # Returns dataframe from loaded metafeatures file
                self.feat = pd.read_csv(self.filepath)
                print(f"SUCCESS: EventLogFeatures loaded features from {self.filepath}")
            elif isinstance(self.filename, list): # Computes metafeatures for list of .xes files
                combined_features=pd.DataFrame()
                if self.filename[0].endswith(".json"):
                    self.filename = [ filename for filename in self.filename if filename.endswith(".json")]
                    dfs = []
                    for filename in self.filename:
                        print(f"INFO: Reading features from {os.path.join(self.root_path, filename)}")
                        data = pd.read_json(str(os.path.join(self.root_path,filename)), lines=True)
                        #data['log']=filename.replace("genEL","").rsplit("_",2)[0]
                        #print(data)
                        dfs.append(data)
                    combined_features= pd.concat(dfs, ignore_index = True)

                    self.feat = combined_features
                    self.filename = os.path.split(self.root_path)[-1] + '_feat.csv'
                    self.root_path=Path(os.path.split(self.root_path)[0])
                    combined_features.to_csv(self.filepath, index=False)
                    print(f"SUCCESS: EventLogFeatures took {dt.now()-start} sec. Saved {len(self.feat.columns)} features for {len(self.feat)} in {self.filepath}")
                    print("=========================== ~ EventLogFeatures Computation=========================")
                    return
                else:
                    self.filename = [ filename for filename in self.filename if filename.endswith(".xes")]

                # TODO: only include xes logs in self.filename, otherwise it will result in less rows. Implement skip exception with warning
                try:
                    num_cores = multiprocessing.cpu_count() if len(
                        self.filename) >= multiprocessing.cpu_count() else len(self.filename)
                    with multiprocessing.Pool(num_cores) as p:
                        try:
                            print(
                                f"INFO: EventLogFeatures starting at {start.strftime('%H:%M:%S')} using {num_cores} cores for {len(self.filename)} files, namely {self.filename}...")
                            result = p.map(partial(self.extract_features_wrapper, feature_set = self.params[FEATURE_SET])
                                       , self.filename)
                            result = [i for i in result if i is not None]
                            combined_features = pd.DataFrame.from_dict(result)
                        except Exception as e:
                            print(e)

                except IndexError as error:
                    print("IndexError:", error)
                    for file in self.filename:
                        print(f"INFO: Computing features for {file}...")
                        features = self.extract_features_wrapper(str(os.path.join(self.root_path, file)),
                                feature_set = self.params[FEATURE_SET])
                        features['log'] = file.rsplit(".", 1)[0]
                        temp = pd.DataFrame.from_dict([features])
                        combined_features = pd.concat([combined_features, temp], ignore_index=True)

                except KeyError as error:
                    print("Ignoring KeyError", error)
                    # Aggregates metafeatures in saved Jsons into dataframe
                    path_to_json = f"output/features/{str(self.root_path).split('/',1)[1]}"
                    df = pd.DataFrame()
                    # Iterate over the files in the directory
                    for filename in os.listdir(path_to_json):
                        if filename.endswith('.json'):
                            i_path = os.path.join(path_to_json, filename)
                            with open(i_path) as f:
                                data = json.load(f)
                                temp_df = pd.DataFrame([data])
                                df = pd.concat([df, temp_df])
                    combined_features = df

                self.filename = os.path.split(self.root_path)[-1] + '_feat.csv'
                self.root_path=Path(os.path.split(self.root_path)[0])
                combined_features.to_csv(self.filepath, index=False)

                self.feat = combined_features
        except (IOError, FileNotFoundError) as err:
            print(err)
            print(f"Cannot load {self.filepath}. Double check for file or change config 'load_results' to false")
        else:
            print(f"SUCCESS: EventLogFeatures took {dt.now()-start} sec. Saved {len(self.feat.columns)-1} features for {len(self.feat)} in {self.filepath}")
            print("=========================== ~ EventLogFeatures Computation=========================")

    #TODO: Implement optional trying to read already computed jsons first.
    def extract_features_wrapper(self, file, feature_set=None):
        try:
            file_path = os.path.join(self.root_path, file)
            print(f"  INFO: Starting FEEED for {file_path} and {feature_set}")
            features = extract_features(file_path, feature_set)

        except Exception as e:
            print("ERROR: for ",file.rsplit(".", 1)[0], feature_set, "skipping and continuing with next log.")
            print(e)
            return None

        identifier = file.rsplit(".", 1)[0]
        print(f"  DONE: {file_path}. FEEED computed {feature_set}")
        dump_features_json(features, self.root_path, identifier)
        return features

