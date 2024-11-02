import json
import warnings

from gedi.utils.param_keys import PIPELINE_STEP, INPUT_PATH, OUTPUT_PATH
from gedi.utils.param_keys.features import FEATURE_SET, FEATURE_PARAMS

def get_model_params_list(alg_json_file: str) :#-> list[dict]:
    """
    Loads the list of model configurations given from a json file or the default list of dictionary from the code.
    @param alg_json_file: str
        Path to the json data with the running configuration
    @return: list[dict]
        list of model configurations
    """
    if alg_json_file is not None:
        return json.load(open(alg_json_file))
    else:
        warnings.warn('The default model parameter list is used instead of a .json-file.\n'
                      '  Use a configuration from the `config_files`-folder together with the args `-a`.')
        return [
            {PIPELINE_STEP: 'feature_extraction', INPUT_PATH: 'data/test',
             FEATURE_PARAMS: {FEATURE_SET: ['ratio_unique_traces_per_trace',
                                            'ratio_most_common_variant']},
             OUTPUT_PATH: 'output/plots'}
            ]
