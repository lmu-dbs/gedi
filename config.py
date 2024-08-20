import json
import os
import warnings

from tqdm import tqdm
from utils.param_keys import INPUT_NAME, FILENAME, FOLDER_PATH, PARAMS

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
            {ALGORITHM_NAME: 'pca', NDIM: TENSOR_NDIM},
            ]
