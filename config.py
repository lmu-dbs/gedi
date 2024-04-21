import json
import os
import warnings

from gedi.utils.io_helpers import sort_files
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
def get_run_params(alg_params_json: str) -> dict:
    """
    Loads the running configuration given from a json file or the default dictionary from the code.
    @param alg_params_json: str
        Path to the json data with the running configuration
    @return: dict
        Running Configuration
    """
    if alg_params_json is not None:
        return json.load(open(alg_params_json))
    else:
        warnings.warn('The default run option is used instead of a .json-file.\n'
                      '  Use a configuration from the `config_files`-folder together with the args `-o`.')
        return {
            RUN_OPTION: COMPARE,
            PLOT_TYPE: COLOR_MAP,  # 'heat_map', 'color_map', '3d_map', 'explained_var_plot'
            PLOT_TICS: True,
            N_COMPONENTS: 2,
            INPUT_NAME: 'runningExample',
            SAVE_RESULTS: True,
            LOAD_RESULTS: True
        }

def get_files_and_kwargs(params: dict):
    """
    This method returns the filename list of the trajectory and generates the kwargs for the DataTrajectory.
    The method is individually created for the available data set.
    Add new trajectory options, if different data set are used.
    @param params: dict
        running configuration
    @return: tuple
        list of filenames of the trajectories AND
        kwargs with the important arguments for the classes
    """
    try:
        input_name = params[INPUT_NAME]
    except KeyError as e:
        raise KeyError(f'Run option parameter is missing the key: `{e}`. This parameter is mandatory.')

    #TODO: generate parent directories if they don't exist
    if input_name == 'test':
        filename_list = list(tqdm(sort_files(os.listdir('data/test_2'))))
        kwargs = {FILENAME: filename_list, FOLDER_PATH: 'data/test_2'}
    elif input_name == 'realLogs':
        filename_list = list(tqdm(sort_files(os.listdir('data/real_event_logs'))))
        kwargs = {FILENAME: filename_list, FOLDER_PATH: 'data/real_event_logs'}
    elif input_name == 'gen5':
        filename_list = list(tqdm(sort_files(os.listdir('data/event_log'))))[:5]
        kwargs = {FILENAME: filename_list, FOLDER_PATH: 'data/event_log'}
    elif input_name == 'gen20':
        filename_list = list(tqdm(sort_files(os.listdir('data/event_log'))))[:20]
        kwargs = {FILENAME: filename_list, FOLDER_PATH: 'data/event_log'}
    elif input_name == 'runningExample':
        filename_list = ['running-example.xes']
        kwargs = {FILENAME: filename_list[0], FOLDER_PATH: 'data/'}
    elif input_name == 'metaFeatures':
        filename_list = ['log_features.csv']
        kwargs = {FILENAME: filename_list[0], FOLDER_PATH: 'results/'}
    else:
        raise ValueError(f'No data trajectory was found with the name `{input_name}`.')

    #filename_list.pop(file_element)
    kwargs[PARAMS] = params
    return filename_list, kwargs
