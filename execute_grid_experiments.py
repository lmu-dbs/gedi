import multiprocessing
import os

from datetime import datetime as dt
from gedi.utils.io_helpers import sort_files
from tqdm import tqdm

#TODO: Pass i properly
def multi_experiment_wrapper(config_file, i=0):
    print(f"=========================STARTING EXPERIMENT #{i+1}=======================")
    print(f"INFO: Executing with {config_file}")
    os.system(f"python -W ignore main.py -o config_files/options/baseline.json -a {config_file}")
    print(f"=========================FINISHED EXPERIMENT #{i+1}=======================")

if __name__ == '__main__':
    EXPERIMENTS_FOLDER = os.path.join('config_files','algorithm','34_bpic_features')
    EXPERIMENTS_FOLDER = os.path.join('config_files','algorithm','grid_1obj')
    EXPERIMENTS_FOLDER = os.path.join('config_files','algorithm','grid_experiments')
    EXPERIMENTS_FOLDER = os.path.join('config_files','algorithm','test')
    start = dt.now()

    experiment_list = list(tqdm(sort_files(os.listdir(EXPERIMENTS_FOLDER))))
    experiment_list = [os.path.join(EXPERIMENTS_FOLDER, config_file) for config_file in experiment_list]
    experiment_list = experiment_list[:10]

    print(f"========================STARTING MULTIPLE EXPERIMENTS=========================")
    print(f"INFO: {EXPERIMENTS_FOLDER} contains config files for {len(experiment_list)}.")
    try:
        num_cores = multiprocessing.cpu_count() if len(
            experiment_list) >= multiprocessing.cpu_count() else len(experiment_list)
        with multiprocessing.Pool(num_cores) as p:
            try:
                print(f"INFO: Multi Experiments starting at {start.strftime('%H:%M:%S')} using {num_cores} cores for {len(experiment_list)} experiments...")
                result = p.map(multi_experiment_wrapper, experiment_list)
            except Exception as e:
                print(e)
    except Exception as e:
        print("pare", e)

        #for i, config_file in enumerate(experiment_list[:2]):