import multiprocessing
import os
import pandas as pd
import random
from ConfigSpace import Configuration, ConfigurationSpace
from datetime import datetime as dt
from feeed.activities import Activities as activities
from feeed.end_activities import EndActivities as end_activities
from feeed.epa_based import Epa_based as epa_based
from feeed.eventropies import Eventropies as eventropies
from feeed.feature_extractor import feature_type
from feeed.simple_stats import SimpleStats as simple_stats
from feeed.start_activities import StartActivities as start_activities
from feeed.trace_length import TraceLength as trace_length
from feeed.trace_variant import TraceVariant as trace_variant
from pm4py import write_xes
from pm4py.sim import play_out
from smac import HyperparameterOptimizationFacade, Scenario
from gedi.features import compute_features_from_event_data
from gedi.generation.generator import generate_log, generate_optimized_log, add_extension_before_traces
from gedi.utils.column_mappings import column_mappings
from gedi.utils.io_helpers import get_output_key_value_location, dump_features_json, compute_similarity
from gedi.utils.io_helpers import read_csvs
from gedi.utils.param_keys import OUTPUT_PATH, INPUT_PATH
from gedi.utils.param_keys.generator import GENERATOR_PARAMS, EXPERIMENT, CONFIG_SPACE, N_TRIALS
from functools import partial

RANDOM_SEED = 10
random.seed(RANDOM_SEED)

def get_tasks(experiment, output_path="", reference_feature=None):
    #Read tasks from file.
    if isinstance(experiment, str) and experiment.endswith(".csv"):
        tasks = pd.read_csv(experiment, index_col=None)
        output_path=os.path.join(output_path,os.path.split(experiment)[-1].split(".")[0])
        if 'task' in tasks.columns:
            tasks.rename(columns={"task":"log"}, inplace=True)
    elif isinstance(experiment, str) and os.path.isdir(os.path.join(os.getcwd(), experiment)):
        tasks = read_csvs(experiment, reference_feature)
    #Read tasks from a real log features selection.
    elif isinstance(experiment, dict) and INPUT_PATH in experiment.keys():
        output_path=os.path.join(output_path,os.path.split(experiment.get(INPUT_PATH))[-1].split(".")[0])
        tasks = pd.read_csv(experiment.get(INPUT_PATH), index_col=None)
        id_col = tasks.select_dtypes(include=['object']).dropna(axis=1).columns[0]
        if "objectives" in experiment.keys():
            incl_cols = experiment["objectives"]
            tasks = tasks[(incl_cols +  [id_col])]
    # TODO: Solve/Catch error for different objective keys.
    #Read tasks from config_file with list of targets
    elif isinstance(experiment, list):
        tasks = pd.DataFrame.from_dict(data=experiment)
    #Read single tasks from config_file
    elif isinstance(experiment, dict):
        tasks = pd.DataFrame.from_dict(data=[experiment])
    else:
        raise FileNotFoundError(f"{experiment} not found. Please check path in filesystem.")
    return tasks, output_path


class GenerateEventLogs():
    """
    Generates event logs with the provided parameters.
    @param params: dict
        contains the generator parameters
        experiment: contains a dict with the desired feature values
        config_space: contains a dict with possible configuration parameter ranges
        n_trials: contains the number of trials
        #system_params: contains the system parameters, which don't change e.g. n_trials
        #generator_type: contains the generator type as a string
    @return: None
    """
    def __init__(self, params=None) -> None:
        print("=========================== Generator ==========================")
        if params is None:
            default_params = {'generator_params': {'experiment': {'ratio_top_20_variants': 0.2, 'epa_normalized_sequence_entropy_linear_forgetting': 0.4},
                                                   'config_space': {'mode': [5, 20], 'sequence': [0.01, 1], 'choice': [0.01, 1], 'parallel': [0.01, 1],
                                                                    'loop': [0.01, 1], 'silent': [0.01, 1], 'lt_dependency': [0.01, 1], 'num_traces': [10, 101], 'duplicate': [0], 'or': [0]},
                                                   'n_trials': 50}}
            raise TypeError(f"Missing 'params'. Please provide a dictionary with generator parameters as so: {default_params}. See https://github.com/lmu-dbs/gedi for more info.")
        print(f"INFO: Running with {params}")
        start = dt.now()
        if params.get(OUTPUT_PATH) is None:
            self.output_path = 'data/generated'
        else:
            self.output_path = params.get(OUTPUT_PATH)
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path, exist_ok=True)

        if self.output_path.endswith('csv'):
            self.generated_features = pd.read_csv(self.output_path)
            return

        generator_params = params.get(GENERATOR_PARAMS)
        experiment = generator_params.get(EXPERIMENT)

        if experiment is not None:
            tasks, output_path = get_tasks(experiment, self.output_path)
            columns_to_rename = {col: column_mappings()[col] for col in tasks.columns if col in column_mappings()}
            tasks = tasks.rename(columns=columns_to_rename)
            self.output_path = output_path

        if tasks is not None:
            self.feature_keys = sorted([feature for feature in tasks.columns.tolist() if feature != "log"])
            num_cores = multiprocessing.cpu_count() if len(tasks) >= multiprocessing.cpu_count() else len(tasks)
            #self.generator_wrapper([*tasks.iterrows()][0])# For testing
            with multiprocessing.Pool(num_cores) as p:
                print(f"INFO: Generator starting at {start.strftime('%H:%M:%S')} using {num_cores} cores for {len(tasks)} tasks...")
                random.seed(RANDOM_SEED)
                partial_wrapper = partial(self.generator_wrapper, generator_params=generator_params)
                generated_features = p.map(partial_wrapper, [(index, row) for index, row in tasks.iterrows()])
            self.generated_features = [
                        {
                            #'log': config.get('log'),
                            'features': config.get('features')}
                            for config in generated_features
                            if 'features' in config #and 'log' in config
                    ]

        else:
            random.seed(RANDOM_SEED)
            configs = self.optimize(generator_params=generator_params)
            if type(configs) is not list:
                configs = [configs]
            temp = self.generate_optimized_log(configs[0])
            self.generated_features = [temp['features']] if 'features' in temp else []
            save_path = get_output_key_value_location(generator_params[EXPERIMENT],
                                             self.output_path, "genEL")+".xes"
            write_xes(temp['log'], save_path)
            add_extension_before_traces(save_path)
            print("SUCCESS: Saved generated event log in", save_path)
        print(f"SUCCESS: Generator took {dt.now()-start} sec. Generated {len(self.generated_features)} event log(s).")
        print(f"         Saved generated logs in {self.output_path}")
        print("========================= ~ Generator ==========================")

    def clear(self):
        print("Clearing parameters...")
        self.generated_features = None
        # self.configs = None
        # self.params = None
        self.output_path = None
        self.feature_keys = None

    def generator_wrapper(self, task, generator_params=None):
        try:
            identifier = [x for x in task[1] if isinstance(x, str)][0]
        except IndexError:
            identifier = task[0]+1
        identifier = "genEL" +str(identifier)

        task = task[1].drop('log', errors='ignore')
        self.objectives = task.dropna().to_dict()
        random.seed(RANDOM_SEED)
        configs = self.optimize(generator_params = generator_params)

        random.seed(RANDOM_SEED)

        generated_features = generate_optimized_log(configs, self.feature_keys, self.output_path, self.objectives, task, identifier)
        return generated_features

    def gen_log(self, config: Configuration, seed: int = RANDOM_SEED):
        random.seed(RANDOM_SEED)
        log = generate_log(config, seed=seed)
        random.seed(RANDOM_SEED)
        result = self.eval_log(log)
        return result

    def eval_log(self, log):
        random.seed(RANDOM_SEED)
        features = compute_features_from_event_data(self.objectives.keys(), log)
        log_evaluation = {}
        for key in self.objectives.keys():
            log_evaluation[key] = abs(self.objectives[key] - features[key])
        return log_evaluation

    def optimize(self, generator_params):
        if generator_params.get(CONFIG_SPACE) is None:
            configspace = ConfigurationSpace({
                "mode": (5, 40),
                "sequence": (0.01, 1),
                "choice": (0.01, 1),
                "parallel": (0.01, 1),
                "loop": (0.01, 1),
                "silent": (0.01, 1),
                "lt_dependency": (0.01, 1),
                "num_traces": (100, 1001),
                "duplicate": (0),
                "or": (0),
            })
            print(f"WARNING: No config_space specified in config file. Continuing with {configspace}")
        else:
            configspace_lists = generator_params[CONFIG_SPACE]
            configspace_tuples = {}
            for k, v in configspace_lists.items():
                if len(v) == 1:
                    configspace_tuples[k] = v[0]
                else:
                    configspace_tuples[k] = tuple(v)
            configspace = ConfigurationSpace(configspace_tuples)

        if generator_params.get(N_TRIALS) is None:
            n_trials = 20
            print(f"INFO: Running with n_trials={n_trials}")
        else:
            n_trials = generator_params[N_TRIALS]

        objectives = [*self.objectives.keys()]

        # Scenario object specifying the multi-objective optimization environment
        scenario = Scenario(
            configspace,
            deterministic=True,
            n_trials=n_trials,
            objectives=objectives,
            n_workers=-1
        )

        # Use SMAC to find the best configuration/hyperparameters
        random.seed(RANDOM_SEED)
        multi_obj = HyperparameterOptimizationFacade.get_multi_objective_algorithm(
                scenario,
                objective_weights=[1]*len(self.objectives),
            )

        random.seed(RANDOM_SEED)
        smac = HyperparameterOptimizationFacade(
            scenario=scenario,
            target_function=self.gen_log,
            multi_objective_algorithm=multi_obj,
            # logging_level=False,
            overwrite=True,
        )

        random.seed(RANDOM_SEED)
        incumbent = smac.optimize()
        return incumbent
