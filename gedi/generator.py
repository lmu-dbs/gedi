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
from pm4py import generate_process_tree
from pm4py import write_xes
from pm4py.sim import play_out
from smac import HyperparameterOptimizationFacade, Scenario
from utils.param_keys import OUTPUT_PATH, INPUT_PATH
from utils.param_keys.generator import GENERATOR_PARAMS, EXPERIMENT, CONFIG_SPACE, N_TRIALS
from gedi.utils.io_helpers import get_output_key_value_location, dump_features_json, read_csvs



"""
   Parameters
    --------------
    parameters
        Parameters of the algorithm, according to the paper:
        - Parameters.MODE: most frequent number of visible activities
        - Parameters.MIN: minimum number of visible activities
        - Parameters.MAX: maximum number of visible activities
        - Parameters.SEQUENCE: probability to add a sequence operator to tree
        - Parameters.CHOICE: probability to add a choice operator to tree
        - Parameters.PARALLEL: probability to add a parallel operator to tree
        - Parameters.LOOP: probability to add a loop operator to tree
        - Parameters.OR: probability to add an or operator to tree
        - Parameters.SILENT: probability to add silent activity to a choice or loop operator
        - Parameters.DUPLICATE: probability to duplicate an activity label
        - Parameters.NO_MODELS: number of trees to generate from model population
"""
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
    # TODO: Clarify nomenclature: experiment, task, objective as in notebook (https://github.com/lmu-dbs/gedi/blob/main/notebooks/grid_objectives.ipynb)
    def __init__(self, params):
        print("=========================== Generator ==========================")
        print(f"INFO: Running with {params}")
        start = dt.now()
        if params.get(OUTPUT_PATH) == None:
            self.output_path = 'data/generated'
        else:
            self.output_path = params.get(OUTPUT_PATH)
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path, exist_ok=True)

        if self.output_path.endswith('csv'):
            self.log_config = pd.read_csv(self.output_path)
            return

        self.params = params.get(GENERATOR_PARAMS)
        experiment = self.params.get(EXPERIMENT)
        if experiment!= None:
            tasks, output_path = get_tasks(experiment, self.output_path)
            self.output_path = output_path

        if tasks is not None:
            num_cores = multiprocessing.cpu_count() if len(tasks) >= multiprocessing.cpu_count() else len(tasks)
            #self.generator_wrapper([*tasks.iterrows()][0])# For testing
            with multiprocessing.Pool(num_cores) as p:
                print(f"INFO: Generator starting at {start.strftime('%H:%M:%S')} using {num_cores} cores for {len(tasks)} tasks...")
                random.seed(RANDOM_SEED)
                log_config = p.map(self.generator_wrapper, tasks.iterrows())
            self.log_config = log_config

        else:
            random.seed(RANDOM_SEED)
            self.configs = self.optimize()
            if type(self.configs) is not list:
                self.configs = [self.configs]
            temp = self.generate_optimized_log(self.configs[0])
            self.log_config = [temp]
            save_path = get_output_key_value_location(self.params[EXPERIMENT],
                                             self.output_path, "genEL")+".xes"
            write_xes(temp['log'], save_path)
            print("SUCCESS: Saved generated event log in", save_path)
        print(f"SUCCESS: Generator took {dt.now()-start} sec. Generated {len(self.log_config)} event logs.")
        print(f"         Saved generated logs in {self.output_path}")
        print("========================= ~ Generator ==========================")

    def generator_wrapper(self, task):
        try:
            identifier = [x for x in task[1] if isinstance(x, str)][0]
        except IndexError:
            identifier = task[0]+1
        task = task[1].loc[lambda x, identifier=identifier: x!=identifier]
        self.objectives = task.to_dict()
        random.seed(RANDOM_SEED)
        self.configs = self.optimize()

        random.seed(RANDOM_SEED)
        if isinstance(self.configs, list):
            log_config = self.generate_optimized_log(self.configs[0])
        else:
            log_config = self.generate_optimized_log(self.configs)

        identifier = 'genEL'+str(identifier)
        save_path = get_output_key_value_location(self.objectives,
                                         self.output_path, identifier)+".xes"

        write_xes(log_config['log'], save_path)
        print("SUCCESS: Saved generated event log in", save_path)
        features_to_dump = log_config['metafeatures']
        features_to_dump['log'] = identifier.replace('genEL', '')
        dump_features_json(features_to_dump, self.output_path, identifier, objectives=self.objectives)
        return log_config

    def generate_optimized_log(self, config):
        ''' Returns event log from given configuration'''
        tree = generate_process_tree(parameters={
            "min": config["mode"],
            "max": config["mode"],
            "mode": config["mode"],
            "sequence": config["sequence"],
            "choice": config["choice"],
            "parallel": config["parallel"],
            "loop": config["loop"],
            "silent": config["silent"],
            "lt_dependency": config["lt_dependency"],
            "duplicate": config["duplicate"],
            "or": config["or"],
            "no_models": 1
        })
        log = play_out(tree, parameters={"num_traces": config["num_traces"]})

        for i, trace in enumerate(log):
            trace.attributes['concept:name']=str(i)
            for j, event in enumerate(trace):
                event['time:timestamp']=dt.now()
        random.seed(RANDOM_SEED)
        metafeatures = self.compute_metafeatures(log)
        return {
            "configuration": config,
            "log": log,
            "metafeatures": metafeatures,
        }

    def gen_log(self, config: Configuration, seed: int = 0):
        random.seed(RANDOM_SEED)
        tree = generate_process_tree(parameters={
            "min": config["mode"],
            "max": config["mode"],
            "mode": config["mode"],
            "sequence": config["sequence"],
            "choice": config["choice"],
            "parallel": config["parallel"],
            "loop": config["loop"],
            "silent": config["silent"],
            "lt_dependency": config["lt_dependency"],
            "duplicate": config["duplicate"],
            "or": config["or"],
            "no_models": 1
        })
        random.seed(RANDOM_SEED)
        log = play_out(tree, parameters={"num_traces": config["num_traces"]})
        random.seed(RANDOM_SEED)
        result = self.eval_log(log)
        return result

    def compute_metafeatures(self, log):
        for i, trace in enumerate(log):
            trace.attributes['concept:name'] = str(i)
            for j, event in enumerate(trace):
                event['time:timestamp'] = dt.fromtimestamp(j * 1000)

        metafeatures_computation = {}
        for ft_name in self.objectives.keys():
            ft_type = feature_type(ft_name)
            metafeatures_computation.update(eval(f"{ft_type}(feature_names=['{ft_name}']).extract(log)"))
        return metafeatures_computation

    def eval_log(self, log):
        random.seed(RANDOM_SEED)
        metafeatures = self.compute_metafeatures(log)
        log_evaluation = {}
        for key in self.objectives.keys():
            log_evaluation[key] = abs(self.objectives[key] - metafeatures[key])
        return log_evaluation

    def optimize(self):
        if self.params.get(CONFIG_SPACE) ==  None:
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
            configspace_lists = self.params[CONFIG_SPACE]
            configspace_tuples = {}
            for k, v in configspace_lists.items():
                if len(v) == 1:
                    configspace_tuples[k] = v[0]
                else:
                    configspace_tuples[k] = tuple(v)
            configspace = ConfigurationSpace(configspace_tuples)

        if self.params.get(N_TRIALS) is None:
            n_trials = 20
            print(f"INFO: Running with n_trials={n_trials}")
        else:
            n_trials = self.params[N_TRIALS]

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
