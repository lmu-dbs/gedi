import pandas as pd
import pytest
from gedi.run import gedi
from gedi.generator import GenerateEventLogs
from gedi.utils.param_keys.features import FEATURE_SET, FEATURE_PARAMS

def test_GenerateEventLogs():
    INPUT_PARAMS = {'generator_params': {'experiment': {'input_path': 'data/test/grid_feat.csv',
                                                        'objectives': ['ratio_top_20_variants',
                                                                       'epa_normalized_sequence_entropy_linear_forgetting']},
                                         'config_space': {'mode': [5, 20], 'sequence': [0.01, 1],
                                                          'choice': [0.01, 1], 'parallel': [0.01, 1],
                                                          'loop': [0.01, 1], 'silent': [0.01, 1],
                                                          'lt_dependency': [0.01, 1], 'num_traces': [10, 10001],
                                                          'duplicate': [0], 'or': [0]},
                                         'n_trials': 50}}
    VALIDATION_OUTPUT = [0.89, 0.7, 0.89, 1.0]
    genED = GenerateEventLogs(INPUT_PARAMS)
    similarities = [round(experiment['features']['target_similarity'], 2) for experiment in genED.generated_features]

    assert len(similarities) == len(VALIDATION_OUTPUT)
    AGREEMENT_THRESHOLD = 0.75
    agreement = sum(1 for o, u in zip(similarities, VALIDATION_OUTPUT) if o == u) / len(VALIDATION_OUTPUT) * 100
    assert agreement >= AGREEMENT_THRESHOLD

def test_GenerateEventLogs_pypible():
    INPUT_PARAMS = {'generator_params': {'experiment':[
                                            {"ratio_top_20_variants": 0.2, "epa_normalized_sequence_entropy_linear_forgetting": 0.4},
                                            {"ratio_top_20_variants": 0.4, "epa_normalized_sequence_entropy_linear_forgetting": 0.7},
                                            {"epa_normalized_sequence_entropy_linear_forgetting": 0.4},
                                            {"ratio_top_20_variants": 0.2}
                                            ],
                                         'config_space': {'mode': [5, 20], 'sequence': [0.01, 1],
                                                          'choice': [0.01, 1], 'parallel': [0.01, 1],
                                                          'loop': [0.01, 1], 'silent': [0.01, 1],
                                                          'lt_dependency': [0.01, 1], 'num_traces': [10, 10001],
                                                          'duplicate': [0], 'or': [0]},
                                         'n_trials': 50}}
    VALIDATION_OUTPUT = [0.89, 0.7, 0.89, 1.0]
    genED = GenerateEventLogs(INPUT_PARAMS)
    similarities = [round(experiment['features']['target_similarity'], 2) for experiment in genED.generated_features]

    assert len(similarities) == len(VALIDATION_OUTPUT)
    AGREEMENT_THRESHOLD = 0.75
    agreement = sum(1 for o, u in zip(similarities, VALIDATION_OUTPUT) if o == u) / len(VALIDATION_OUTPUT) * 100
    assert agreement >= AGREEMENT_THRESHOLD



def test_abbr_GenerateEventLogs():
    INPUT_PARAMS = {'generator_params': {'experiment': {'input_path': 'data/test/igedi_table_1.csv',
                                                        'objectives': ['rmcv', 'ense']}, 'config_space': {'mode': [5, 20], 'sequence': [0.01, 1], 'choice': [0.01, 1], 'parallel': [0.01, 1], 'loop': [0.01, 1], 'silent': [0.01, 1], 'lt_dependency': [0.01, 1], 'num_traces': [10, 10001], 'duplicate': [0], 'or': [0]}, 'n_trials': 2}}
    VALIDATION_OUTPUT = [0.9, 0.7, 0.7]
    genED = GenerateEventLogs(INPUT_PARAMS)
    similarities = [round(experiment['features']['target_similarity'], 1) for experiment in genED.generated_features]
    assert len(similarities) == len(VALIDATION_OUTPUT)
    AGREEMENT_THRESHOLD = 0.75
    agreement = sum(1 for o, u in zip(similarities, VALIDATION_OUTPUT) if o == u) / len(VALIDATION_OUTPUT) * 100
    assert agreement >= AGREEMENT_THRESHOLD
