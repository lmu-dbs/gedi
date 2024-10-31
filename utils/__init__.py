from .param_keys.generator import GENERATOR_PARAMS, EXPERIMENT, CONFIG_SPACE, N_TRIALS

def function_name(function: callable):
    return str(function).split()[1]


