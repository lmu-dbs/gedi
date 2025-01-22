from datetime import datetime as dt
from pm4py.sim import play_out

def simulate_log(tree, config, simulation_type = 'play_out'):
    if simulation_type == 'play_out':
        log = play_out(tree, parameters={
            "num_traces": config["num_traces"
            ]})
    else:
        raise Exception("Unknown simulation type")

    for i, trace in enumerate(log):
        trace.attributes['concept:name'] = str(i)
        for j, event in enumerate(trace):
            event['time:timestamp'] = dt.now()
            event['lifecycle:transition'] = "complete"
    return log