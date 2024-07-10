from copy import deepcopy
from importlib import reload
from itertools import product
from pylab import *
import itertools
import json
import os
import pandas as pd
import pm4py
import random
import streamlit as st

st.set_page_config(layout='wide')
INPUT_XES="output/inputlog_temp.xes"

"""
# Configuration File fabric for
## GEDI: **G**enerating **E**vent **D**ata with **I**ntentional Features for Benchmarking Process Mining
"""
def double_switch(label_left, label_right):
    # Create two columns for the labels and toggle switch
    col0, col1, col2, col3, col4 = st.columns([4,1, 1, 1,4])

    # Add labels to the columns
    with col1:
        st.write(label_left)

    with col2:
        # Create the toggle switch
        toggle_option = st.toggle(" ",value=False,
            key="toggle_switch_"+label_left,
        )

    with col3:
        st.write(label_right)
    return toggle_option

def view(config_file):
    st.write(config_file)

def get_ranges(stats, tuple_values):
    col_for_row = ", ".join([f"x[\'{i}\'].astype(float)" for i in tuple_values])
    stats['range'] = stats.apply(lambda x: tuple([eval(col_for_row)]), axis=1)
    #tasks = eval(f"list(itertools.product({(parameters*n_para_obj)[:-2]}))")
    result = [f"np.around({x}, 2)" for x in stats['range']]
    result = ", ".join(result)
    return result

def create_objectives_grid(df, objectives, n_para_obj=2):
    parameters_o = "objectives, "
    sel_features = df.index.to_list()
    if n_para_obj==len(objectives):
        parameters = get_ranges(df, sorted(objectives))
        tasks = eval(f"list(itertools.product({parameters}))")[0]
        cartesian_product = list(product(*tasks))
        experiments = [{key: value[idx] for idx, key in enumerate(sel_features)} for value in cartesian_product]
        return experiments
    else:
        if n_para_obj==1:
            experiments = [[exp] for exp in objectives]
        else:
            experiments = eval(f"[exp for exp in list(itertools.product({(parameters_o*n_para_obj)[:-2]})) if exp[0]!=exp[1]]")
        experiments = list(set([tuple(sorted(exp)) for exp in experiments]))
        parameters = "np.around(np.arange(0.0, 1.5,0.5),2), "
        tasks = eval(f"list(itertools.product({(parameters*n_para_obj)[:-2]}))")
    print("TASKS", tasks,  type(parameters), type(n_para_obj), parameters*n_para_obj)
    #print(len(experiments), experiments)

    print(len(tasks))

    for exp in experiments:
        df = pd.DataFrame(data=tasks, columns=["task", *exp])
        #experiment_path = os.path.join('..','data', f'grid_{n_para_obj}obj')
        #os.makedirs(experiment_path, exist_ok=True)
        #experiment_path = os.path.join(experiment_path, f"grid_{len(df.columns)-1}objectives_{abbrev_obj_keys(exp)}.csv") 
        #df.to_csv(experiment_path, index=False)
        #print(f"Saved experiment in {experiment_path}")
        #write_generator_experiment(experiment_path, objectives=exp)

def set_up(generator_params):
    create_button = False
    experiments = []

    col1, col2 = st.columns(2)
    if True:
        grid_option = double_switch("Point-", "Grid-based")
        csv_option = double_switch("Manual", "From CSV")
    if csv_option:
        uploaded_file = st.file_uploader(f"Pick a csv-file containing feature values for features:", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            sel_features = st.multiselect("Selected features", list(df.columns))
            df = df[sel_features]
            if grid_option:
                add_quantile = st.slider('Add %-quantile', min_value=0.0, max_value=100.0, value=50.0, step=5.0)
                stats = df.describe().transpose()
                stats[str(int(add_quantile))+"%"] = df.quantile(q=add_quantile/100)
                view(stats)
                tuple_values = st.multiselect("Tuples including", list(stats.columns)[3:], default=['min', 'max'])
                experiments = create_objectives_grid(stats, tuple_values, n_para_obj=len(tuple_values))
            else:
                view(df)
                experiments = df.to_dict(orient='records')
    else:
        sel_features = st.multiselect("Selected features", list(generator_params['experiment'].keys()))
        for sel_feature in sel_features:
            generator_params['experiment'][sel_feature] = float(st.text_input(sel_feature, generator_params['experiment'][sel_feature]))
    generator_params['experiment'] = experiments
    st.write(f"...result in {len(generator_params['experiment'])} experiments")

    """
    #### Configuration space
    """
    for key in generator_params['config_space'].keys():
        generator_params['config_space'][key] = st.text_input(key, generator_params['config_space'][key])

    #generator_params['config_space'] = st.text_input('config_space', generator_params['config_space'])
    generator_params['n_trials'] = int(st.text_input('n_trials', generator_params['n_trials']))
    return generator_params

if __name__ == '__main__':
    config_layout = json.load(open("config_files/config_layout.json"))
    type(config_layout)
    step_candidates = ["instance_augmentation","event_logs_generation","feature_extraction","benchmark_test"]
    pipeline_steps = st.multiselect(
        "Choose pipeline step",
        step_candidates,
        []
    )
    step_configs = []
    set_col, view_col = st.columns([3, 2])
    for pipeline_step in pipeline_steps:
        step_config = [d for d in config_layout if d['pipeline_step'] == pipeline_step][0]
        with set_col:
            st.header(pipeline_step)
            for step_key in step_config.keys():
                if step_key == "generator_params":
                    st.subheader("Set-up experiments")
                    step_config[step_key] = set_up(step_config[step_key])
                elif step_key != "pipeline_step":
                    step_config[step_key] = st.text_input(step_key, step_config[step_key])
        with view_col:
            view(step_config)
        step_configs.append(step_config)
    config_file = json.dumps(step_configs, indent=4)
    output_path = st.text_input("Output file path", "config_files/experiment_config.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    create_button = st.button("Save config file")
    if create_button:
        with open(output_path, "w") as f:
            f.write(config_file)
