from copy import deepcopy
from importlib import reload
from itertools import product, combinations
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

def create_objectives_grid(df, objectives, n_para_obj=2, method="combinatorial"):
        if method=="combinatorial":
        #if n_para_obj==len(objectives):
            sel_features = df.index.to_list()
            parameters_o = "objectives, "
            parameters = get_ranges(df, sorted(objectives))
            tasks = f"list(itertools.product({parameters}))[0]"

        else:
            sel_features = objectives
            tasks = ""
            for objective in objectives:
                min_col, max_col, step_col = st.columns(3)
                with min_col:
                    selcted_min = st.slider(objective+': min', min_value=float(df[objective].min()), max_value=float(df[objective].max()), value=df[objective].quantile(0.1), step=0.1, key=objective+"min")
                with max_col:
                    selcted_max = st.slider('max', min_value=selcted_min, max_value=float(df[objective].max()), value=df[objective].quantile(0.9), step=0.1, key=objective+"max")
                with step_col:
                    step_value = st.slider('step', min_value=float(df[objective].min()), max_value=float(df[objective].quantile(0.9)), value=df[objective].median()/df[objective].min(), step=0.01, key=objective+"step")
                tasks += f"np.around(np.arange({selcted_min}, {selcted_max}+{step_value}, {step_value}),2), "

        cartesian_product = list(product(*eval(tasks)))
        experiments = [{key: value[idx] for idx, key in enumerate(sel_features)} for value in cartesian_product]
        return experiments

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
            if sel_features:
                df = df[sel_features]
                if grid_option:
                    combinatorial = double_switch("Range", "Combinatorial")
                    if combinatorial:
                        add_quantile = st.slider('Add %-quantile', min_value=0.0, max_value=100.0, value=50.0, step=5.0)
                        stats = df.describe().transpose()
                        stats[str(int(add_quantile))+"%"] = df.quantile(q=add_quantile/100)
                        view(stats)
                        tuple_values = st.multiselect("Tuples including", list(stats.columns)[3:], default=['min', 'max'])
                        triangular_option = double_switch("Square", "Triangular")
                        if triangular_option:
                            elements = sel_features
                            # List to store all combinations
                            all_combinations = []

                            # Generate combinations of length 1, 2, and 3
                            for r in range(1, len(elements) + 1):
                                # Generate combinations of length r
                                combinations_r = list(combinations(elements, r))
                                # Extend the list of all combinations
                                all_combinations.extend(combinations_r)
                            # Print or use the result as needed
                            for comb in all_combinations:
                                sel_stats = stats.loc[list(comb)]
                                experiments += create_objectives_grid(sel_stats, tuple_values, n_para_obj=len(tuple_values))
                        else:
                            experiments = create_objectives_grid(stats, tuple_values, n_para_obj=len(tuple_values))
                    else:
                        experiments = create_objectives_grid(df, sel_features, n_para_obj=len(sel_features), method="range")
                else:
                    view(df)
                    experiments = df.to_dict(orient='records')
    else:
        sel_features = st.multiselect("Selected features", list(generator_params['experiment'].keys()))
        if sel_features != None:
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
