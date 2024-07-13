from copy import deepcopy
from importlib import reload
from itertools import product, combinations
from pylab import *
import itertools
import json
import math
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
def double_switch(label_left, label_right, third_label=None, fourth_label=None):
    if third_label==None and fourth_label==None:
        # Create two columns for the labels and toggle switch
        col0, col1, col2, col3, col4 = st.columns([2,1,1,1,2])
    else:
        # Create two columns for the labels and toggle switch
        col0, col1, col2, col3, col4, col5, col6, col7, col8 = st.columns([1,1,1,1,1,1,1,1,1])

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
    if third_label is None and fourth_label is None:return toggle_option
    else:
        with col5:
            st.write(third_label)

        with col6:
            # Create the toggle switch
            toggle_option_2 = st.toggle(" ",value=False,
                key="toggle_switch_"+third_label,
            )

        with col7:
            st.write(fourth_label)
        return toggle_option, toggle_option_2

def input_multicolumn(labels, default_values, n_cols=5):
    result = {}
    cols = st.columns(n_cols)
    factor = math.ceil(len(labels)/n_cols)
    extended = cols.copy()
    for _ in range(factor):
        extended.extend(cols)
    for label, default_value, col in zip(labels, default_values, extended):
        with col:
            result[label] = col.text_input(label, default_value, key=f"input_"+label+'_'+str(default_value))
    return result.values()

def split_list(input_list, n):
    # Calculate the size of each chunk
    k, m = divmod(len(input_list), n)
    # Use list comprehension to create n sublists
    return [input_list[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

def get_ranges_from_stats(stats, tuple_values):
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
            parameters = get_ranges_from_stats(df, sorted(objectives))
            tasks = f"list(itertools.product({parameters}))[0]"

        elif method=="range-from-csv":
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
        else :#method=="range-manual":
            experitments = []
            tasks=""
            if objectives != None:
                cross_labels =  [feature[0]+': '+feature[1] for feature in list(product(objectives,['min', 'max', 'step']))]
                cross_values = [round(eval(str(combination[0])+combination[1]), 2) for combination in list(product(list(df.values()), ['*1', '*2', '/3']))]
                ranges = zip(objectives, split_list(list(input_multicolumn(cross_labels, cross_values, n_cols=3)), n_para_obj))
                for objective, range_value in ranges:
                    selcted_min, selcted_max, step_value = range_value
                    tasks += f"np.around(np.arange({selcted_min}, {selcted_max}+{step_value}, {step_value}),2), "

        cartesian_product = list(product(*eval(tasks)))
        experiments = [{key: value[idx] for idx, key in enumerate(objectives)} for value in cartesian_product]
        return experiments

def set_generator_experiments(generator_params):
    create_button = False
    experiments = []

    grid_option, csv_option = double_switch("Point-", "Grid-based", third_label="Manual", fourth_label="From CSV")
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
                        st.write(stats)
                        tuple_values = st.multiselect("Tuples including", list(stats.columns)[3:], default=['min', 'max'])
                        triangular_option = double_switch("Square", "Triangular")
                        if triangular_option:
                            elements = sel_features
                            # List to store all combinations
                            all_combinations = []

                            # Generate combinations of length 1, 2, ... and len(elements)
                            for r in range(1, len(elements) + 1):
                                # Generate combinations of length r
                                combinations_r = list(combinations(elements, r))
                                # Extend the list of all combinations
                                all_combinations.extend(combinations_r)
                            # Print or use the result as needed
                            for comb in all_combinations:
                                sel_stats = stats.loc[list(comb)]
                                experiments += create_objectives_grid(sel_stats, tuple_values, n_para_obj=len(tuple_values), method="combinatorial")
                        else: #Square
                            experiments = create_objectives_grid(stats, tuple_values, n_para_obj=len(tuple_values), method="combinatorial")
                    else: #Range
                        experiments = create_objectives_grid(df, sel_features, n_para_obj=len(sel_features), method="range-from-csv")
                else: #Point
                    st.write(df)
                    experiments = df.to_dict(orient='records')
    #Manual
    else:
        sel_features = st.multiselect("Selected features", list(generator_params['experiment'].keys()))
        experitments = []
        if sel_features != None:
            if grid_option:
                experiments = create_objectives_grid(generator_params['experiment'], sel_features, n_para_obj=len(sel_features), method="range-manual")
            else:
                experiment = {}
                for sel_feature in sel_features:
                    experiment[sel_feature] = float(st.text_input(sel_feature, generator_params['experiment'][sel_feature]))
                experiments.append(experiment)
    generator_params['experiment'] = experiments
    st.write(f"...result in {len(generator_params['experiment'])} experiment(s)")

    """
    #### Configuration space
    """
    updated_values = input_multicolumn(generator_params['config_space'].keys(), generator_params['config_space'].values())
    for key, new_value in zip(generator_params['config_space'].keys(), updated_values):
        generator_params['config_space'][key] = new_value
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
                    step_config[step_key] = set_generator_experiments(step_config[step_key])
                elif step_key != "pipeline_step":
                    step_config[step_key] = st.text_input(step_key, step_config[step_key])
        with view_col:
            st.write(step_config)
        step_configs.append(step_config)
    config_file = json.dumps(step_configs, indent=4)
    output_path = st.text_input("Output file path", "config_files/experiment_config.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    create_button = st.button("Save config file")
    if create_button:
        with open(output_path, "w") as f:
            f.write(config_file)
