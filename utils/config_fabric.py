from copy import deepcopy
from importlib import reload
from itertools import product as cproduct
from itertools import combinations
from pylab import *
import itertools
import json
import math
import os
import pandas as pd
import pm4py
import random
import streamlit as st
import subprocess

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

def multi_button(labels):
    cols = st.columns(len(labels))
    activations = []
    for col, label in zip(cols, labels):
        activations.append(col.button(label))
    return activations

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
            sel_features = df.index.to_list()
            parameters_o = "objectives, "
            parameters = get_ranges_from_stats(df, sorted(objectives))
            objectives = sorted(sel_features)
            tasks = f"list(cproduct({parameters}))[0]"

        elif method=="range-from-csv":
            tasks = ""
            for objective in objectives:
                min_col, max_col, step_col = st.columns(3)
                with min_col:
                    selcted_min = st.slider(objective+': min', min_value=float(df[objective].min()), max_value=float(df[objective].max()), value=df[objective].quantile(0.1), step=0.1, key=objective+"min")
                with max_col:
                    selcted_max = st.slider('max', min_value=selcted_min, max_value=float(df[objective].max()), value=df[objective].quantile(0.9), step=0.1, key=objective+"max")
                with step_col:
                    step_value = st.slider('step', min_value=float(df[objective].min()), max_value=float(df[objective].quantile(0.9)), value=df[objective].median()/(df[objective].min()+0.0001), step=0.01, key=objective+"step")
                tasks += f"np.around(np.arange({selcted_min}, {selcted_max}+{step_value}, {step_value}),2), "
        else :#method=="range-manual":
            experitments = []
            tasks=""
            if objectives != None:
                cross_labels =  [feature[0]+': '+feature[1] for feature in list(cproduct(objectives,['min', 'max', 'step']))]
                cross_values = [round(eval(str(combination[0])+combination[1]), 2) for combination in list(cproduct(list(df.values()), ['*1', '*2', '/3']))]
                ranges = zip(objectives, split_list(list(input_multicolumn(cross_labels, cross_values, n_cols=3)), n_para_obj))
                for objective, range_value in ranges:
                    selcted_min, selcted_max, step_value = range_value
                    tasks += f"np.around(np.arange({selcted_min}, {selcted_max}+{step_value}, {step_value}),2), "

        #import pdb; pdb.set_trace()
        cartesian_product = list(cproduct(*eval(tasks)))
        experiments = [{key: value[idx] for idx, key in enumerate(objectives)} for value in cartesian_product]
        return experiments

def set_generator_experiments(generator_params):
    def handle_csv_file(grid_option):
        uploaded_file = st.file_uploader("Pick a csv-file containing feature values for features:", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            sel_features = st.multiselect("Selected features", list(df.columns))
            if sel_features:
                df = df[sel_features]
                return df, sel_features
        return None, None

    def handle_combinatorial(sel_features, stats, tuple_values):
        triangular_option = double_switch("Square", "Triangular")
        if triangular_option:
            experiments = []
            elements = sel_features
            # List to store all combinations
            all_combinations = [combinations(sel_features, r) for r in range(1, len(sel_features) + 1)]
            all_combinations = [comb for sublist in all_combinations for comb in sublist]

            # Print or use the result as needed
            for comb in all_combinations:
                sel_stats = stats.loc[sorted(list(comb))]
                experiments += create_objectives_grid(sel_stats, tuple_values, n_para_obj=len(tuple_values), method="combinatorial")
        else:
            experiments = create_objectives_grid(stats, tuple_values, n_para_obj=len(tuple_values))
        return experiments

    def handle_grid_option(grid_option, df, sel_features):
        if grid_option:
            combinatorial = double_switch("Range", "Combinatorial")
            if combinatorial:
                add_quantile = st.slider('Add %-quantile', min_value=0.0, max_value=100.0, value=50.0, step=5.0)
                stats = df.describe().transpose().sort_index()
                stats[f"{int(add_quantile)}%"] = df.quantile(q=add_quantile / 100)
                st.write(stats)
                tuple_values = st.multiselect("Tuples including", list(stats.columns)[3:], default=['min', 'max'])
                return handle_combinatorial(sel_features, stats, tuple_values)
            else:  # Range
                return create_objectives_grid(df, sel_features, n_para_obj=len(sel_features), method="range-from-csv")
        else:  # Point
            st.write(df)
            return df.to_dict(orient='records')

    def handle_manual_option(sel_features, grid_option):
        if sel_features:
            if grid_option:
                combinatorial = double_switch("Range", "Combinatorial")
                if combinatorial:
                    num_values = st.number_input('How many values to define?', min_value=1, step=1)

                    # Dictionary to store the values for each feature
                    feature_values_dict = {}
                    for feature in sel_features:
                        st.write(f"Define {num_values} values for feature: {feature}")
                        feature_values = []

                        for i in range(int(num_values)):
                            value = st.text_input(f"Value {i+1} for {feature}", key=f"value_{feature}_{i}")
                            feature_values.append(value)
                        
                        # Store the list of values for the feature
                        feature_values_dict[feature] = feature_values

                    # Generate the cartesian product of all possible combinations
                    cartesian_product = list(itertools.product(*feature_values_dict.values()))

                    # Convert each combination into a dictionary with the appropriate feature keys
                    experiments = [dict(zip(sel_features, values)) for values in cartesian_product]
                    
                    return experiments

                else:
                    return create_objectives_grid(generator_params['experiment'], sel_features, n_para_obj=len(sel_features), method="range-manual")
            else:
                experiment = {sel_feature: float(st.text_input(sel_feature, generator_params['experiment'][sel_feature])) for sel_feature in sel_features}
                return [experiment]
        return []


    grid_option, csv_option = double_switch("Point-", "Grid-based", third_label="Manual", fourth_label="From CSV")

    if csv_option:
        df, sel_features = handle_csv_file(grid_option)
        if df is not None and sel_features is not None:
            experiments = handle_grid_option(grid_option, df, sel_features)
        else:
            experiments = []
    else:  # Manual
        sel_features = st.multiselect("Selected features", list(generator_params['experiment'].keys()))
        experiments = handle_manual_option(sel_features, grid_option)

    generator_params['experiment'] = experiments
    st.write(f"...result in {len(generator_params['experiment'])} experiment(s)")

    """
    #### Configuration space
    """
    updated_values = input_multicolumn(generator_params['config_space'].keys(), generator_params['config_space'].values())
    for key, new_value in zip(generator_params['config_space'].keys(), updated_values):
        generator_params['config_space'][key] = eval(new_value)
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
                elif step_key == "feature_params":
                    layout_features = list(step_config[step_key]['feature_set'])
                    step_config[step_key]["feature_set"] = st.multiselect(
                            "features to extract",
                            layout_features)
                elif step_key != "pipeline_step":
                    step_config[step_key] = st.text_input(step_key, step_config[step_key])
        with view_col:
            st.write(step_config)
        step_configs.append(step_config)
    config_file = json.dumps(step_configs, indent=4)
    output_path = st.text_input("Output file path", "config_files/experiment_config.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_labels = ["Save config file", "Save and run config_file"]
    save_labels = ["Save configuration file"]
    #create_button, create_run_button = multi_button(save_labels)
    create_button = multi_button(save_labels)
    if create_button: # or create_run_button:
        with open(output_path, "w") as f:
            f.write(config_file)
        st.write("Saved configuration in ", output_path, ". Run command:")
        #if create_run_button:
        if True:
            options_path = os.path.join("config_files", "options", "baseline.json")
            var = f"python -W ignore main.py -o {options_path} -a {output_path}"
            st.code(var, language='bash')
        if False: #FIXME: Command fails when using multiprocessing 
            command = var.split()

            # Run the command
            result = subprocess.run(command, capture_output=True, text=True)

            if len(result.stderr)==0:
                st.write(result.stdout)
            else:
                st.write("ERROR: ", result.stderr)
