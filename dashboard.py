from copy import deepcopy
from meta_feature_extraction.simple_stats import simple_stats
from meta_feature_extraction.trace_length import trace_length
from meta_feature_extraction.trace_variant import trace_variant
from meta_feature_extraction.activities import activities
from meta_feature_extraction.start_activities import start_activities
from meta_feature_extraction.end_activities import end_activities
from meta_feature_extraction.entropies import entropies
from pm4py import discover_petri_net_inductive as inductive_miner
from pm4py import generate_process_tree
from pm4py import save_vis_petri_net, save_vis_process_tree
from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.algo.simulation.tree_generator import algorithm as tree_generator
from pm4py.algo.simulation.playout.process_tree import algorithm as playout
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.log.util import dataframe_utils
from pm4py.sim import play_out

import matplotlib.image as mpimg
import os
import pandas as pd
import streamlit as st

OUTPUT_PATH = "output"
SAMPLE_EVENTS = 500

@st.cache(allow_output_mutation=True)
def load_from_xes(uploaded_file):
    bytes_data = uploaded_file.getvalue()
    log1 = xes_importer.deserialize(bytes_data)
    get_stats(log1)
    return log1

@st.cache
def load_from_csv(uploaded_file, sep):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, sep=sep, index_col=False)
        return df

def get_stats(log, save=True):
    """Returns the statistics of an event log."""
    num_traces = len(log)
    num_events = sum([len(c) for c in log])
    num_utraces = len(variants_filter.get_variants(log))
    if save:
        st.session_state["num_traces"] = num_traces
        st.session_state["num_events"] = num_events
        st.session_state["num_utraces"] = num_utraces
    return num_utraces, num_traces, num_events

#@st.cache
def df_to_log(df, case_id, activity, timestamp):
    df.rename(columns={case_id: 'case:concept:name',
                       activity: 'concept:name',
                       timestamp: "time:timestamp"}, inplace=True)
    temp = dataframe_utils.convert_timestamp_columns_in_df(df)
    #temp = temp.sort_values(timestamp)
    log = log_converter.apply(temp)
    return log, 'concept:name', "time:timestamp"

def read_uploaded_file(uploaded_file):
    extension = uploaded_file.name.split('.')[-1]
    log_name = uploaded_file.name.split('.')[-2]

    st.sidebar.write("Loaded ", extension.upper(), '-File: ', uploaded_file.name)
    if extension == "xes":
        event_log = load_from_xes(uploaded_file)
        log_columns = [*list(event_log[0][0].keys())]
        convert_button = False
        case_id = "case:concept:name"
        activity = "concept:name"
        timestamp = "time:timestamp"
        default_act_id = log_columns.index("concept:name")
        default_tst_id = log_columns.index("time:timestamp")

        event_df = log_converter.apply(event_log, variant=log_converter.Variants.TO_DATA_FRAME)
        df_path = OUTPUT_PATH+"/"+log_name+".csv"
        event_df.to_csv(df_path, sep =";", index=False)
        return event_log, event_df, case_id, activity

    elif extension == "csv":
        sep = st.sidebar.text_input("Columns separator", ";")
        event_df = load_from_csv(uploaded_file, sep)
        old_df = deepcopy(event_df)
        log_columns = event_df.columns

        case_id = st.sidebar.selectbox("Choose 'case' column:", log_columns)
        activity = st.sidebar.selectbox("Choose 'activity' column:", log_columns, index=0)
        timestamp = st.sidebar.selectbox("Choose 'timestamp' column:", log_columns, index=0)

        convert_button = st.sidebar.button('Confirm selection')
        if convert_button:
            temp = deepcopy(event_df)
            event_log, activity, timestamp = df_to_log(temp, case_id, activity, timestamp)
            #xes_exporter.apply(event_log, INPUT_XES)
            log_columns = [*list(event_log[0][0].keys())]
            st.session_state['log'] = event_log
            return event_log, event_df, case_id, activity

def sample_log_traces(complete_log, sample_size):
    '''
    Samples random traces out of logs.
    So that number of events is slightly over SAMPLE_SIZE.
    :param complete_log: Log extracted from xes
    '''

    log_traces = variants_filter.get_variants(complete_log)
    keys = list(log_traces.keys())
    sample_traces = {}
    num_evs = 0
    while num_evs < sample_size:
        if len(keys) == 0:
            break
        random_trace = keys.pop()
        sample_traces[random_trace] = log_traces[random_trace]
        evs = sum([len(case_id) for case_id in sample_traces[random_trace]])
        num_evs += evs
    log1 = variants_filter.apply(complete_log, sample_traces)
    return log1

def show_process_petrinet(event_log, filter_info, OUTPUT_PATH):
            OUTPUT_PLOT = f"{OUTPUT_PATH}_{filter_info}".replace(":","").replace(".","")+".png" # OUTPUT_PATH is OUTPUT_PATH+INPUT_FILE

            try:
                fig_pt = mpimg.imread(OUTPUT_PLOT)
                st.write("Loaded from memory")
            except FileNotFoundError:
                net, im, fm = inductive_miner(event_log)
                           # parameters={heuristics_miner.Variants.CLASSIC.value.Parameters.DEPENDENCY_THRESH: 0.99,
                           #     pn_visualizer.Variants.FREQUENCY.value.Parameters.FORMAT: "png"})
                #parameters = {pn_visualizer.Variants.FREQUENCY.value.Parameters.FORMAT: "png"}
                save_vis_petri_net(net, im, fm, OUTPUT_PLOT)
                st.write("Saved in: ", OUTPUT_PLOT)
            fig_pt = mpimg.imread(OUTPUT_PLOT)
            st.image(fig_pt)

def show_loaded_event_log(event_log, event_df):
        get_stats(event_log)
        st.write("### Loaded event-log")
        col1, col2 = st.columns(2)
        with col2:
            st.dataframe(event_df)
        with col1:
            show_process_petrinet(event_log, None, OUTPUT_PATH+"running-example")

def extract_meta_features(log, log_name):
    mtf_cols = ["log", "n_traces", "n_unique_traces", "ratio_unique_traces_per_trace", "n_events", "trace_len_min", "trace_len_max",
                "trace_len_mean", "trace_len_median", "trace_len_mode", "trace_len_std", "trace_len_variance", "trace_len_q1",
                "trace_len_q3", "trace_len_iqr", "trace_len_geometric_mean", "trace_len_geometric_std", "trace_len_harmonic_mean",
                "trace_len_skewness", "trace_len_kurtosis", "trace_len_coefficient_variation", "trace_len_entropy", "trace_len_hist1",
                "trace_len_hist2", "trace_len_hist3", "trace_len_hist4", "trace_len_hist5", "trace_len_hist6", "trace_len_hist7",
                "trace_len_hist8", "trace_len_hist9", "trace_len_hist10", "trace_len_skewness_hist", "trace_len_kurtosis_hist",
                "ratio_most_common_variant", "ratio_top_1_variants", "ratio_top_5_variants", "ratio_top_10_variants", "ratio_top_20_variants",
                "ratio_top_50_variants", "ratio_top_75_variants", "mean_variant_occurrence", "std_variant_occurrence", "skewness_variant_occurrence",
                "kurtosis_variant_occurrence", "n_unique_activities", "activities_min", "activities_max", "activities_mean", "activities_median",
                "activities_std", "activities_variance", "activities_q1", "activities_q3", "activities_iqr", "activities_skewness",
                "activities_kurtosis", "n_unique_start_activities", "start_activities_min", "start_activities_max", "start_activities_mean",
                "start_activities_median", "start_activities_std", "start_activities_variance", "start_activities_q1", "start_activities_q3",
                "start_activities_iqr", "start_activities_skewness", "start_activities_kurtosis", "n_unique_end_activities", "end_activities_min",
                "end_activities_max", "end_activities_mean", "end_activities_median", "end_activities_std", "end_activities_variance",
                "end_activities_q1", "end_activities_q3", "end_activities_iqr", "end_activities_skewness", "end_activities_kurtosis", "entropy_trace",
                "entropy_prefix", "entropy_global_block", "entropy_lempel_ziv", "entropy_k_block_diff_1", "entropy_k_block_diff_3",
                "entropy_k_block_diff_5", "entropy_k_block_ratio_1", "entropy_k_block_ratio_3", "entropy_k_block_ratio_5", "entropy_knn_3",
                "entropy_knn_5", "entropy_knn_7"]
    features = [log_name]
    features.extend(simple_stats(log))
    features.extend(trace_length(log))
    features.extend(trace_variant(log))
    features.extend(activities(log))
    features.extend(start_activities(log))
    features.extend(end_activities(log))
    features.extend(entropies(log_name, OUTPUT_PATH))

    mtf = pd.DataFrame([features], columns=mtf_cols)

    st.dataframe(mtf)
    return mtf

def generate_pt(mtf):
    OUTPUT_PLOT = f"{OUTPUT_PATH}/generated_pt".replace(":","").replace(".","")#+".png" # OUTPUT_PATH is OUTPUT_PATH+INPUT_FILE

    st.write("### PT Gen configurations")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
            param_mode = st.text_input('Mode', str(round(mtf['activities_median'].iat[0]))) #?
            st.write("Sum of probabilities must be one")
    with col2:
            param_min = st.text_input('Min', str(mtf['activities_min'].iat[0]))
            param_seq = st.text_input('Probability Sequence', 0.25)
    with col3:
            param_max = st.text_input('Max', str(mtf['activities_max'].iat[0]))
            param_cho = st.text_input('Probability Choice (XOR)', 0.25)
    with col4:
            param_nmo = st.text_input('Number of models', 1)
            param_par = st.text_input('Probability Parallel', 0.25)
    with col5:
            param_dup = st.text_input('Duplicates', 0)
            param_lop = st.text_input('Probability Loop', 0.25)
    with col6:
            param_sil = st.text_input('Silent', 0.2)
            param_or = st.text_input('Probability Or', 0.0)

    PT_PARAMS = {tree_generator.Variants.PTANDLOGGENERATOR.value.Parameters.MODE: round(float(param_mode)), #most frequent number of visible activities
            tree_generator.Variants.PTANDLOGGENERATOR.value.Parameters.MIN: int(param_min), #minimum number of visible activities
            tree_generator.Variants.PTANDLOGGENERATOR.value.Parameters.MAX: int(param_max), #maximum number of visible activities
            tree_generator.Variants.PTANDLOGGENERATOR.value.Parameters.SEQUENCE: float(param_seq), #probability to add a sequence operator to tree
            tree_generator.Variants.PTANDLOGGENERATOR.value.Parameters.CHOICE: float(param_cho), #probability to add a choice (XOR) operator to tree
            tree_generator.Variants.PTANDLOGGENERATOR.value.Parameters.PARALLEL: float(param_par), #probability to add a parallel operator to tree
            tree_generator.Variants.PTANDLOGGENERATOR.value.Parameters.LOOP: float(param_lop), #probability to add a loop operator to tree
            tree_generator.Variants.PTANDLOGGENERATOR.value.Parameters.OR: float(param_or), #probability to add an or operator to tree
            tree_generator.Variants.PTANDLOGGENERATOR.value.Parameters.SILENT: float(param_sil), #probability to add silent activity to a choice or loop operator
            tree_generator.Variants.PTANDLOGGENERATOR.value.Parameters.DUPLICATE: int(param_dup), #probability to duplicate an activity label
            tree_generator.Variants.PTANDLOGGENERATOR.value.Parameters.NO_MODELS: int(param_nmo)} #number of trees to generate from model population

    process_tree = generate_process_tree(parameters=PT_PARAMS)
    save_vis_process_tree(process_tree, OUTPUT_PLOT+"_tree.png")

    st.write("### Playout configurations")

    param_ntraces = st.text_input('Number of traces', str(mtf['n_traces'].iat[0]))
    PO_PARAMS = {playout.Variants.BASIC_PLAYOUT.value.Parameters.NO_TRACES : int(param_ntraces)}

    ptgen_log = play_out(process_tree, parameters=PO_PARAMS)

    net, im, fm = inductive_miner(ptgen_log)
    save_vis_petri_net(net, im, fm, OUTPUT_PLOT+".png")
    st.write("Saved in: ", OUTPUT_PLOT)
    fig_pt_net = mpimg.imread(OUTPUT_PLOT+".png")
    fig_pt_tree = mpimg.imread(OUTPUT_PLOT+"_tree.png")

    fcol1, fcol2 = st.columns(2)
    with fcol1:
        st.image(fig_pt_tree)
    with fcol2:
        st.image(fig_pt_net)
    extract_meta_features(ptgen_log, "gen_pt")


if __name__ == '__main__':
    st.set_page_config(layout='wide')
    """
    # Event Log Generator
    """
    start_options =  ['Event-Log', 'Meta-features']
    start_preference = st.sidebar.selectbox("Do you want to start with a log or with metafeatures?", start_options,0)
    #lets_start = st.sidebar.button("Let's start with "+start_preference+'!')

    if start_preference==start_options[0]:
        st.sidebar.write("Upload a dataset in csv or xes-format:")
        uploaded_file = st.sidebar.file_uploader("Pick a logfile")

        bar = st.progress(0)

        os.makedirs(OUTPUT_PATH, exist_ok=True)
        event_log = st.session_state['log'] if "log" in st.session_state else None
        if uploaded_file:
            event_log, event_df, case_id, activity_id = read_uploaded_file(uploaded_file)
            #event_log = deepcopy(event_log)

            use_sample = st.sidebar.checkbox('Use random sample', True)
            if use_sample:
                sample_size = st.sidebar.text_input('Sample size of approx number of events', str(SAMPLE_EVENTS))
                sample_size = int(sample_size)

                event_log = sample_log_traces(event_log, sample_size)
                sample_cases = [event_log[i].attributes['concept:name'] for i in range(0, len(event_log))]
                event_df = event_df[event_df[case_id].isin(sample_cases)]

            show_loaded_event_log(event_log, event_df)
            ext_mtf = extract_meta_features(event_log, "running-example")
            generate_pt(ext_mtf)

    elif start_preference==start_options[1]:
        LOG_COL = 'log'
        st.sidebar.write("Upload a dataset in csv-format")
        uploaded_file = st.sidebar.file_uploader("Pick a file containing meta-features")

        bar = st.progress(0)

        os.makedirs(OUTPUT_PATH, exist_ok=True)
        event_log = st.session_state[LOG_COL] if "log" in st.session_state else None
        if uploaded_file:
            sep = st.sidebar.text_input("Columns separator", ";")
            mtf = load_from_csv(uploaded_file, sep)
            st.dataframe(mtf)

            log_options = mtf['log'].unique()
            log_preference = st.selectbox("What log should we use for generating a new event-log?", log_options,1)
            mtf_selection = mtf[mtf[LOG_COL]==log_preference]
            generate_pt(mtf_selection)
            st.write("##### Original")
            st.write(mtf_selection)

