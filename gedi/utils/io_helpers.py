import glob
import json
import os
import pandas as pd
import re
import shutil
import numpy as np
from collections import defaultdict
from pathlib import PurePath
from scipy.spatial.distance import euclidean

def select_instance(source_dir, log_path, destination=os.path.join("output","generated","instance_selection")):
    os.makedirs(destination, exist_ok=True)
    try:
        source_path=glob.glob(os.path.join(source_dir, log_path))[0]
        destination_path = os.path.join(destination, "_".join(source_path.rsplit("/")[-2:]))
        shutil.copyfile(source_path, destination_path)
    except IndexError:
        print(f"ERROR: No files found for {source_dir}{log_path}. Continuing.")
    return destination, len(os.listdir(destination))

def read_csvs(input_path, ref_feature):
    f_dict = defaultdict(pd.DataFrame)
    ref_short_name = get_keys_abbreviation([ref_feature])
    for file in glob.glob(f'{input_path}*.csv'):
        if ref_short_name in file[:-4].split(os.sep)[-1].split("_"):
            c_file = pd.read_csv(file, delimiter=",")
            if c_file.columns[0] == 'task':
                c_file = c_file.reindex(columns=[c_file.columns[1], c_file.columns[2], c_file.columns[0]])
                c_file.rename(columns={"task":"log"}, inplace=True)
            f_dict[c_file.columns[0] if c_file.columns[0]!=ref_feature else c_file.columns[1]] = c_file
    return f_dict


def sort_files(data):
    """
    Returns a alphanumeric sortered list
    """
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(data, key=alphanum_key)

def get_keys_abbreviation(obj_keys):
    abbreviated_keys = []
    for obj_key in obj_keys:
        key_slices = obj_key.split("_")
        chars = []
        for key_slice in key_slices:
            for idx, single_char in enumerate(key_slice):
                if idx == 0 or single_char.isdigit():
                    chars.append(single_char)
        abbreviated_key = ''.join(chars)
        abbreviated_keys.append(abbreviated_key)
    return '_'.join(abbreviated_keys)

def get_output_key_value_location(obj, output_path, identifier, obj_keys=None):
    obj_sorted = dict(sorted(obj.items()))
    if obj_keys is None:
        obj_keys = [*obj_sorted.keys()]

    obj_values = [round(x, 4) for x in [*obj_sorted.values()]]

    if len(obj_keys) > 10:
        folder_path = os.path.join(output_path, f"{len(obj_keys)}_features")
        generated_file_name = f"{identifier}"
    else:
        folder_path = os.path.join(output_path, f"{len(obj_keys)}_{get_keys_abbreviation(obj_keys)}")
        obj_values_joined = '_'.join(map(str, obj_values)).replace('.', '')
        generated_file_name = f"{identifier}_{obj_values_joined}"


    os.makedirs(folder_path, exist_ok=True)
    save_path = os.path.join(folder_path, generated_file_name)
    return save_path

def dump_features_json(features: dict, output_path, content_type="features"):
    output_parts = PurePath(output_path.split(".xes")[0]).parts
    features_path = os.path.join(output_parts[0], content_type,
                                   *output_parts[1:])
    json_path = features_path+'.json'

    os.makedirs(os.path.split(json_path)[0], exist_ok=True)
    with open(json_path, 'w') as fp:
        json.dump(features, fp, default=int)
        print(f"SUCCESS: Saved {len(features)-1} {content_type} in {json_path}")#-1 because 'log' is not a feature

def normalize_value(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val) if max_val != min_val else 0.0

def compute_similarity(v1, v2):
    feature_ranges = bpic_feature_values()
    
    # Convert all values to float except for the "log" key
    v1 = {k: (float(v) if k != "log" else v) for k, v in v1.items()}
    v2 = {k: (float(v) if k != "log" else v) for k, v in v2.items()}
    
    # Identify common numeric keys
    common_keys = set(v1.keys()).intersection(set(v2.keys()), set(feature_ranges.keys()))
    numeric_keys = [k for k in common_keys if isinstance(v1[k], (int, float)) and isinstance(v2[k], (int, float))]
    
    if not numeric_keys:
        print("[ERROR]: No common numeric keys found for similarity calculation.")
        return None
    
    # Normalize values and compute differences
    differences = []
    for key in numeric_keys:
        min_val, max_val = feature_ranges[key]
        norm_v1 = normalize_value(v1[key], min_val, max_val)
        norm_v2 = normalize_value(v2[key], min_val, max_val)
        differences.append(abs(norm_v1 - norm_v2))
    
    # Compute average difference as similarity metric
    target_similarity = 1 - np.mean(differences)
    return target_similarity

    
def bpic_feature_values():
    
    data_dict = {
        "n_traces": [226.0, 251734.0],
        "n_variants": [6.0, 28457.0],
        "ratio_variants_per_number_of_traces": [0.0, 1.0],
        "trace_len_min": [1.0, 24.0],
        "trace_len_max": [1.0, 2973.0],
        "trace_len_mean": [1.0, 131.49],
        "trace_len_median": [1.0, 55.0],
        "trace_len_mode": [1.0, 61.0],
        "trace_len_std": [0.0, 202.53],
        "trace_len_variance": [0.0, 41017.89],
        "trace_len_q1": [1.0, 44.0],
        "trace_len_q3": [1.0, 169.0],
        "trace_len_iqr": [0.0, 161.0],
        "trace_len_geometric_mean": [1.0, 53.78],
        "trace_len_geometric_std": [1.0, 5.65],
        "trace_len_harmonic_mean": [1.0, 51.65],
        "trace_len_skewness": [-0.58, 111.97],
        "trace_len_kurtosis": [-0.97, 14006.75],
        "trace_len_coefficient_variation": [0.0, 4.74],
        "trace_len_entropy": [5.33, 12.04],
        "trace_len_hist1": [0.0, 1.99],
        "trace_len_hist2": [0.0, 0.42],
        "trace_len_hist3": [0.0, 0.4],
        "trace_len_hist4": [0.0, 0.19],
        "trace_len_hist5": [0.0, 0.14],
        "trace_len_hist6": [0.0, 10.0],
        "trace_len_hist7": [0.0, 0.02],
        "trace_len_hist8": [0.0, 0.04],
        "trace_len_hist9": [0.0, 0.0],
        "trace_len_hist10": [0.0, 2.7],
        "trace_len_skewness_hist": [-0.58, 111.97],
        "trace_len_kurtosis_hist": [-0.97, 14006.75],
        "ratio_most_common_variant": [0.0, 0.79],
        "ratio_top_1_variants": [0.0, 0.87],
        "ratio_top_5_variants": [0.0, 0.98],
        "ratio_top_10_variants": [0.0, 0.99],
        "ratio_top_20_variants": [0.2, 1.0],
        "ratio_top_50_variants": [0.5, 1.0],
        "ratio_top_75_variants": [0.75, 1.0],
        "mean_variant_occurrence": [1.0, 24500.67],
        "std_variant_occurrence": [0.04, 42344.04],
        "skewness_variant_occurrence": [1.54, 64.77],
        "kurtosis_variant_occurrence": [0.66, 5083.46],
        "n_unique_activities": [1.0, 1152.0],
        "activities_min": [1.0, 66058.0],
        "activities_max": [34.0, 466141.0],
        "activities_mean": [4.13, 66058.0],
        "activities_median": [2.0, 66058.0],
        "activities_std": [0.0, 120522.25],
        "activities_variance": [0.0, 14525612122.34],
        "activities_q1": [1.0, 66058.0],
        "activities_q3": [4.0, 79860.0],
        "activities_iqr": [0.0, 77290.0],
        "activities_skewness": [-0.06, 15.21],
        "activities_kurtosis": [-1.5, 315.84],
        "n_unique_start_activities": [1.0, 809.0],
        "start_activities_min": [1.0, 150370.0],
        "start_activities_max": [27.0, 199867.0],
        "start_activities_mean": [3.7, 150370.0],
        "start_activities_median": [1.0, 150370.0],
        "start_activities_std": [0.0, 65387.49],
        "start_activities_variance": [0.0, 4275524278.19],
        "start_activities_q1": [1.0, 150370.0],
        "start_activities_q3": [4.0, 150370.0],
        "start_activities_iqr": [0.0, 23387.25],
        "start_activities_skewness": [0.0, 9.3],
        "start_activities_kurtosis": [-2.0, 101.82],
        "n_unique_end_activities": [1.0, 757.0],
        "end_activities_min": [1.0, 16653.0],
        "end_activities_max": [28.0, 181328.0],
        "end_activities_mean": [3.53, 24500.67],
        "end_activities_median": [1.0, 16653.0],
        "end_activities_std": [0.0, 42344.04],
        "end_activities_variance": [0.0, 1793017566.89],
        "end_activities_q1": [1.0, 16653.0],
        "end_activities_q3": [3.0, 39876.0],
        "end_activities_iqr": [0.0, 39766.0],
        "end_activities_skewness": [-0.7, 13.82],
        "end_activities_kurtosis": [-2.0, 255.39],
        "eventropy_trace": [0.0, 13.36],
        "eventropy_prefix": [0.0, 16.77],
        "eventropy_global_block": [0.0, 24.71],
        "eventropy_lempel_ziv": [0.0, 685.0],
        "eventropy_k_block_diff_1": [-328.0, 962.0],
        "eventropy_k_block_diff_3": [0.0, 871.0],
        "eventropy_k_block_diff_5": [0.0, 881.0],
        "eventropy_k_block_ratio_1": [0.0, 935.0],
        "eventropy_k_block_ratio_3": [0.0, 7.11],
        "eventropy_k_block_ratio_5": [0.0, 7.11],
        "eventropy_knn_3": [0.0, 8.93],
        "eventropy_knn_5": [0.0, 648.0],
        "eventropy_knn_7": [0.0, 618.0],
        "epa_variant_entropy": [0.0, 11563842.15],
        "epa_normalized_variant_entropy": [0.0, 0.9],
        "epa_sequence_entropy": [0.0, 21146257.12],
        "epa_normalized_sequence_entropy": [0.0, 0.76],
        "epa_sequence_entropy_linear_forgetting": [0.0, 14140225.9],
        "epa_normalized_sequence_entropy_linear_forgetting": [0.0, 0.42],
        "epa_sequence_entropy_exponential_forgetting": [0.0, 15576076.83],
        "epa_normalized_sequence_entropy_exponential_forgetting": [0.0, 0.51]
    }
    
    return data_dict
