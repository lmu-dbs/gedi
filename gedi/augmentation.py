import pandas as pd
from collections import Counter
from datetime import datetime as dt
from imblearn.over_sampling import SMOTE, SVMSMOTE, BorderlineSMOTE, KMeansSMOTE
from sklearn.preprocessing import Normalizer
from gedi.utils.matrix_tools import insert_missing_data
from utils.param_keys import INPUT_PATH, OUTPUT_PATH
from utils.param_keys.augmentation import AUGMENTATION_PARAMS, NO_SAMPLES, FEATURE_SELECTION, METHOD

class InstanceAugmentator:
    def __init__(self, aug_params=None, samples=None):
        if samples is None:
            if aug_params is None:
                    self.new_samples = None
                    self.all = None
                    return
            try:
                samples = pd.read_csv(aug_params[INPUT_PATH], index_col=None)
            except KeyError as e:
                print("ERROR: Specify 'input_path' in config_params/algorithm/*.json containing samples in csv.")
                return
        print("=========================== Instance Augmentator ==========================")

        print(f"INFO: Running with {aug_params}")

        start = dt.now()
        self.output_path = aug_params[OUTPUT_PATH]
        aug_params = aug_params[AUGMENTATION_PARAMS]
        no_samples = aug_params[NO_SAMPLES]
        feature_selection = aug_params[FEATURE_SELECTION]
        feature_selection.insert(0, 'log')

        samples = samples[feature_selection]

        if len(samples.loc[:, samples.isna().any()].columns) > 0:
            imp_df = insert_missing_data(samples)
            print("INFO: Instance Selection:  Before preprocessing:",
                  len(samples.loc[:, samples.isna().any()].columns), "columns in",
                  len(samples.loc[:, samples.isna().any()]), 'rows had null values, after:',
                  len(imp_df.loc[:, imp_df.isna().any()].columns))
        else:
            imp_df = samples.drop(['log'], axis=1)

        # samples = pd.DataFrame(Normalizer().fit_transform(imp_df), columns=imp_df.columns)
        samples = pd.DataFrame(imp_df, columns=imp_df.columns)
        samples['y_el']=samples.apply(lambda x: 1, axis=1)

        artificial_class = pd.DataFrame(0, index=range(len(samples)+100), columns=samples.columns)
        samples = pd.concat([samples, artificial_class]).reset_index(drop=True)

        method = aug_params[METHOD]
        sampling_strat = {1: len(samples[samples["y_el"] == 1]) + no_samples}
        sampler = eval(f'{method}(random_state=42, sampling_strategy={sampling_strat})')

        original_shape = Counter(samples["y_el"])
        #print(len(samples), samples['y_el'], samples.drop(["y_el"], axis=1))
        X_res, y_res = sampler.fit_resample(samples.drop(["y_el"], axis=1), samples["y_el"])
        resample_shape = Counter(y_res)

        self.new_samples = X_res[len(samples):]
        self.all = pd.concat([X_res[:len(samples)-len(artificial_class)], self.new_samples], ignore_index=True, sort=False)
        print(f"SUCCESS: InstanceAugmentator took {dt.now()-start} sec. Original dataset had {original_shape} points. Augmented has {resample_shape} points.")
        print("========================= ~ Instance Augmentation ==========================")
