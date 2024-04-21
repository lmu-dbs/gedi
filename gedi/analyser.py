import numpy as np
import warnings

from sklearn.decomposition import FastICA, PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import Normalizer, StandardScaler
from gedi.features import EventLogFeatures
from gedi.plotter import ModelResultPlotter
from gedi.utils.matrix_tools import insert_missing_data
# TODO: Call param_keys explicitly e.g. import INPUT_PATH
from utils.param_keys import *
from utils.param_keys.analyser import MODEL, INPUT_PARAMS, PERPLEXITY


# FUDO: Use this class to compare models during evaluation
class FeatureAnalyser:
    def __init__(self, features, params=None):
        self.features: EventLogFeatures = features
        self.params: dict = {
            PLOT_TYPE: params.get(PLOT_TYPE, COLOR_MAP),
            PLOT_TICS: params.get(PLOT_TICS, True),
            INTERACTIVE: params.get(INTERACTIVE, True),
            N_COMPONENTS: params.get(N_COMPONENTS, 2),
            PERPLEXITY: params.get(PERPLEXITY, 3)
        }
    def compare(self, model_parameter_list: list[dict], plot_results: bool = True) -> list[dict]:
        """
        :param model_parameter_list: list[dict]
            Different model input parameters, saved in a list
        :param plot_results: bool
            Plots the components of the different models (default: True)
            The function can be calculated
        :return: list[dict]
            The results of the models {MODEL, PROJECTION, EXPLAINED_VAR, INPUT_PARAMS}
        """
        model_results = []
        for model_parameters in model_parameter_list:
            try:
                model_results.append(self.get_model_result(model_parameters))
            except np.linalg.LinAlgError as e:
                warnings.warn(f'Eigenvalue decomposition for model `{model_parameters}` could not be calculated:\n {e}')
            except AssertionError as e:
                warnings.warn(f'{e}')

        if plot_results:
            self.compare_with_plot(model_results)

        return model_results

    def compare_with_plot(self, model_results_list):
        """
        This method is used to compare the results in a plot, after fit_transforming different models.
        @param model_results_list: list[dict]
            Different model input parameters, saved in a list.
        """
        ModelResultPlotter().plot_models(
            model_results_list,
            plot_type=self.params[PLOT_TYPE],
            plot_tics=self.params[PLOT_TICS],
            components=self.params[N_COMPONENTS]
        )

    def get_model_result(self, model_parameters: dict, log: bool = True) -> dict:
        """
        Returns a dict of all the important result values. Used for analysing the different models
        :param model_parameters: dict
            The input parameters for the model
        :param log: bool
            Enables the log output while running the program (default: True)
        :return: dict of the results: {MODEL, PROJECTION, EXPLAINED_VAR, INPUT_PARAMS}
        """
        model, projection = self.get_model_and_projection(model_parameters, log=log)
        try:
            ex_var = model.explained_variance_ratio_
        except AttributeError as e:
            warnings.warn(str(e))
            ex_var = 0
        return {MODEL: model, PROJECTION: projection, EXPLAINED_VAR: ex_var, INPUT_PARAMS: model_parameters}

    def get_model_and_projection(self, model_parameters: dict, inp: np.ndarray = None, log: bool = True):
        """
        This method is fitting a model with the given parameters :model_parameters: and
        the inp(ut) data is transformed on the model.
        @param model_parameters: dict
            The input parameters for the model.
        @param inp: np.ndarray
            Input data for the model (optional), (default: None -> calculated on the basis of the model_parameters)
        @param log: bool
            Enables the log output while running the program (default: True)
        @return: fitted model and transformed data
        """
        if log:
            print(f'Running {model_parameters}...')

        if inp is None:
            inp = insert_missing_data(self.features.feat)

        if ALGORITHM_NAME not in model_parameters.keys():
            raise KeyError(f'{ALGORITHM_NAME} is a mandatory model parameter.')

        if model_parameters[ALGORITHM_NAME].startswith('normalized'):
            inp = Normalizer(norm="l2").fit_transform(inp)
        elif model_parameters[ALGORITHM_NAME].startswith('std_scaled'):
            scaler = StandardScaler()
            inp = scaler.fit_transform(inp)
        try:
            if 'pca' in model_parameters[ALGORITHM_NAME]:
                # from sklearn.decomposition import PCA
                pca = PCA(n_components=self.params[N_COMPONENTS])
                # pca = coor.pca(data=inp, dim=self.params[N_COMPONENTS])
                return pca, pca.fit_transform(inp)
            elif 'tsne' in model_parameters[ALGORITHM_NAME]:
                tsne = TSNE(n_components=self.params[N_COMPONENTS], learning_rate='auto',
                            init='random', perplexity=self.params[PERPLEXITY])
                return tsne, tsne.fit_transform(inp)
            #elif model_parameters[ALGORITHM_NAME] == 'original_ica':
            #    ica = FastICA(n_components=self.params[N_COMPONENTS])
            #    return ica, ica.fit_transform(inp)
            else:
                warnings.warn(f'No original algorithm was found with name: {model_parameters[ALGORITHM_NAME]}')
        except TypeError:
            raise TypeError(f'Input data of the function is not correct. '
                            f'Original algorithms take only 2-n-dimensional ndarray')
