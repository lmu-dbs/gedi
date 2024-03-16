import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

def ensure_matrix_symmetry(matrix):
    return 0.5 * (matrix + matrix.T)

def insert_missing_data(matrix):
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp_mean.fit(matrix.drop(['log'], axis=1))
    imp_df = imp_mean.transform(matrix.drop(['log'], axis=1))
    result = pd.DataFrame(imp_df, columns = matrix.columns[1:])
    return result
