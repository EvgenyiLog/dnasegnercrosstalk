import pandas as pd
import numpy as np

def calculate_inverse_matrix(corr_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the inverse of the correlation matrix.

    :param corr_matrix: Correlation matrix as a DataFrame.
    :return: Inverse of the correlation matrix as a DataFrame.
    
    Example:
    >>> corr_matrix = pd.DataFrame([[1, 0.5], [0.5, 1]])
    >>> inv_matrix = calculate_inverse_matrix(corr_matrix)
    """
    return np.linalg.inv(corr_matrix.values)