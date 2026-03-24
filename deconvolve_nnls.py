import numpy as np 
import pandas as pd
from scipy.optimize import nnls

def deconvolve_nnls(data:pd.DataFrame, M:np.ndarray):
    """NNLS деконволюция — гарантирует неотрицательность."""
    corrected = np.zeros_like(data)
    for i in range(len(data)):
        corrected[i, :], _ = nnls(M, data[i, :])
    return corrected 