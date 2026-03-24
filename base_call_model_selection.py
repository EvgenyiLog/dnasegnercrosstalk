import numpy as np
import pandas as pd
from scipy.signal import find_peaks 
def base_call_model_selection(data:pd.DataFrame, M:np.ndarray, min_height:int=500, min_distance:int=10):
    """Base calling через model selection (Kheterpal 1998)."""
    bases = []
    qualities = []
    data = data.values
    peak_positions, _ = find_peaks(data, height=min_height, 
                              distance=min_distance)
    for p in peak_positions:
        I_obs = data[p, :]
        best_residual = np.inf
        best_base = -1
        residuals = []
        
        for j in range(4):
            # Модель: только краситель j активен
            # I = c * m_j → c = (m_j^T I) / (m_j^T m_j)
            m_j = M[:, j]
            c = np.dot(m_j, I_obs) / np.dot(m_j, m_j)
            residual = np.sum((I_obs - c * m_j)**2)
            residuals.append(residual)
            
            if residual < best_residual:
                best_residual = residual
                best_base = j
        
        bases.append(best_base)
        
        # Качество: отношение лучшего к второму лучшему
        sorted_res = sorted(residuals)
        if sorted_res[0] > 0:
            ratio = sorted_res[1] / sorted_res[0]
        else:
            ratio = 100
        qualities.append(ratio)
    
    return np.array(bases), np.array(qualities)