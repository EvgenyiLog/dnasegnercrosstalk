import numpy as np
import pandas as pd

def estimate_M_correlation(data:pd.DataFrame):
    """Оценка M через корреляции (Ye et al. 2010).
    data: (N_clusters_or_scans, 4)"""
    data=data.values
    # Корреляционная матрица
    C = np.corrcoef(data.T)  # (4, 4)
    
    # Нормируем столбцы так, чтобы диагональ была максимальной
    # и сумма столбца = 1
    M = np.abs(C)  # корреляции → положительные
    M = M / M.sum(axis=0, keepdims=True)  # нормировка
    
    return M