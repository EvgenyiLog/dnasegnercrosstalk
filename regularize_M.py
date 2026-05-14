import numpy as np

def regularize_M(M:np.ndarray, reg:float=0.01):
    """Добавляет небольшую добавку к диагонали"""
    M_reg = M + np.eye(4) * reg
    # Перенормируем столбцы
    M_reg = M_reg / M_reg.sum(axis=0, keepdims=True)
    return M_reg

