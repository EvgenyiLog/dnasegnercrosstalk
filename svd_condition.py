import numpy as np

def svd_condition(M:np.ndarray, target_cond:float=20):
    """Ограничивает число обусловленности через SVD"""
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    
    # Ограничиваем отношение макс/мин сингулярных чисел
    s_max = s.max()
    s_min_desired = s_max / target_cond
    s_regularized = np.maximum(s, s_min_desired)
    
    # Восстанавливаем матрицу
    M_reg = U @ np.diag(s_regularized) @ Vt
    # Нормируем и делаем положительной
    M_reg = np.maximum(M_reg, 0)
    M_reg = M_reg / M_reg.sum(axis=0, keepdims=True)
    return M_reg

