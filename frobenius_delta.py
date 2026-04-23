import numpy as np

def frobenius_delta(M_new:np.ndarray, M_old:np.ndarray):
    """
    Frobenius-норма 
    < 1e-6	сошлось
    1e-6 – 1e-3	почти стабильно
    > 1e-3	ещё идёт обучение
    """
    return np.linalg.norm(M_new - M_old, ord='fro')