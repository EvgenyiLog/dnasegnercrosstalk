import numpy as np

def assignment_change(a_old:np.ndarray, a_new:np.ndarray):
    """
    assignment_change
    < 1%	сошлось
    1–5%	почти
    > 5%	ещё идёт
    """
    return np.mean(a_old != a_new)
