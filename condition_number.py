import numpy as np 
def condition_number(M:np.ndarray):
    """
    Condition number
    < 100	отлично
    100–1000	нормально
    1e3–1e4	опасно 
    """
    return np.linalg.cond(M)