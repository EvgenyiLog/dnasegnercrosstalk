import numpy as np

def compute_chastity(concentrations:np.ndarray):
    """
    chastity
    """
    conc = np.clip(concentrations, 0, None)
    sorted_vals = np.sort(conc, axis=1)
    I1 = sorted_vals[:, -1]
    I2 = sorted_vals[:, -2]
    denom = I1 + I2
    denom[denom == 0] = 1
    return I1 / denom