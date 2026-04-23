import numpy as np

def compute_purity(concentrations:np.ndarray):
    """
    purity
    """
    conc = np.clip(concentrations, 0, None)
    sums = conc.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1
    purity = conc.max(axis=1) / sums[:, 0]
    return purity