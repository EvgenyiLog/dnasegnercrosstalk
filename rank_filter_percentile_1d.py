import numpy as np

def rank_filter_percentile_1d(signal: np.ndarray, window: int, percentile: float) -> np.ndarray:
    half_win = window // 2
    filtered = np.empty_like(signal)
    n = len(signal)
    for i in range(n):
        left = max(0, i - half_win)
        right = min(n, i + half_win + 1)
        window_vals = signal[left:right]
        filtered[i] = np.percentile(window_vals, percentile)
    return filtered