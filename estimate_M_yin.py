import numpy as np
import pandas as pd
from scipy.signal import find_peaks 


def estimate_M_yin(data:pd.DataFrame, min_height:int=500, min_distance:int=10, 
                    min_purity:float=0.7, n_best:int=5):
    """
    Метод Yin et al. (1996): автоматический подбор M.
    
    n_best: сколько лучших пиков усреднять на каждый краситель.
            Yin использовал 1, но усреднение по нескольким устойчивее.
    """
    data=data.values
    envelope = data.max(axis=1)
    peaks, _ = find_peaks(envelope, height=min_height, distance=min_distance)
    
    peak_data = data[peaks, :]  # (N_peaks, 4)
    norms = peak_data.sum(axis=1, keepdims=True)
    norms[norms == 0] = 1
    normalized = peak_data / norms
    
    M = np.zeros((4, 4))
    
    for ch in range(4):
        # Отбираем пики, где канал ch доминирует
        mask = (np.argmax(peak_data, axis=1) == ch) & \
               (normalized[:, ch] >= min_purity)
        
        if mask.sum() == 0:
            # Снижаем порог
            mask = np.argmax(peak_data, axis=1) == ch
        
        if mask.sum() == 0:
            M[ch, ch] = 1.0
            continue
        
        # Сортируем по чистоте, берём лучшие
        candidates = normalized[mask]
        purities = candidates[:, ch]
        best_idx = np.argsort(purities)[-n_best:]
        
        M[:, ch] = candidates[best_idx].mean(axis=0)
    
    return M