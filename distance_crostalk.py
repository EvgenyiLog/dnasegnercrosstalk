import numpy as np
import pandas as pd
from scipy.signal import find_peaks   
from scipy.spatial.distance import pdist

def distance_crostalk(data:pd.DataFrame,n_iter:int=30, min_height:int=200, 
                         min_distance:int=10, min_purity:float=0.75):
    """Оценка M через корреляции (Ye et al. 2010).
    data: (N_clusters_or_scans, 4)"""
    data=data.values
   
    # --- Найти все пики (один раз) ---
    envelope = data.max(axis=1)
    peak_pos, _ = find_peaks(envelope, height=min_height, 
                              distance=min_distance)
    
    peak_I = np.clip(data[peak_pos, :], 0, None) # (N_peaks, 4)
    
    # Нормируем пики для M-шага
    norms = peak_I.sum(axis=1, keepdims=True)
    norms[norms == 0] = 1
    peak_normalized = peak_I / norms
    # 3. Расстояния между КАНАЛАМИ (транспонируем: 4 канала × N_peaks)
    dist_condensed = pdist(peak_normalized.T, metric='euclidean')
    
    # 4. Явное сопоставление парам каналов (порядок pdist для 4 элементов фиксирован)
    channel_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    
    res = pd.DataFrame({
        'Channel_Pair': [f'Ch{i}-Ch{j}' for i, j in channel_pairs],
        'Distance': dist_condensed,
        'N_Peaks': len(peak_pos)
    })
    
    return res