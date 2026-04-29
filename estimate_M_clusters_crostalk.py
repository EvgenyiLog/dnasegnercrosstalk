from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from scipy.signal import find_peaks    

def estimate_M_clusters_crostalk(data:pd.DataFrame,n_iter:int=30, min_height:int=200, 
                         min_distance:int=10, min_purity:float=0.75,init_M=None, verbose:bool=True):
    """Оценка M через корреляции (Ye et al. 2010).
    data: (N_clusters_or_scans, 4)"""
    data=data.values
    M = init_M if init_M is not None else np.eye(4)
    
    # --- Найти все пики (один раз) ---
    envelope = data.max(axis=1)
    peak_pos, _ = find_peaks(envelope, height=min_height, 
                              distance=min_distance)
     
    peak_I = np.clip(data[peak_pos, :], 0, None) # (N_peaks, 4)
    
    # Нормируем пики для M-шага
    norms = peak_I.sum(axis=1, keepdims=True)
    norms[norms == 0] = 1
    peak_normalized = peak_I / norms
    # KMeans кластеризация
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = kmeans.fit_predict(peak_normalized)
    centers = kmeans.cluster_centers_
    
    # Нормируем центры (чтобы сумма = 1)
    centers = centers / centers.sum(axis=1, keepdims=True)
    M=centers
    if verbose:
        print(f"Найдено пиков: {len(peak_pos)}")


    # --- Итерации ---
    for iteration in range(n_iter):
        M_inv = np.linalg.inv(M)
        
        # E-шаг: деконволюция и назначение
        concentrations = (M_inv @ peak_I.T).T  # (N_peaks, 4)
        assignments = np.argmax(concentrations, axis=1)
        
        # Чистота после деконволюции
        conc_sums = concentrations.clip(0).sum(axis=1)
        conc_sums[conc_sums == 0] = 1
        purities = concentrations.clip(0).max(axis=1) / conc_sums
        
        # M-шаг: обновление столбцов
        M_new = np.zeros((4, 4))
        for j in range(4):
            mask = (assignments == j) & (purities >= min_purity)
            
            if mask.sum() < 3:
                # Недостаточно данных — понижаем порог
                mask = assignments == j
            
            if mask.sum() < 1:
                M_new[:, j] = M[:, j]  # оставляем старый
                continue
            
            M_new[:, j] = peak_normalized[mask].mean(axis=0)
        
        # Проверка сходимости
        change = np.abs(M_new - M).max()
        M = M_new
        
        if verbose and (iteration < 3 or iteration % 5 == 0):
            print(f"  Итерация {iteration+1}: max Δ = {change:.6f}")
        
        if change < 1e-6:
            if verbose:
                print(f"  Сходимость на итерации {iteration+1}")
            break
    
    return M
