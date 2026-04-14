import numpy as np
import pandas as pd
from scipy.signal import find_peaks

def estimate_M_from_clean_peaks(data:pd.DataFrame,min_height:int=200, n_iter:int=50, 
                         min_distance:int=10, min_purity:float=0.5 ,eps=1e-12, verbose:bool=True):
    """
    peaks_by_dye: dict {0:[peak1, peak2, ...], 1:[...], 2:[...], 3:[...]}
    Каждый peak = array-like shape (4,), порядок каналов один и тот же
    Возвращает M shape (4,4), где столбец j соответствует красителю j.
    """
    M = np.zeros((4, 4), dtype=float)
    data=data.values
    
    
    # --- Найти все пики (один раз) ---
    envelope = data.max(axis=1)
    peak_pos, _ = find_peaks(envelope, height=min_height, 
                              distance=min_distance)
    peak_I = data[peak_pos, :]  # (N_peaks, 4)
    
    # Нормируем пики для M-шага
    norms = peak_I.sum(axis=1, keepdims=True)
    norms[norms == 0] = 1
    peak_normalized = peak_I / norms
    top_indices = np.argsort(peak_I.sum(axis=1))[-4:]  # индексы 4 самых ярких пиков
    peaks_by_dye = peak_normalized[top_indices]  # (4, 4)

    for j in range(4):
        peaks = peaks_by_dye[j,:]
        Y = []
        W = []

        for x in peaks:
            x = np.asarray(x, dtype=float)
            s = x.sum()
            if s <= eps:
                continue
            y = x / s
            Y.append(y)
            W.append(s)   # вес: общая интенсивность пика

        if len(Y) == 0:
            M[:, j] = np.eye(4)[:, j]
            continue

        Y = np.vstack(Y)              # (n_peaks, 4)
        W = np.asarray(W)[:, None]    # (n_peaks, 1)

        m = (W * Y).sum(axis=0) / W.sum()
        m = np.clip(m, 0, None)
        m_sum = m.sum()
        if m_sum > eps:
            m /= m_sum
        else:
            m = np.eye(4)[:, j]

        M[:, j] = m

    # --- Итерации ---
    for iteration in range(n_iter):
        M_inv = np.linalg.pinv(M)
        
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