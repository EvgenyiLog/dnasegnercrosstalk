import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from condition_number import condition_number
from rank_matrix import rank_matrix


def estimate_M_sklearn(data:pd.DataFrame,min_height:int=200,n_iter:int=50, 
                         min_distance:int=10, min_purity:float=0.5 ,eps=1e-12,verbose:bool=True):
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
    peak_I = np.clip(data[peak_pos, :], 0, None) # (N_peaks, 4)
    
    # Нормируем пики для M-шага
    norms = peak_I.sum(axis=1, keepdims=True)
    norms[norms == 0] = 1
    peak_normalized = peak_I / norms
    top_indices = np.argsort(peak_I.sum(axis=1))[-4:]  # индексы 4 самых ярких пиков
    peaks_by_dye = peak_normalized[top_indices]  # (4, 4)

    M = np.zeros((4, 4),dtype=float)
    for j in range(4):
        # пики где канал j доминирует
        idx = np.argsort(peak_I[:, j])[-10:]
    
        Y = peak_normalized[idx]   # (N, 4)
    
        # X = one-hot
        X = np.zeros((len(idx), 4))
        X[:, j] = 1
    
        reg = LinearRegression(fit_intercept=False)
        reg.fit(X, Y)
        reg = Ridge(alpha=1e-3, fit_intercept=False)
        reg.fit(X,Y)
    
        m = reg.coef_[j]  # важный момент!
        m = np.clip(m, 0, None)
    
        M[:, j] = m / m.sum()

    cond=condition_number(M)
    print(f"Число обусловленности: {cond:.2f}")
    

    if verbose:
        print(f"Найдено пиков: {len(peak_pos)}")

    # --- Итерации ---
    for iteration in range(n_iter):
        try:
           M_inv = np.linalg.inv(M)
        except Exception:
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
        cond=condition_number(M_new)
        rankm=rank_matrix(M_new)
        
        
        if verbose and (iteration < 3 or iteration % 5 == 0):
            print(f"  Итерация {iteration+1}: max Δ = {change:.6f}")
            print(f"  Итерация {iteration+1}:  cond = {cond:.6f}")
            print(f"  Итерация {iteration+1}:  rank = {rankm:.6f}")
        
        if verbose and (iteration < 3 or iteration % 5 == 0):
            print(f"  Итерация {iteration+1}: max Δ = {change:.6f}")
        
        if change < 1e-6 or cond>50:
            if verbose:
                print(f"  Сходимость на итерации {iteration+1}")
            break
    return M