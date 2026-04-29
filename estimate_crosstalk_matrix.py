import  numpy as np     # инверсия, condition number
from scipy.signal import find_peaks      # поиск пиков
import pandas as pd 
from frobenius_delta import frobenius_delta
from compute_chastity import compute_chastity
from compute_purity import compute_purity
from assignment_change import assignment_change
from  condition_number import  condition_number

def estimate_crosstalk_matrix(data:pd.DataFrame, n_iter:int=30, min_height:int=150, 
                         min_distance:int=10, min_purity:float=0.75,ridge:float=1e-8,
                         init_M=None, verbose:bool=True):
    """
    Итеративная оценка матрицы M по Li & Speed (1999).
    
    Параметры:
        data: (N_scans, 4) — после baseline correction
        n_iter: макс. число итераций
        min_height: порог высоты пика
        min_distance: мин. расстояние между пиками (сканы)
        min_purity: порог чистоты для включения пика
        init_M: начальное приближение (None → единичная)
    """
    # --- Инициализация ---
    if not isinstance(data, pd.DataFrame):
        raise TypeError("raw должен быть pandas DataFrame")
    data=data.values
    M = init_M if init_M is not None else np.eye(4)
    
    # --- Найти все пики (один раз) ---
    envelope = data.max(axis=1)
    peak_pos, _ = find_peaks(envelope, height=min_height, 
                              distance=min_distance)
      
    peak_I = np.clip(data[peak_pos, :], 0, None)# (N_peaks, 4)
    
    # Нормируем пики для M-шага
    norms = peak_I.sum(axis=1, keepdims=True)
    norms[norms == 0] = 1
    peak_normalized = peak_I / norms
    
    if verbose:
        print(f"Найдено пиков: {len(peak_pos)}")

    prev_assignments=None
    
    # --- Итерации ---
    for iteration in range(n_iter):
        M_inv = np.linalg.inv(M)
       
        # E-шаг: деконволюция и назначение
        concentrations = (M_inv @ peak_I.T).T  # (N_peaks, 4)
        purity=compute_purity(concentrations)
        chastity=compute_chastity(concentrations)
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
            
            M_new[:, j] = peak_normalized[mask].mean(axis=0)+ridge
            prev_assignments=assignments
           
        
        # Проверка сходимости
        change = np.abs(M_new - M).max()
        
        frob=frobenius_delta(M_new,M)
        cond=condition_number(M_new)
        if prev_assignments is not None:
            assign_delta=assignment_change(prev_assignments, assignments)
            assign_delta=np.mean(assign_delta)

        else:
            assign_delta=1.0

        M = M_new
       
        
        if verbose and (iteration < 3 or iteration % 5 == 0):
            print(f"  Итерация {iteration+1}: max Δ = {change:.6f}")
            print(f"  Итерация {iteration+1}:  Δassign = {assign_delta:.6f}")
            print(f"  Итерация {iteration+1}:  Δfrob = {frob:.6f}")
            print(f"  Итерация {iteration+1}:  cond = {cond:.6f}")
            print(f"  Итерация {iteration+1}:  mean purity = {purity.mean():.6f}")
            print(f"  Итерация {iteration+1}:  mean chastity= {chastity.mean():.6f}")
        
        if change < 1e-6:
            if verbose:
                print(f"  Сходимость на итерации {iteration+1}")
            break
    
    return M