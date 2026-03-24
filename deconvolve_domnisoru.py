import numpy as np
import pandas as pd
def deconvolve_domnisoru(data:pd.DataFrame, M:np.ndarray, estimate_T:bool=True, 
                          baseline_window:int=50, peak_height:int=100):
    """
    Двухшаговая деконволюция по Domnisoru et al. (2000).
    
    Шаг 1: Линейная деконволюция на разностях (обход baseline)
    Шаг 2: Нелинейная коррекция через матрицу T
    """
    columns=data.columns
    index=data.index
    data=data.values
    M_inv = np.linalg.inv(M)
    N = len(data)
    
    # === ШАГ 1: разностная деконволюция ===
    
    # Разности
    delta_I = np.diff(data, axis=0)  # (N-1, 4)
    
    # Деконволюция разностей
    delta_c = (M_inv @ delta_I.T).T  # (N-1, 4)
    
    # Кумулятивное суммирование для восстановления
    c_step1 = np.zeros_like(data)
    c_step1[1:, :] = np.cumsum(delta_c, axis=0)
    c_step1 = np.clip(c_step1, 0, None)
    
    if not estimate_T:
        return c_step1, None
    
    # === ШАГ 2: оценка и применение матрицы T ===
    
    # Определяем межпиковые области
    envelope = c_step1.max(axis=1)
    is_peak_region = envelope > peak_height
    is_baseline_region = ~is_peak_region
    
    # Оцениваем T по корреляциям в межпиковых областях
    baseline_data = c_step1[is_baseline_region, :]
    
    T = np.zeros((4, 4))
    if len(baseline_data) > 50:  # достаточно данных
        for i in range(4):
            for j in range(4):
                if i == j:
                    continue
                # T_ij = регрессионный коэффициент c_i на c_j
                # в межпиковых областях
                if baseline_data[:, j].std() > 0:
                    # Простая линейная регрессия без intercept
                    T[i, j] = (np.sum(baseline_data[:, i] * baseline_data[:, j]) / 
                               np.sum(baseline_data[:, j]**2))
                    T[i, j] = max(0, T[i, j])  # только положительные
    
    # Применяем коррекцию
    correction_matrix = np.eye(4) - T
    c_final = (correction_matrix @ c_step1.T).T
    c_final = np.clip(c_final, 0, None)
    c_final_df = pd.DataFrame(
    c_final, 
    columns=columns,  # используем исходные имена колонок
    index=index       # сохраняем исходный индекс
)
    
    return c_final_df, T