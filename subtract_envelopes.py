import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from typing import Tuple

def subtract_envelopes(
    signal_df: pd.DataFrame,
    order: int = 5
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Вычитает верхнюю и нижнюю огибающие из сигналов в DataFrame.
    Огибающие вычисляются по локальным максимумам и минимумам с помощью scipy.signal.argrelextrema,
    затем интерполируются на всю длину сигнала и вычитаются из исходного сигнала.

    Параметры
    ----------
    signal_df : pd.DataFrame
        Входной DataFrame с сигналами. Каждая колонка — отдельный сигнал.
    
    order : int, optional (по умолчанию = 5)
        Минимальное расстояние между соседними экстремумами при поиске максимумов и минимумов.

    Возвращает
    ----------
    sub_low : pd.DataFrame
        Результат вычитания нижней огибающей из сигнала.
    
    sub_high : pd.DataFrame
        Результат вычитания верхней огибающей из сигнала.
    """
    sub_low = pd.DataFrame(index=signal_df.index, columns=signal_df.columns, dtype=float)
    sub_high = pd.DataFrame(index=signal_df.index, columns=signal_df.columns, dtype=float)

    for col in signal_df.columns:
        signal = signal_df[col].values
        x = np.arange(len(signal))

        # Найдём индексы минимумов и максимумов
        min_idx = argrelextrema(signal, np.less, order=order)[0]
        max_idx = argrelextrema(signal, np.greater, order=order)[0]

        # Добавим граничные точки
        if 0 not in min_idx:
            min_idx = np.insert(min_idx, 0, 0)
        if len(signal)-1 not in min_idx:
            min_idx = np.append(min_idx, len(signal)-1)

        if 0 not in max_idx:
            max_idx = np.insert(max_idx, 0, 0)
        if len(signal)-1 not in max_idx:
            max_idx = np.append(max_idx, len(signal)-1)

        # Интерполяция нижней и верхней огибающей
        lower_env = np.interp(x, min_idx, signal[min_idx])
        upper_env = np.interp(x, max_idx, signal[max_idx])

        # Вычитание огибающих
        sub_low[col] = signal - lower_env
        sub_high[col] = signal - upper_env

    return sub_low, sub_high