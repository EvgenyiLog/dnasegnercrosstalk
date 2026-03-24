import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from typing import Optional, Dict, Any, List, Tuple


def estimate_M_from_data(
    raw: pd.DataFrame,
    dye_order: List[str],
    min_purity: float = 0.5,
    peak_height: float = 500,
    peak_distance: int = 10,
    peak_prominence: float = 200,
    fallback_to_identity: bool = True
) -> np.ndarray:
    """
    Автоматическая оценка матрицы спектральной разделения M из сырых данных.
    
    Функция идентифицирует пики в каждом канале, где сигнал доминирует
    (чистый пик), и использует их для построения матрицы разделения.
    
    Parameters
    ----------
    raw : pd.DataFrame
        DataFrame с сырыми данными, где строки соответствуют временным точкам,
        а столбцы - каналам детекции. Столбцы должны быть названы в соответствии
        с dye_order.
        
    dye_order : List[str]
        Порядок красителей (столбцов), например ['G', 'A', 'T', 'C'].
        Определяет соответствие каналов нуклеотидам.
        
    min_purity : float, default=0.7
        Минимальная доля сигнала в доминирующем канале для определения "чистого" пика.
        Значение должно быть в диапазоне (0, 1].
        
    peak_height : float, default=500
        Минимальная высота пика для find_peaks.
        Пики ниже этого значения игнорируются.
        
    peak_distance : int, default=10
        Минимальное расстояние между пиками в точках (индексах).
        
    peak_prominence : float, default=200
        Минимальная проминенция (выдающаяся высота) пика.
        Помогает отфильтровать шумовые пики.
        
    fallback_to_identity : bool, default=True
        Если для канала не найдено чистых пиков, использовать единичный вектор
        (диагональный элемент = 1) вместо усредненных значений.
        Если False, для таких каналов будет поднято предупреждение и оставлен 0.
        
    Returns
    -------
    np.ndarray
        Матрица разделения M размером 4x4, где столбцы соответствуют каналам,
        а строки - красителям. M[i, j] показывает вклад красителя j в канал i.
        
    Notes
    -----
    Алгоритм:
    1. Для каждого канала (красителя) ищем пики с заданными параметрами
    2. Проверяем "чистоту" пика: доля сигнала в канале >= min_purity
    3. Для чистых пиков сохраняем нормализованный спектр (вклады во все каналы)
    4. Усредняем все чистые пики для каждого канала
    5. Формируем матрицу M, где столбец j - усредненный спектр для канала j
    
    Examples
    --------
    >>> # Пример использования с данными из секвенатора
    >>> data0 = sdr_channels.loc[:, ['dR110', 'dR6G', 'dTAMRA', 'dROX']]
    >>> data0.columns = ['G', 'A', 'T', 'C']
    >>> M = estimate_M_from_data(
    ...     raw=data0,
    ...     dye_order=['G', 'A', 'T', 'C'],
    ...     min_purity=0.7,
    ...     peak_height=500,
    ...     peak_distance=10,
    ...     peak_prominence=200
    ... )
    >>> print(M.shape)
    (4, 4)
    """
    
    # Проверки входных данных
    if not isinstance(raw, pd.DataFrame):
        raise TypeError("raw должен быть pandas DataFrame")
    
    if len(dye_order) != 4:
        raise ValueError(f"dye_order должен содержать 4 элемента, получено {len(dye_order)}")
    
    if not (0 < min_purity <= 1):
        raise ValueError(f"min_purity должен быть в диапазоне (0, 1], получено {min_purity}")
    
    # Преобразуем в numpy массив для удобства
    raw_array = raw.values if isinstance(raw, pd.DataFrame) else np.array(raw)
    n_channels = raw_array.shape[1]
    
    if n_channels != 4:
        raise ValueError(f"Данные должны содержать 4 канала, получено {n_channels}")
    
    # Словарь для сбора спектров чистых пиков для каждого канала
    pure_spectra: Dict[int, List[np.ndarray]] = {ch: [] for ch in range(4)}
    
    # Для каждого канала ищем пики
    for ch in range(4):
        # Поиск пиков в текущем канале
        peaks, peak_properties = find_peaks(
            raw_array[:, ch],
            height=peak_height,
            distance=peak_distance,
            prominence=peak_prominence
        )
        
        # Анализируем каждый найденный пик
        for peak_idx in peaks:
            intensities = raw_array[peak_idx, :]
            total_intensity = intensities.sum()
            
            if total_intensity <= 0:
                continue
                
            # Нормализованные интенсивности (спектр пика)
            normalized_spectrum = intensities / total_intensity
            
            # Проверяем, доминирует ли текущий канал
            if normalized_spectrum[ch] >= min_purity:
                pure_spectra[ch].append(normalized_spectrum)
    
    # Формируем матрицу M
    M = np.zeros((4, 4))
    
    for ch in range(4):
        if len(pure_spectra[ch]) > 0:
            # Усредняем все чистые спектры для этого канала
            M[:, ch] = np.mean(pure_spectra[ch], axis=0)
        else:
            # Если чистых пиков не найдено
            if fallback_to_identity:
                M[ch, ch] = 1.0
                print(f"Предупреждение: для канала {ch} ({dye_order[ch]}) не найдено чистых пиков. "
                      f"Использован единичный вектор как fallback.")
            else:
                print(f"Предупреждение: для канала {ch} ({dye_order[ch]}) не найдено чистых пиков. "
                      f"Соответствующий столбец матрицы M останется нулевым.")
    
    return M


