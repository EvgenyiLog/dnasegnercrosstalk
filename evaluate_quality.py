import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from typing import Union, Optional, Tuple, Dict, List


def evaluate_quality(
    corrected: Union[np.ndarray, pd.DataFrame],
    min_height: float = 100,
    min_distance: int = 8,
    peak_width: Optional[Tuple[int, int]] = None,
    prominence: Optional[float] = None,
    verbose: bool = True,
    return_stats: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, float]]]:
    """
    Оценка качества спектральной коррекции по метрике чистоты пиков (Li & Speed).
    
    Функция анализирует скорректированные данные, находит пики в огибающей сигнала
    и вычисляет для каждого пика долю доминирующего канала. Чем выше чистота пиков,
    тем лучше качество разделения сигналов.
    
    Parameters
    ----------
    corrected : np.ndarray или pd.DataFrame
        Скорректированные данные размером (n_samples, 4), где каждый столбец
        соответствует каналу детекции (красителю). Может быть numpy массивом
        или DataFrame с 4 столбцами.
        
    min_height : float, default=100
        Минимальная высота пика для обнаружения. Пики ниже этого значения
        игнорируются (считаются шумом).
        
    min_distance : int, default=8
        Минимальное расстояние между пиками в точках (индексах).
        Помогает избежать дублирования пиков.
        
    peak_width : tuple of int, optional
        Диапазон ширины пика в точках, например (1, 20).
        Если указано, учитываются только пики с шириной в этом диапазоне.
        
    prominence : float, optional
        Минимальная проминенция (выдающаяся высота) пика.
        Помогает отфильтровать шумовые пики.
        
    verbose : bool, default=True
        Если True, выводит статистику чистоты пиков в консоль.
        
    return_stats : bool, default=False
        Если True, возвращает кортеж (purities, stats), где stats - словарь
        с дополнительной статистикой.
        
    Returns
    -------
    purities : np.ndarray
        Массив значений чистоты для каждого найденного пика.
        Чистота = max(интенсивности) / сумма(интенсивностей) для пика.
        Значения в диапазоне [0, 1], где 1 - идеально чистый пик.
        
    stats : dict (если return_stats=True)
        Словарь со статистикой:
        - n_peaks: количество найденных пиков
        - mean_purity: средняя чистота
        - median_purity: медианная чистота
        - std_purity: стандартное отклонение чистоты
        - min_purity: минимальная чистота
        - max_purity: максимальная чистота
        - n_purity_90: количество пиков с чистотой > 0.90
        - n_purity_95: количество пиков с чистотой > 0.95
        - n_purity_99: количество пиков с чистотой > 0.99
        
    Notes
    -----
    Алгоритм:
    1. Вычисляем огибающую сигнала как максимум по всем каналам для каждой точки
    2. Находим пики в огибающей с заданными параметрами
    3. Для каждого пика вычисляем чистоту: max(intensity) / sum(intensities)
    4. Агрегируем статистику и выводим результаты
    
    Интерпретация:
    - Чистота > 0.90: отличное качество разделения
    - Чистота > 0.80: хорошее качество разделения
    - Чистота < 0.70: требуется улучшение матрицы разделения
    
    Examples
    --------
    >>> # Базовое использование
    >>> purities = evaluate_quality(corrected_data, min_height=100, min_distance=8)
    Пиков: 42
    Средняя чистота: 0.9432
    Медианная чистота: 0.9512
    Пиков с чистотой > 0.90: 38 / 42
    Пиков с чистотой > 0.95: 31 / 42
    
    >>> # С дополнительной статистикой
    >>> purities, stats = evaluate_quality(corrected_data, return_stats=True, verbose=False)
    >>> print(f"Средняя чистота: {stats['mean_purity']:.3f}")
    Средняя чистота: 0.943
    
    >>> # Использование с DataFrame
    >>> import pandas as pd
    >>> df = pd.DataFrame(corrected_data, columns=['G', 'A', 'T', 'C'])
    >>> purities = evaluate_quality(df, min_height=150, prominence=50)
    
    References
    ----------
    Li, L., & Speed, T. P. (1999). An estimate of the crosstalk matrix in 
    four-dye fluorescence-based DNA sequencing. Electrophoresis, 20(7), 1433-1442.
    """
    
    # Преобразуем входные данные в numpy массив для единообразия
    if isinstance(corrected, pd.DataFrame):
        data_array = corrected.values
    else:
        data_array = np.asarray(corrected)
    
    # Проверки входных данных
    if data_array.ndim != 2:
        raise ValueError(f"Входные данные должны быть 2-мерными, получена размерность {data_array.ndim}")
    
    if data_array.shape[1] != 4:
        raise ValueError(f"Данные должны содержать 4 канала, получено {data_array.shape[1]}")
    
    # Вычисляем огибающую сигнала (максимум по каналам)
    envelope = data_array.max(axis=1)
    
    # Параметры для find_peaks
    peak_params = {
        'height': min_height,
        'distance': min_distance
    }
    
    if peak_width is not None:
        peak_params['width'] = peak_width
    
    if prominence is not None:
        peak_params['prominence'] = prominence
    
    # Находим пики в огибающей
    peaks, peak_properties = find_peaks(envelope, **peak_params)
    
    # Если пики не найдены, возвращаем пустой массив
    if len(peaks) == 0:
        if verbose:
            print("Предупреждение: пики не найдены. Попробуйте уменьшить min_height.")
        
        if return_stats:
            empty_stats = {
                'n_peaks': 0,
                'mean_purity': np.nan,
                'median_purity': np.nan,
                'std_purity': np.nan,
                'min_purity': np.nan,
                'max_purity': np.nan,
                'n_purity_90': 0,
                'n_purity_95': 0,
                'n_purity_99': 0
            }
            return np.array([]), empty_stats
        return np.array([])
    
    # Вычисляем чистоту для каждого пика
    purities = []
    for peak_idx in peaks:
        intensities = data_array[peak_idx, :]
        total_intensity = intensities.sum()
        
        if total_intensity > 0:
            purity = intensities.max() / total_intensity
            purities.append(purity)
        else:
            purities.append(0.0)  # Если сумма нулевая, чистота = 0
    
    purities = np.array(purities)
    
    # Вычисляем статистику
    n_peaks = len(purities)
    mean_purity = np.mean(purities)
    median_purity = np.median(purities)
    std_purity = np.std(purities)
    min_purity = np.min(purities)
    max_purity = np.max(purities)
    
    n_purity_90 = np.sum(purities > 0.90)
    n_purity_95 = np.sum(purities > 0.95)
    n_purity_99 = np.sum(purities > 0.99)
    
    # Выводим результаты, если verbose=True
    if verbose:
        print(f"Пиков: {n_peaks}")
        print(f"Средняя чистота: {mean_purity:.4f}")
        print(f"Медианная чистота: {median_purity:.4f}")
        print(f"Станд. отклонение: {std_purity:.4f}")
        print(f"Диапазон чистоты: [{min_purity:.4f}, {max_purity:.4f}]")
        print(f"Пиков с чистотой > 0.90: {n_purity_90} / {n_peaks}")
        print(f"Пиков с чистотой > 0.95: {n_purity_95} / {n_peaks}")
        print(f"Пиков с чистотой > 0.99: {n_purity_99} / {n_peaks}")
        
        # Добавляем рекомендации
        if mean_purity < 0.7:
            print("\n⚠️  ВНИМАНИЕ: Низкое качество коррекции (<0.7)")
            print("Рекомендации:")
            print("  - Проверьте матрицу разделения M")
            print("  - Увеличьте min_purity в estimate_M_from_data")
            print("  - Проверьте наличие чистых пиков в исходных данных")
        elif mean_purity < 0.8:
            print("\n⚠️  Среднее качество коррекции (0.7-0.8)")
            print("Рекомендации:")
            print("  - Можно улучшить матрицу M, собрав больше чистых пиков")
            print("  - Проверьте параметры find_peaks в оценке M")
        elif mean_purity > 0.9:
            print("\n✓ Отличное качество коррекции (>0.9)")
    
    if return_stats:
        stats = {
            'n_peaks': n_peaks,
            'mean_purity': mean_purity,
            'median_purity': median_purity,
            'std_purity': std_purity,
            'min_purity': min_purity,
            'max_purity': max_purity,
            'n_purity_90': n_purity_90,
            'n_purity_95': n_purity_95,
            'n_purity_99': n_purity_99
        }
        return purities, stats
    
    return purities


