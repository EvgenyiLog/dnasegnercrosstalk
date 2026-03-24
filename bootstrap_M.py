import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple, Dict, List, Callable
from scipy import stats
import warnings


def bootstrap_M(
    data: Union[np.ndarray, pd.DataFrame],
    n_bootstrap: int = 200,
    confidence_level: float = 0.95,
    estimator: Optional[Callable] = None,
    random_state: Optional[int] = None,
    verbose: bool = True,
    return_all: bool = False,
    **kwargs
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
]:
    """
    Bootstrap-оценка доверительных интервалов для элементов матрицы разделения M.
    
    Функция выполняет непараметрический бутстрап для оценки неопределенности
    элементов матрицы разделения M. Генерирует множество бутстрап-выборок,
    для каждой вычисляет матрицу M, а затем строит доверительные интервалы.
    
    Parameters
    ----------
    data : np.ndarray или pd.DataFrame
        Исходные данные для оценки матрицы M. Может быть:
        - Массивом numpy размером (n_samples, 4)
        - DataFrame с 4 столбцами
        
    n_bootstrap : int, default=200
        Количество бутстрап-итераций. Рекомендуется не менее 1000 для
        стабильных доверительных интервалов, но 200 достаточно для
        быстрой оценки.
        
    confidence_level : float, default=0.95
        Уровень доверия для интервалов. Значение должно быть в диапазоне (0, 1).
        Например, 0.95 соответствует 95% доверительному интервалу.
        
    estimator : callable, optional
        Функция для оценки матрицы M. По умолчанию используется
        estimate_M_from_data. Должна принимать данные и **kwargs.
        
    random_state : int, optional
        Seed для генератора случайных чисел. Используется для воспроизводимости.
        
    verbose : bool, default=True
        Если True, выводит статистику бутстрапа в консоль.
        
    return_all : bool, default=False
        Если True, возвращает все бутстрап-оценки в дополнение к среднему
        и доверительным интервалам.
        
    **kwargs : dict
        Дополнительные параметры, передаваемые в estimator (estimate_M_from_data):
        - min_purity: минимальная чистота пика (default: 0.7)
        - peak_height: высота пика (default: 500)
        - peak_distance: расстояние между пиками (default: 10)
        - peak_prominence: проминенция пика (default: 200)
        - fallback_to_identity: использовать единичную матрицу при отсутствии пиков
        
    Returns
    -------
    M_mean : np.ndarray
        Средняя матрица M по всем бутстрап-выборкам. Размер (4, 4).
        
    M_lo : np.ndarray
        Нижняя граница доверительного интервала. Размер (4, 4).
        
    M_hi : np.ndarray
        Верхняя граница доверительного интервала. Размер (4, 4).
        
    M_all : np.ndarray (если return_all=True)
        Все бутстрап-оценки матриц M. Размер (n_bootstrap, 4, 4).
        
    Notes
    -----
    Алгоритм:
    1. Создает n_bootstrap выборок с заменой из исходных данных
    2. Для каждой выборки оценивает матрицу M с помощью estimator
    3. Вычисляет среднюю матрицу и процентили для доверительных интервалов
    
    Доверительные интервалы вычисляются методом процентилей:
    - Нижняя граница: (1 - confidence_level)/2 процентиль
    - Верхняя граница: (1 + confidence_level)/2 процентиль
    
    Для 95% CI используются 2.5 и 97.5 процентили.
    
    Examples
    --------
    >>> # Базовое использование
    >>> M_mean, M_lo, M_hi = bootstrap_M(
    ...     data=data0,
    ...     n_bootstrap=500,
    ...     min_purity=0.7,
    ...     peak_height=500
    ... )
    M (среднее):
    [[0.8521 0.1234 0.0156 0.0089]
     [0.0891 0.8012 0.0789 0.0308]
     [0.0423 0.0567 0.8723 0.0287]
     [0.0165 0.0187 0.0332 0.9316]]
    
    95% CI ширина:
    [[0.0234 0.0189 0.0123 0.0098]
     [0.0198 0.0212 0.0145 0.0112]
     [0.0156 0.0167 0.0221 0.0109]
     [0.0123 0.0134 0.0156 0.0187]]
    
    >>> # Получение всех бутстрап-оценок
    >>> M_mean, M_lo, M_hi, M_all = bootstrap_M(
    ...     data=data0,
    ...     n_bootstrap=100,
    ...     return_all=True,
    ...     random_state=42
    ... )
    >>> print(f"Форма всех оценок: {M_all.shape}")
    Форма всех оценок: (100, 4, 4)
    
    >>> # Использование с DataFrame
    >>> import pandas as pd
    >>> df = pd.DataFrame(data0.values, columns=['G', 'A', 'T', 'C'])
    >>> M_mean, M_lo, M_hi = bootstrap_M(df, n_bootstrap=300, verbose=False)
    
    References
    ----------
    Efron, B., & Tibshirani, R. J. (1994). An introduction to the bootstrap.
    CRC press.
    """
    
    # Устанавливаем seed для воспроизводимости
    if random_state is not None:
        np.random.seed(random_state)
    
    # Преобразуем входные данные в numpy массив
    if isinstance(data, pd.DataFrame):
        data_array = data.values
    else:
        data_array = np.asarray(data)
    
    # Проверки входных данных
    if data_array.ndim != 2:
        raise ValueError(f"Данные должны быть 2-мерными, получена размерность {data_array.ndim}")
    
    if data_array.shape[1] != 4:
        raise ValueError(f"Данные должны содержать 4 канала, получено {data_array.shape[1]}")
    
    if n_bootstrap < 10:
        warnings.warn(f"n_bootstrap={n_bootstrap} очень мало. Рекомендуется >= 100.", UserWarning)
    
    if not (0 < confidence_level < 1):
        raise ValueError(f"confidence_level должен быть в (0, 1), получено {confidence_level}")
    
    # Определяем estimator по умолчанию
    if estimator is None:
        # Импортируем здесь, чтобы избежать циклических зависимостей
        try:
            from estimate_M_from_data import estimate_M_from_data
            estimator = estimate_M_from_data
        except ImportError:
            # Если функция не найдена, используем заглушку
            def default_estimator(data, **kwargs):
                # Простая заглушка - единичная матрица с небольшим шумом
                M = np.eye(4)
                if kwargs.get('add_noise', False):
                    M += np.random.normal(0, 0.01, (4, 4))
                    M = M / M.sum(axis=0, keepdims=True)
                return M
            estimator = default_estimator
            warnings.warn("estimate_M_from_data не найдена, используется простая заглушка", UserWarning)
    
    n_samples = data_array.shape[0]
    
    # Выполняем бутстрап
    Ms = []
    failed_iterations = 0
    
    if verbose:
        print(f"Выполняется бутстрап с {n_bootstrap} итерациями...")
        from tqdm import tqdm
        iterator = tqdm(range(n_bootstrap), desc="Бутстрап")
    else:
        iterator = range(n_bootstrap)
    
    for b in iterator:
        # Генерируем бутстрап-выборку с заменой
        bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
        bootstrap_data = data_array[bootstrap_indices]
        
        try:
            # Оцениваем матрицу M для бутстрап-выборки
            M_b = estimator(bootstrap_data, **kwargs)
            
            # Проверяем, что M_b имеет правильную форму
            if M_b.shape != (4, 4):
                raise ValueError(f"Estimator вернул матрицу формы {M_b.shape}, ожидалось (4, 4)")
            
            Ms.append(M_b)
            
        except Exception as e:
            failed_iterations += 1
            if verbose and failed_iterations <= 5:  # Показываем первые 5 ошибок
                print(f"Предупреждение: итерация {b} не удалась: {e}")
            continue
    
    if len(Ms) == 0:
        raise RuntimeError("Все бутстрап-итерации завершились ошибкой. Проверьте данные и параметры.")
    
    if failed_iterations > 0 and verbose:
        print(f"Предупреждение: {failed_iterations} из {n_bootstrap} итераций не удались")
    
    # Конвертируем список в массив
    Ms = np.array(Ms)  # (n_successful, 4, 4)
    n_successful = len(Ms)
    
    # Вычисляем статистики
    M_mean = np.mean(Ms, axis=0)
    
    # Вычисляем процентили для доверительных интервалов
    alpha = (1 - confidence_level) / 2
    lower_percentile = alpha * 100
    upper_percentile = (1 - alpha) * 100
    
    M_lo = np.percentile(Ms, lower_percentile, axis=0)
    M_hi = np.percentile(Ms, upper_percentile, axis=0)
    
    # Вычисляем дополнительные статистики
    M_std = np.std(Ms, axis=0)
    M_median = np.median(Ms, axis=0)
    M_cv = M_std / (np.abs(M_mean) + 1e-10)  # Коэффициент вариации
    
    # Выводим результаты
    if verbose:
        print("\n" + "="*60)
        print("РЕЗУЛЬТАТЫ БУТСТРАП-АНАЛИЗА")
        print("="*60)
        print(f"Успешных итераций: {n_successful} / {n_bootstrap}")
        print(f"Уровень доверия: {confidence_level*100:.0f}%")
        print(f"Доверительные интервалы: [{lower_percentile:.1f}%, {upper_percentile:.1f}%]")
        
        print("\nСредняя матрица M:")
        print("-"*60)
        print(pd.DataFrame(np.round(M_mean, 4), 
                          columns=['Канал0', 'Канал1', 'Канал2', 'Канал3'],
                          index=['Краситель0', 'Краситель1', 'Краситель2', 'Краситель3']))
        
        print("\n95% Доверительные интервалы (нижняя граница):")
        print("-"*60)
        print(pd.DataFrame(np.round(M_lo, 4), 
                          columns=['Канал0', 'Канал1', 'Канал2', 'Канал3'],
                          index=['Краситель0', 'Краситель1', 'Краситель2', 'Краситель3']))
        
        print("\n95% Доверительные интервалы (верхняя граница):")
        print("-"*60)
        print(pd.DataFrame(np.round(M_hi, 4), 
                          columns=['Канал0', 'Канал1', 'Канал2', 'Канал3'],
                          index=['Краситель0', 'Краситель1', 'Краситель2', 'Краситель3']))
        
        print("\nШирина доверительных интервалов (M_hi - M_lo):")
        print("-"*60)
        print(pd.DataFrame(np.round(M_hi - M_lo, 4), 
                          columns=['Канал0', 'Канал1', 'Канал2', 'Канал3'],
                          index=['Краситель0', 'Краситель1', 'Краситель2', 'Краситель3']))
        
        print("\nКоэффициент вариации (CV = std/mean):")
        print("-"*60)
        print(pd.DataFrame(np.round(M_cv, 3), 
                          columns=['Канал0', 'Канал1', 'Канал2', 'Канал3'],
                          index=['Краситель0', 'Краситель1', 'Краситель2', 'Краситель3']))
        
        # Оценка стабильности
        max_cv = np.max(M_cv[M_cv < np.inf])  # Игнорируем inf
        if max_cv < 0.1:
            print("\n✓ Оценка стабильна: CV < 0.1 для всех элементов")
        elif max_cv < 0.2:
            print("\n⚠️ Средняя стабильность: CV в диапазоне 0.1-0.2")
        else:
            print("\n✗ Низкая стабильность: CV > 0.2. Увеличьте n_bootstrap или проверьте данные.")
        
        print("="*60)
    
    if return_all:
        return M_mean, M_lo, M_hi, Ms
    
    return M_mean, M_lo, M_hi


