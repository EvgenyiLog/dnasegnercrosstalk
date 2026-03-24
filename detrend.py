from scipy import signal
import pandas as pd 
import numpy as np
def detrend_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes the trend from each column of the DataFrame.

    :param df: An input DataFrame containing data for trend removal.
    ::return: A new Data Frame with the trend removed from each column.
    """
    # Функция для детрендинга отдельного столбца
    def detrend_column(column):
        try:
            # Удаляем линейный тренд
            detrended = signal.detrend(column.values, type='const')
            # Сдвигаем так, чтобы минимум стал 0
            detrended -= detrended.min()
            return detrended
        except Exception as e:
            print(f"An error occurred during detrending for column {column.name}: {e}")
            return column  # Возвращаем оригинальный столбец в случае ошибки

    # Применяем детрендинг к каждому столбцу
    detrended_df = df.apply(detrend_column)

    return  detrended_df


import pandas as pd
import numpy as np
from scipy import signal
from typing import Union

def detrend_preserve_mean_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Удаляет линейный тренд из каждого столбца и восстанавливает исходный средний уровень.

    Эта функция устраняет линейный дрейф (тренд) с помощью `scipy.signal.detrend`,
    а затем добавляет обратно среднее значение каждого столбца, чтобы сохранить
    физический масштаб данных. Это полезно, когда важно, чтобы сигнал оставался
    в привычном диапазоне значений (например, температура, давление и т.п.).

    :param df: Входной DataFrame, где каждый столбец — временной ряд.
    :type df: pd.DataFrame

    :return: Новый DataFrame с удалённым линейным трендом, но с восстановленным средним уровнем.
             Все столбцы имеют ту же длину и индекс, что и исходный.
    :rtype: pd.DataFrame

    Пример:
        >>> import pandas as pd
        >>> import numpy as np
        >>> t = np.arange(100)
        >>> df = pd.DataFrame({'temp': 0.3 * t + 25 + 2 * np.random.randn(100)})
        >>> df_clean = detrend_preserve_mean_level(df)
        >>> print(df_clean.mean())  # Должно быть близко к 25

    Примечание:
        - Подходит для визуализации и анализа, когда важно сохранить масштаб.
        - Не гарантирует положительность, если исходные данные имели низкий уровень.
    """
    detrended = df.apply(lambda col: signal.detrend(col.values, type='linear'))
    restored = detrended + df.mean()  # Восстанавливаем среднее
    return restored


def detrend_and_shift_positive(df: pd.DataFrame) -> pd.DataFrame:
    """
    Удаляет линейный тренд из каждого столбца и сдвигает данные так, чтобы все значения были >= 0.

    После удаления тренда данные центрируются вокруг нуля. Эта функция дополнительно
    сдвигает каждый столбец вверх на величину его минимального значения, обеспечивая
    неотрицательность. Полезно для алгоритмов, требующих положительных входов,
    или для визуализации.

    :param df: Входной DataFrame с положительными или смешанными значениями.
    :type df: pd.DataFrame

    :return: DataFrame с удалённым трендом и сдвинутыми значениями (все >= 0).
    :rtype: pd.DataFrame

    Пример:
        >>> df = pd.DataFrame({'signal': [10, 12, 14, 16, 18, 20]})
        >>> df_positive = detrend_and_shift_positive(df)
        >>> print(df_positive.min().values)  # [0.]

    Примечание:
        - Все столбцы сдвигаются независимо.
        - Абсолютные значения теряют физический смысл, но сохраняется форма сигнала.
    """
    detrended = df.apply(lambda col: signal.detrend(col.values, type='linear'))
    shifted = detrended - detrended.min()  # Сдвигаем так, чтобы min = 0
    return shifted

def detrend_preserve_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Удаляет только наклон линейного тренда, сохраняя начальное смещение (базовый уровень).

    В отличие от полного детрендинга, эта функция:
        1. Выполняет линейную регрессию: y = a*x + b
        2. Удаляет только наклонную часть (a * x)
        3. Оставляет свободный член (b) — то есть начальный уровень сигнала

    Это позволяет убрать постепенный дрейф, не "проваливая" начало сигнала и не
    уводя его в отрицательную область.

    :param df: Входной DataFrame, где каждый столбец — временной ряд.
    :type df: pd.DataFrame

    :return: DataFrame с удалённым наклоном, но с сохранённым начальным уровнем.
             Форма и индекс совпадают с исходным DataFrame.
    :rtype: pd.DataFrame

    Пример:
        >>> t = np.arange(50)
        >>> df = pd.DataFrame({'voltage': 0.1 * t + 5.0 + 0.5 * np.sin(t)})
        >>> df_clean = detrend_preserve_baseline(df)
        >>> # Сигнал колеблется вокруг уровня ~5.0, без линейного роста

    Примечание:
        - Подходит, если важно сохранить физическое смещение (например, опорное напряжение).
        - Не удаляет постоянное смещение, только наклон.
        - Устойчив к появлению отрицательных значений, если наклон небольшой.
    """
    detrended_df = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
    
    for col in df.columns:
        y = df[col].values
        x = np.arange(len(y))
        
        try:
            # Линейная регрессия: y = a*x + b
            a, b = np.polyfit(x, y, 1)
            
            # Удаляем только наклон, оставляя смещение b
            trend = a * x
            detrended = y - trend
            
            detrended_df[col] = detrended
        except Exception as e:
            print(f"Ошибка при детрендинге столбца {col}: {e}")
            detrended_df[col] = df[col].values  # Сохраняем оригинал при ошибке

    return detrended_df
import pandas as pd
import numpy as np
from typing import Union

def detrend_full(df: pd.DataFrame) -> pd.DataFrame:
    """
    Полностью удаляет линейный тренд (наклон и смещение) из каждого столбца DataFrame.

    Выполняет линейную регрессию y = a*x + b для каждого столбца и вычитает
    полный тренд (a*x + b), оставляя только отклонения от прямой.
    Результат центрирован вокруг нуля.

    :param df: Входной DataFrame, где каждый столбец — временной ряд.
    :type df: pd.DataFrame

    :return: Новый DataFrame с полностью удалённым линейным трендом.
             Все значения — отклонения от линейного приближения.
    :rtype: pd.DataFrame

    Пример:
        >>> t = np.arange(100)
        >>> df = pd.DataFrame({'temp': 0.2 * t + 25 + 2 * np.random.randn(100)})
        >>> df_detrended = detrend_full(df)
        >>> print(df_detrended.mean())  # Должно быть близко к 0.0

    Примечание:
        - Аналогично scipy.signal.detrend(type='linear'), но с явной регрессией.
        - Подходит для анализа остатков, FFT, фильтрации.
        - После детрендинга данные могут быть отрицательными.
    """
    detrended_df = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
    
    for col in df.columns:
        y = df[col].values
        x = np.arange(len(y))  # [0, 1, 2, ..., N-1]
        
        try:
            # Линейная регрессия: y = a*x + b
            a, b = np.polyfit(x, y, 1)
            
            # Полный тренд: и наклон, и смещение
            trend = a * x + b
            
            # Вычитаем весь тренд
            detrended = y - trend
            
            detrended_df[col] = detrended-np.min(y)
        except Exception as e:
            print(f"Ошибка при детрендинге столбца {col}: {e}")
            detrended_df[col] = df[col].values  # Сохраняем оригинал при ошибке
        
    return detrended_df


import pandas as pd
import numpy as np
from scipy import signal

def detrend_df(df: pd.DataFrame, window: int=50) -> pd.DataFrame:
    """
    Removes the trend from each column of the DataFrame using a rolling window.

    :param df: An input DataFrame containing data for trend removal.
    :param window: The size of the rolling window to use for detrending.
    :return: A new DataFrame with the trend removed from each column.
    """
    # Функция для детрендинга отдельного столбца
    def detrend_column(column):
        try:
            # Сохраняем знак оригинальных значений
            sign = np.sign(column.values)
            # Создаем массив для хранения детрендированных значений
            detrended = np.zeros(len(column))
            # Применяем детрендинг с использованием скользящего окна
            for i in range(len(column)):
                # Определяем границы окна
                start = max(0, i - window // 2)
                end = min(len(column), i + window // 2 + 1)
                # Получаем данные в окне
                window_data = column.values[start:end]
                # Выполняем детрендинг на данных в окне
                detrended[i] = signal.detrend(window_data, type='linear')[window // 2]  # Берем центральное значение
            return detrended
        except Exception as e:
            print(f"An error occurred during detrending for column {column.name}: {e}")
            return column.values  # Возвращаем оригинальный массив в случае ошибки

    # Применяем детрендинг к каждому столбцу
    detrended_df = df.apply(detrend_column)

    return detrended_df