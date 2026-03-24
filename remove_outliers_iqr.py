import numpy as np
def remove_outliers_iqr(data: np.ndarray, k: float = 1.5) -> np.ndarray:
    """
    Удаляет выбросы из одномерного массива методом межквартильного размаха (IQR).

    Параметры:
        data (np.ndarray): Входной одномерный массив данных.
        k (float): Множитель IQR для определения границ выбросов.

    Возвращает:
        np.ndarray: Массив без выбросов.
    """
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    return data[(data >= q1 - k * iqr) & (data <= q3 + k * iqr)]