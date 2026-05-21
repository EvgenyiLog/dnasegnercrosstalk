import numpy as np
from numpy.typing import NDArray

def init_M_farthest(
    peak_normalized: NDArray[np.float64],n_components:int=4
) -> NDArray[np.float64]:
    """
    Инициализация матрицы M методом farthest-point (max-min spread).

    Выбирает 4 наиболее различающихся (по L2 расстоянию) нормированных пика
    для формирования начальной матрицы перекрёстных помех (crosstalk matrix).

    Алгоритм:
    1. Выбирается первый пик с максимальной суммой компонент.
    2. Итеративно добавляются пики, максимально удалённые от уже выбранных
       (по минимальному расстоянию до множества выбранных).
    3. Возвращается матрица M, где выбранные пики являются столбцами.

    Parameters
    ----------
    peak_normalized : NDArray[np.float64], shape (N_peaks, 4)
        Нормированные пики (каждая строка суммируется к 1).
        Значения должны быть неотрицательными.

    Returns
    -------
    M : NDArray[np.float64], shape (4, 4)
        Начальная матрица M, где столбцы соответствуют выбранным пикам.

    Notes
    -----
    - Метод устойчивее, чем выбор top-k по амплитуде, так как
      избегает линейной зависимости столбцов.
    - Работает в предположении, что чистые пики расположены
      ближе к вершинам симплекса.
    - Используется евклидово расстояние (L2), но можно заменить на cosine.

    Raises
    ------
    ValueError
        Если количество пиков меньше 4.

    Examples
    --------
    >>> peaks = np.random.rand(100, 4)
    >>> peaks /= peaks.sum(axis=1, keepdims=True)
    >>> M = init_M_farthest(peaks)
    >>> M.shape
    (4, 4)
    """
    if peak_normalized.shape[0] < 4:
        raise ValueError("Need at least 4 peaks for initialization")
    N = peak_normalized.shape[0]
    
    # 1. первый — самый "сильный" (можно любой, но так стабильнее)
    norms = np.linalg.norm(peak_normalized, axis=1)
    idx = np.argmax(norms)
    
    selected = [idx]
    
    for _ in range(1, n_components):
        dist = np.min([
            np.linalg.norm(peak_normalized - peak_normalized[j], axis=1) for j in selected
        ], axis=0)
        
        idx = np.argmax(dist)
        selected.append(idx)
    
    M = peak_normalized[selected].T  # (4,4)
    
    return M
