import numpy as np
from numpy.typing import NDArray

def init_M_farthest(
    peak_normalized: NDArray[np.float64],
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

    # 1. стартовый пик (максимальная энергия)
    idx0 = int(np.argmax(peak_normalized.sum(axis=1)))
    selected: list[int] = [idx0]

    # 2. farthest-point selection
    for _ in range(3):
        dists = np.min(
            [
                np.linalg.norm(peak_normalized - peak_normalized[i], axis=1)
                for i in selected
            ],
            axis=0,
        )

        next_idx = int(np.argmax(dists))
        selected.append(next_idx)

    centers = peak_normalized[selected]  # (4, 4)

    return centers.T  # (4, 4) — столбцы = пики