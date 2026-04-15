import numpy as np
from numpy.typing import ArrayLike, NDArray
import warnings

def divide_matrices_np(mat1: ArrayLike, mat2: ArrayLike) -> NDArray[np.float64]:
    """
    Выполняет поэлементное деление двух матриц (или многомерных массивов).

    Каждый элемент `mat1` делится на соответствующий элемент `mat2`
    (mat1[i, j] / mat2[i, j]). Входные структуры обязаны иметь одинаковую форму.

    Parameters
    ----------
    mat1 : ArrayLike
        Массив-делимое. Принимает списки, кортежи, numpy.ndarray и другие
        объекты, поддерживающие протокол array-like.
    mat2 : ArrayLike
        Массив-делитель. Должен совпадать по размерам с `mat1`.

    Returns
    -------
    NDArray[np.float64]
        Новый массив с результатами поэлементного деления. Тип данных автоматически
        приводится к float64 для сохранения точности.

    Raises
    ------
    ValueError
        Если формы `mat1` и `mat2` не совпадают.

    Notes
    -----
    - При наличии нулей в `mat2` NumPy не выбрасывает исключение, а возвращает 
      `inf` или `nan`, выводя `RuntimeWarning`. Для безопасного деления без 
      предупреждений используйте:
      `np.divide(mat1, mat2, out=np.zeros_like(mat1, dtype=float), where=mat2!=0)`
    - Функция не изменяет исходные массивы, возвращает новый объект.

    Examples
    --------
    >>> m1 = [[4.0, 6.0], [8.0, 10.0]]
    >>> m2 = [[2.0, 3.0], [4.0, 5.0]]
    >>> divide_matrices_np(m1, m2)
    array([[2., 2.],
           [2., 2.]])
    """
    a = np.asarray(mat1)
    b = np.asarray(mat2)

    if a.shape != b.shape:
        raise ValueError(
            f"Матрицы должны иметь одинаковые размеры. "
            f"Получено: mat1.shape={a.shape}, mat2.shape={b.shape}"
        )
    
    # Отключаем предупреждения NumPy при делении на 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # where=b!=0 → делим только там, где делитель ≠ 0
        result = np.divide(a, b, out=np.zeros_like(a, dtype=np.float64), where=b != 0)

    return result


