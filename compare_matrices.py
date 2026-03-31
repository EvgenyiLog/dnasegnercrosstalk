import numpy as np
from typing import Tuple, Optional

def compare_matrices(
    matrix_a: np.ndarray, 
    matrix_b: np.ndarray,
    tolerance: float = 1e-10
) -> Tuple[np.ndarray, bool]:
    """
    Сравнивает две матрицы поэлементно с заданной точностью.
    
    Parameters
    ----------
    matrix_a : np.ndarray
        Первая матрица для сравнения.
    matrix_b : np.ndarray
        Вторая матрица для сравнения.
    tolerance : float, optional
        Допустимая погрешность для сравнения чисел с плавающей точкой.
        По умолчанию 1e-10.
    
    Returns
    -------
    Tuple[np.ndarray, bool]
        - Логическая матрица, где True означает равенство элементов
        - Общий флаг полного равенства матриц
    
    Raises
    ------
    ValueError
        Если размерности матриц не совпадают.
    
    Examples
    --------
    >>> A = np.array([[1, 2], [3, 4]])
    >>> B = np.array([[1, 2], [3, 5]])
    >>> eq_matrix, is_equal = compare_matrices(A, B)
    >>> print(eq_matrix)
    [[ True  True]
     [ True False]]
    >>> print(is_equal)
    False
    """
    # Проверка размерностей
    if matrix_a.shape != matrix_b.shape:
        raise ValueError(
            f"Размерности матриц не совпадают: "
            f"{matrix_a.shape} vs {matrix_b.shape}"
        )
    
    # Поэлементное сравнение с учетом погрешности
    if np.issubdtype(matrix_a.dtype, np.floating) or np.issubdtype(matrix_b.dtype, np.floating):
        equality_matrix = np.isclose(matrix_a, matrix_b, rtol=tolerance)
    else:
        equality_matrix = (matrix_a == matrix_b)
    
    # Проверка полного равенства
    are_equal = np.all(equality_matrix)
    
    return equality_matrix, are_equal