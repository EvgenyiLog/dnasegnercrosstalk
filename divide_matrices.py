import numpy as np
from typing import Tuple, Optional

def divide_matrices(
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
    division_type: str = 'right'
) -> np.ndarray:
    """
    Выполняет матричное деление через умножение на обратную матрицу.
    
    Parameters
    ----------
    matrix_a : np.ndarray
        Делимое (числитель).
    matrix_b : np.ndarray
        Делитель (знаменатель).
    division_type : str, optional
        Тип деления:
        - 'right': A * B⁻¹ (правое деление)
        - 'left':  A⁻¹ * B (левое деление)
        По умолчанию 'right'.
    
    Returns
    -------
    np.ndarray
        Результат матричного деления.
    
    Raises
    ------
    ValueError
        - Если division_type не 'right' или 'left'
        - Если матрицы не квадратные
        - Если размерности не подходят для выбранного типа деления
    np.linalg.LinAlgError
        Если матрица-делитель вырождена (необратима).
    
    Examples
    --------
    >>> A = np.array([[4, 1], [1, 5]])
    >>> B = np.array([[2, 1], [1, 3]])
    >>> result = divide_matrices(A, B, 'right')
    >>> print(result)
    [[1.4  0.2 ]
     [0.2  1.6 ]]
    """
    # Проверка типа деления
    if division_type not in ['right', 'left']:
        raise ValueError(
            f"division_type должен быть 'right' или 'left', "
            f"получено '{division_type}'"
        )
    
    # Проверка что матрицы квадратные
    if matrix_a.shape[0] != matrix_a.shape[1]:
        raise ValueError(
            f"Матрица A должна быть квадратной, "
            f"получена размерность {matrix_a.shape}"
        )
    
    if matrix_b.shape[0] != matrix_b.shape[1]:
        raise ValueError(
            f"Матрица B должна быть квадратной, "
            f"получена размерность {matrix_b.shape}"
        )
    
    # Проверка совместимости размерностей
    if division_type == 'right' and matrix_a.shape[1] != matrix_b.shape[1]:
        raise ValueError(
            f"Для правого деления количество столбцов должно совпадать: "
            f"A: {matrix_a.shape[1]}, B: {matrix_b.shape[1]}"
        )
    
    if division_type == 'left' and matrix_a.shape[0] != matrix_b.shape[0]:
        raise ValueError(
            f"Для левого деления количество строк должно совпадать: "
            f"A: {matrix_a.shape[0]}, B: {matrix_b.shape[0]}"
        )
    
    try:
        if division_type == 'right':
            # Правое деление: A * B⁻¹
            return matrix_a @ np.linalg.inv(matrix_b)
        else:
            # Левое деление: A⁻¹ * B
            return np.linalg.inv(matrix_a) @ matrix_b
            
    except np.linalg.LinAlgError as e:
        raise np.linalg.LinAlgError(
            f"Матрица-делитель необратима (вырождена). "
            f"Определитель: {np.linalg.det(matrix_b if division_type == 'right' else matrix_a):.2e}. "
            f"Детали: {str(e)}"
        )