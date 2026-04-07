def normalize_diagonal(matrix):
    """
    Приводит матрицу к виду с единицами на главной диагонали
    """
    result = matrix.copy().astype(float)
    n = len(result)
    
    for i in range(n):
        divisor = result[i, i]
        if divisor != 0:
            result[i] = result[i] / divisor
        else:
            print(f"Предупреждение: нулевой элемент на диагонали в строке {i}")
    
    return result