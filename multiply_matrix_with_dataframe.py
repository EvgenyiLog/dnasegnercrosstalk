import pandas as pd
import numpy as np



def multiply_matrix_with_dataframe(matrix: np.ndarray, df: pd.DataFrame) -> pd.DataFrame:
    """
    Умножает обратную заданной матрицы 4x4 на DataFrame размером 9400x4 и возвращает результат в новом DataFrame
    с теми же названиями колонок.
    
    :param matrix: numpy.ndarray, матрица размером 4x4.
    :param df: pandas.DataFrame, DataFrame размером 9400x4.
    :return: pandas.DataFrame, результат умножения матрицы на DataFrame.
    
    :raises ValueError: если размеры матрицы или DataFrame не соответствуют ожиданиям.
    """
    
    # print(df)
    # Транспонируем DataFrame для изменения его размера на 4x9400
    df_v = df.values  # Теперь размер 4x9400
    # print(transposed_df)

    transposed_df=df_v.T
    # print(type(matrix))
    # Если входные данные - DataFrame
    if isinstance(matrix, pd.DataFrame):
        # Создаем копию, чтобы не изменять исходные данные
        df = matrix.copy()
        
        # Заменяем все нечисловые символы (кроме цифр, точки и минуса)
        df = df.replace(r'[^\d.-]', '', regex=True)
        
        # Конвертируем в float, неконвертируемые значения -> NaN
        df = df.apply(pd.to_numeric, errors='coerce')
        
        # Проверка на наличие NaN
        if df.isna().any().any():
            print("Предупреждение: В матрице есть некорректные значения (заменены на NaN)")
        
        # Получаем numpy array
        matrix = df.values
    else:
        matrix = np.array(matrix)
    try:
        matrix= np.linalg.inv(matrix)
    except:
        matrix=np.linalg.pinv(matrix)
    # Умножаем матрицу на транспонированный DataFrame
    result = matrix @ transposed_df  # Результат будет размером 4x9400
    # result =np.dot(matrix,transposed_df)
    
    # Транспонируем результат обратно в 9400x4
    final_result = result.T  # Теперь размер 9400x4
    final_result = final_result

    # Возвращаем результат в виде DataFrame с оригинальными названиями колонок
    return pd.DataFrame(final_result, columns=df.columns)


import pandas as pd
import numpy as np

def matrix_multiply_with_dataframe(matrix: np.ndarray, df: pd.DataFrame) -> pd.DataFrame:
    """
    Умножает обратную заданной матрицы 4x4 на DataFrame размером 9400x4 и возвращает результат в новом DataFrame
    с теми же названиями колонок.
    
    :param matrix: numpy.ndarray, матрица размером 4x4.
    :param df: pandas.DataFrame, DataFrame размером 9400x4.
    :return: pandas.DataFrame, результат умножения матрицы на DataFrame.
    
    :raises ValueError: если размеры матрицы или DataFrame не соответствуют ожиданиям.
    """
    
    # print(df)
    # Транспонируем DataFrame для изменения его размера на 4x9400
    df_v = df.values  # Теперь размер 4x9400
    # print(transposed_df)

    transposed_df=df_v.T
    
    # Умножаем матрицу на транспонированный DataFrame
    result = matrix @ transposed_df  # Результат будет размером 4x9400
    # result =np.dot(matrix,transposed_df)
    
    # Транспонируем результат обратно в 9400x4
    final_result = result.T  # Теперь размер 9400x4
    final_result = final_result

    # Возвращаем результат в виде DataFrame с оригинальными названиями колонок
    return pd.DataFrame(final_result, columns=df.columns)


def multiply_matrix_with_dataframe_simple(matrix: np.ndarray, df: pd.DataFrame) -> pd.DataFrame:
    """
    Упрощенная версия без лишних преобразований
    """
    # Проверяем размерности
    if matrix.shape != (4, 4):
        raise ValueError(f"Матрица должна быть 4x4, получено {matrix.shape}")
    
    if df.shape[1] != 4:
        raise ValueError(f"DataFrame должен иметь 4 колонки, получено {df.shape[1]}")
    
    # Получаем обратную матрицу
    try:
        inv_matrix = np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        inv_matrix = np.linalg.pinv(matrix)
    
    # Умножаем: (9401x4) @ (4x4).T = (9401x4)
    # Альтернативно: (9401x4) @ inv_matrix.T
    result = df.values @ inv_matrix.T
    
    return pd.DataFrame(result, columns=df.columns, index=df.index)