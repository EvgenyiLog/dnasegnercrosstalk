import numpy as np
import pandas as pd 

def subtract_mean_from_first_n(df: pd.DataFrame, n: int = 500) -> pd.DataFrame:
    """
    Вычитает среднее значение из каждой колонки для первых n значений DataFrame.
    
    :param df: pandas.DataFrame, входной DataFrame.
    :param n: int, количество первых значений для вычитания среднего (по умолчанию 1000).
    :return: pandas.DataFrame, DataFrame с вычтенными средними значениями.
    
    :raises ValueError: если n больше количества строк в DataFrame.
    """
    if n > df.shape[0]:
        raise ValueError("n должно быть меньше или равно количеству строк в DataFrame.")
    
    # Вычисление среднего для первых n значений
    mean_values = df.iloc[:n].mean(axis=0)
    
    # Вычитание среднего из всего DataFrame
    result = df - mean_values
    # Подсчет количества отрицательных значений по колонкам
    negative_counts = (result < 0).sum()
    
    print(f"Количество отрицательных значений по колонкам:\n{negative_counts}")
    
    # Сохранение знака исходных значений
    # Сохранение знака исходных значений
    result_signs = np.sign(df)
    
    # Объединяем результат с сохраненными знаками
    result_with_signs = np.abs(result) * result_signs
    
    # Подсчет количества отрицательных значений по колонкам
    negative_counts = (result_with_signs < 0).sum()
    
    print(f"Количество отрицательных значений по колонкам с сохранением знаков:\n{negative_counts}")
    return result_with_signs