import pandas as pd
import numpy as np

def replace_outliers(df: pd.DataFrame, coefficient: float = 15) -> pd.DataFrame:
    """
    Заменяет выбросы в каждом столбце датафрейма с использованием метода межквартильного диапазона (IQR).
    
    Параметры:
    df (pd.DataFrame): Входной датафрейм, в котором будут заменены выбросы.
    coefficient (float): Коэффициент для определения границ выбросов (по умолчанию 1.5).
    
    Возвращает:
    pd.DataFrame: Новый датафрейм с замененными выбросами.
    
    Примечание:
    Эта функция работает только с числовыми столбцами.
    """
    # Копируем датафрейм, чтобы не изменять оригинал
    df_replaced = df.copy()
    
    for column in df_replaced.select_dtypes(include=[np.number]).columns:
        # Вычисляем первый (Q1) и третий (Q3) квартили
        Q1 = df_replaced[column].quantile(0.25)
        Q3 = df_replaced[column].quantile(0.75)
        IQR = Q3 - Q1
        
        # Определяем границы для выбросов
        lower_bound = Q1 - coefficient * IQR
        upper_bound = Q3 + coefficient * IQR
        
        # Заменяем выбросы на границы
        df_replaced[column] = np.where(df_replaced[column] < lower_bound, lower_bound, df_replaced[column])
        df_replaced[column] = np.where(df_replaced[column] > upper_bound, upper_bound, df_replaced[column])
    
    return df_replaced