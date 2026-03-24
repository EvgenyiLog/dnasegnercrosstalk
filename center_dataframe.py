import pandas as pd
import numpy as np
from typing import Optional, Union

def center_dataframe(
    df: pd.DataFrame,
    method: str = 'mean',
    percentile: float = 50.0,
    inplace: bool = False
) -> Optional[pd.DataFrame]:
    """
    Центрирует данные в DataFrame, вычитая по каждой колонке либо среднее, 
    либо указанный перцентиль (например, медиану для 50-го перцентиля).

    Параметры:
    -----------
    df : pd.DataFrame
        Входной DataFrame с числовыми данными для центрирования.
        
    method : {'mean', 'percentile'}, default='mean'
        Метод центрирования:
        - 'mean': вычитает среднее значение
        - 'percentile': вычитает указанный перцентиль
        
    percentile : float, default=50.0
        Значение перцентиля для вычитания (0-100). Используется только при method='percentile'.
        Например, 50.0 соответствует медиане.
        
    inplace : bool, default=False
        Если True, изменяет исходный DataFrame. Если False, возвращает новый DataFrame.

    Возвращает:
    -----------
    pd.DataFrame или None:
        - Если inplace=True, возвращает None (изменяет исходный df)
        - Если inplace=False, возвращает новый центрированный DataFrame

    Примечания:
    -----------
    1. Для корреляционного анализа, PCA и других статистических методов 
       рекомендуется использовать method='mean'
    2. Перцентиль может быть полезен для робастного центрирования при наличии выбросов
    3. Функция автоматически пропускает нечисловые колонки

    Примеры:
    --------
    # Центрирование по среднему
    df_centered = center_dataframe(df, method='mean')
    
    # Центрирование по медиане (50-й перцентиль)
    center_dataframe(df, method='percentile', percentile=50, inplace=True)
    
    # Центрирование по 25-му перцентилю
    df_q25 = center_dataframe(df, method='percentile', percentile=25)
    """
    
    # Валидация параметров
    if method not in ['mean', 'percentile']:
        raise ValueError("method должен быть 'mean' или 'percentile'")
    
    if not 0 <= percentile <= 100:
        raise ValueError("percentile должен быть в диапазоне [0, 100]")
    
    # Работа с копией данных если не inplace
    if not inplace:
        df = df.copy()
    
    # Выбираем только числовые колонки
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Вычисляем значения для вычитания
    if method == 'mean':
        center_values = df[numeric_cols].mean()
    else:
        center_values = df[numeric_cols].quantile(percentile / 100.0)
    
    # Применяем центрирование
    df[numeric_cols] = df[numeric_cols] - center_values
    
    # Возвращаем результат согласно флагу inplace
    return None if inplace else df