from typing import Optional, Union
import pandas as pd
import numpy as np
from scipy.signal import wiener

def wiener_df(
    df: pd.DataFrame,
    mysize: Optional[Union[int, tuple[int, ...]]] = None,
    noise: Optional[float] = None
) -> pd.DataFrame:
    """
    Применяет фильтр Винера (Wiener) ко всем колонкам DataFrame.

    Каждая колонка рассматривается как отдельный сигнал.

    Parameters
    ----------
    df : pd.DataFrame
        Входной DataFrame с числовыми колонками.
    mysize : int или tuple of ints, optional
        Размер окна фильтра Винера. Если None, используется значение по умолчанию из scipy.signal.wiener.
    noise : float, optional
        Уровень шума. Если None, оценивается автоматически.

    Returns
    -------
    pd.DataFrame
        Новый DataFrame с отфильтрованными колонками.
    """
    df_filtered = pd.DataFrame(index=df.index)
    for col in df.columns:
        # Преобразуем колонку в numpy array
        signal = df[col].to_numpy(dtype=float)
        # Применяем фильтр Винера
        filtered = wiener(signal, mysize=mysize, noise=noise)
        # Добавляем в результат
        df_filtered[col] = filtered
    return df_filtered
