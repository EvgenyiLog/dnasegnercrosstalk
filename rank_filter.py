import pandas as pd
import numpy as np
from typing import Optional

def rank_filter(df: pd.DataFrame, rang: float, window_size: int) -> pd.DataFrame:
    if not (0 <= rang <= 1):
        raise ValueError("rang должен быть в диапазоне от 0 до 1.")
    
    rank_index = int(window_size * rang)
    if rank_index == 0:
        rank_index = 1

    def apply_rank_filter(series):
        return series.rolling(window=window_size, min_periods=1).apply(
            lambda x: np.partition(x, rank_index)[rank_index] if len(x) > rank_index else np.nan,
            raw=False
        )

    return df.apply(apply_rank_filter, axis=0)
