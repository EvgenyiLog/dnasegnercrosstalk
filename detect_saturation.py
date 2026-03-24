import pandas as pd
from typing import Optional, Dict, Any, List, Tuple

def detect_saturation(raw_original:pd.Dataframe, dye_order:List[str], bit_depth:int=16, threshold_frac:float=0.95):
    """Определение насыщенных сканов в сырых данных."""
    max_val = (2**bit_depth - 1) if bit_depth == 16 else 32767  # signed 16-bit
    threshold = max_val * threshold_frac
    
    mask = raw_original >= threshold
    n_saturated = mask.any(axis=1).sum()
    
    print(f"Порог насыщения: {threshold:.0f}")
    print(f"Насыщенных сканов: {n_saturated} / {len(raw_original)} "
          f"({n_saturated/len(raw_original):.2%})")
    
    for ch in range(4):
        n = mask[:, ch].sum()
        if n > 0:
            print(f"  Канал {dye_order[ch]}: {n} насыщенных сканов")
    
    return mask