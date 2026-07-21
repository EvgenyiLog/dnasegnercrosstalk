import numpy as np
import pandas as pd
from scipy import signal
from scipy.interpolate import interp1d

def time_warp_inverse_width(
    df: pd.DataFrame,
    min_height: int = 200,
    min_distance: int = 10
) -> pd.DataFrame:
    """
    Нормализация времени через обратную ширину пиков.

    Алгоритм:
    1. Находим пики по envelope
    2. Оцениваем ширины пиков
    3. Интерполируем ширину на всю ось времени
    4. Строим τ(t) = ∫ 1/w(t) dt
    5. Пересэмплируем сигнал

    Parameters
    ----------
    df : pd.DataFrame
        Индекс = время, значения = сигнал (1 канал или несколько)

    Returns
    -------
    pd.DataFrame
        Сигнал в нормированном времени τ ∈ [0,1]
    """

    t = df.index.values
    data = df.values

    # --- envelope (для мультиканала) ---
    if data.ndim == 2:
        envelope = data.max(axis=1)
    else:
        envelope = data

    # --- пики ---
    peak_pos, _ = signal.find_peaks(
        envelope,
        height=min_height,
        distance=min_distance
    )

    if len(peak_pos) < 2:
        raise ValueError("Недостаточно пиков для оценки ширины")

    # --- ширины ---
    widths, _, left_ips, right_ips = signal.peak_widths(
        envelope, peak_pos
    )

    # --- время пиков ---
    t_peaks = t[peak_pos]

    # --- интерполяция ширины на всё время ---
    f_width = interp1d(
        t_peaks,
        widths,
        kind='linear',
        fill_value="extrapolate"
    )

    width_t = f_width(t)

    # защита от нулей
    width_t = np.clip(width_t, 1e-6, None)

    # --- скорость времени ---
    speed = 1.0 / width_t

    # --- новое время ---
    tau = np.cumsum(speed)
    tau = tau - tau[0]
    tau = tau / tau[-1]

    # --- интерполяция сигнала ---
    tau_uniform = np.linspace(0, 1, len(t))

    result = {}

    for i, col in enumerate(df.columns):
        x = data[:, i] if data.ndim == 2 else data

        f = interp1d(
            tau,
            x,
            kind='linear',
            fill_value="extrapolate"
        )

        result[col] = f(tau_uniform)

    return pd.DataFrame(result, index=tau_uniform)