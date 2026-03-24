from typing import Tuple
import numpy as np
from numpy.fft import rfft, irfft, fft, ifft
import pandas as pd


def tikhonov_filter_1d(
    s: np.ndarray,
    lmbda: float,
    npd: int = 16
) -> Tuple[np.ndarray, np.ndarray]:
    """Lowpass filter based on Tikhonov regularization for 1D signals.

    Applies a global smoothing filter by solving a Tikhonov-regularized optimization problem
    that penalizes the L2 norm of the discrete gradient (first differences). The input signal
    is decomposed into low-frequency (smoothed) and high-frequency (residual) components.

    The lowpass component is the solution to:
        argmin_x  (1/2)||x - s||² + (λ/2)||Gx||²
    where G is the first-order finite difference operator (discrete gradient).

    After filtering, the highpass component is computed as: shp = s - slp.

    Args:
        s (np.ndarray): 1D input signal (shape: (N,)).
        lmbda (float): Regularization parameter. Larger values produce smoother outputs.
            Typical range: 0.1 (weak) to 50 (strong smoothing).
        npd (int, optional): Number of samples to pad at each boundary to reduce edge effects.
            Must be non-negative. Default is 16.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - slp (np.ndarray): Lowpass-filtered signal (same shape as input).
            - shp (np.ndarray): Highpass component (details + noise), computed as s - slp.

    Raises:
        ValueError: If input `s` is not a 1D array.

    Example:
        >>> import numpy as np
        >>> t = np.linspace(0, 4*np.pi, 100)
        >>> s = np.sin(t) + 0.3 * np.random.randn(100)
        >>> slp, shp = tikhonov_filter_1d(s, lmbda=5.0)
        >>> print(slp.shape, shp.shape)
        (100,) (100,)
    """
    s = np.asarray(s)
    if s.ndim != 1:
        raise ValueError("Input must be a 1D array")

    # Choose FFT function based on data type
    if np.isrealobj(s):
        fft_func = rfft
        ifft_func = irfft
    else:
        fft_func = fft
        ifft_func = ifft

    N = len(s)
    n_padded = N + 2 * npd

    # Discrete gradient filter: [-1, 1]
    g = np.array([-1.0, 1.0])
    G = fft_func(g, n=n_padded)

    # Frequency-domain regularization term: |G(ω)|²
    G_sq = np.conj(G) * G  # Power spectrum of gradient kernel
    A = 1.0 + lmbda * G_sq  # Denominator in frequency domain

    # Symmetric padding to reduce boundary artifacts
    sp = np.pad(s, (npd, npd), mode='symmetric')

    # Forward FFT
    Sp = fft_func(sp)

    # Apply filter in frequency domain
    Sp_filtered = Sp / A

    # Inverse FFT
    sp_filtered = ifft_func(Sp_filtered, n=n_padded)

    # Remove padding
    slp = sp_filtered[npd : npd + N]

    # Highpass component
    shp = s - slp

    return slp.astype(s.dtype), shp.astype(s.dtype)  

from typing import Tuple
import numpy as np
import spkit as sp


def tikhonov_filter_1d_frft(
    s: np.ndarray,
    lmbda: float,
    alpha: float = 1.0,
    npd: int = 16
) -> Tuple[np.ndarray, np.ndarray]:
    """Tikhonov lowpass filter using Fractional Fourier Transform (spkit.frft).

    Args:
        s (np.ndarray): 1D input signal.
        lmbda (float): Regularization parameter.
        alpha (float): Fractional order of Fourier transform (α=1 is standard FFT).
        npd (int): Padding size for edge effect reduction.

    Returns:
        (slp, shp): Lowpass and highpass components.
    """
    s = np.asarray(s, dtype=float)
    if s.ndim != 1:
        raise ValueError("Input must be 1D")

    N = len(s)
    n_padded = N + 2 * npd

    # Симметричное дополнение
    spad = np.pad(s, (npd, npd), mode="symmetric")

    # Прямое дробное Фурье преобразование
    Sp = sp.frft(spad, alpha)

    # Дискретный градиент [-1, 1] → во фракционном Фурье
    g = np.zeros_like(spad)
    g[0], g[1] = -1.0, 1.0
    G = sp.frft(g, alpha)

    # |G|² и регуляризационный знаменатель
    G_sq = np.conj(G) * G
    A = 1.0 + lmbda * G_sq

    # Фильтрация в области дробного Фурье
    Sp_filtered = Sp / A

    # Обратное дробное Фурье
    sp_filtered = sp.ifrft(Sp_filtered, alpha)

    # Убираем паддинг
    slp = np.real(sp_filtered[npd : npd + N])

    # Высокочастотный остаток
    shp = s - slp

    return slp.astype(s.dtype), shp.astype(s.dtype)
  

import numpy as np
from scipy.interpolate import interp1d
from typing import Optional, Tuple

def select_lambda_gcv_1d(
    s: np.ndarray,
    lambda_min: float = 0.1,
    lambda_max: float = 100.0,
    num_lambdas: int = 50,
    npd: int = 16
) -> float:
    """
    Выбирает оптимальный параметр регуляризации λ по методу GCV (Generalized Cross-Validation).

    Подходит для длинных сигналов (N ~ 10_000). Использует FFT для ускорения.

    Args:
        s (np.ndarray): 1D input signal.
        lambda_min (float): Минимальное значение λ для поиска.
        lambda_max (float): Максимальное значение λ.
        num_lambdas (int): Количество значений λ в логарифмической сетке.
        npd (int): Число отсчётов для padding'а.

    Returns:
        float: Оптимальное значение λ, минимизирующее GCV-функцию.
    """
    s = np.asarray(s, dtype=np.float64)
    N = len(s)

    # Логарифмическая сетка по λ
    lambdas = np.logspace(np.log10(lambda_min), np.log10(lambda_max), num_lambdas)

    # Оператор градиента в частотной области
    g = np.array([-1.0, 1.0])
    n_padded = N + 2 * npd
    if np.isrealobj(s):
        from numpy.fft import rfft, irfft
        fft_func, ifft_func = rfft, irfft
    else:
        from numpy.fft import fft, ifft
        fft_func, ifft_func = fft, ifft

    G = fft_func(g, n=n_padded)
    G_sq = np.conj(G) * G  # |G(ω)|²

    # Подготовим padded-сигнал один раз (для padding)
    s_padded = np.pad(s, (npd, npd), mode='symmetric')
    S_padded = fft_func(s_padded)

    gcv_scores = []

    for lmbda in lambdas:
        # Знаменатель в частотной области
        A = 1.0 + lmbda * G_sq
        S_filtered = S_padded / A
        x_lmbda = ifft_func(S_filtered, n=n_padded)[npd:npd+N]

        # Остаток
        residual = s - x_lmbda
        res_norm_sq = np.linalg.norm(residual) ** 2

        # След матрицы фильтрации: tr(I - A⁻¹)
        # В частотной области: A(ω) = 1 + λ|G(ω)|² → A⁻¹(ω)
        # trace_term = sum(1 - 1/A(ω)) ≈ эффективное число параметров
        A_trunc = A[:len(residual)]  # урезаем до N (для rfft это N//2+1, но мы аппроксимируем)
        if len(A_trunc) != len(residual):
            # Для rfft: частот только N//2+1, но можно интерполировать
            freq_ratio = len(residual) / len(A_trunc)
            trace_approx = np.sum(1 - 1 / A_trunc) * freq_ratio
        else:
            trace_approx = np.sum(1 - 1 / A_trunc)

        if trace_approx <= 0 or trace_approx >= N:
            gcv = np.inf
        else:
            denominator = (N - trace_approx) ** 2
            gcv = (res_norm_sq / N) / (denominator / N**2)  # GCV formula
            gcv = gcv  # можно упростить

        gcv_scores.append(gcv)

    # Интерполяция для более точного минимума
    min_idx = np.argmin(gcv_scores)
    if 0 < min_idx < len(lambdas) - 1:
        # Квадратичная интерполяция вокруг минимума
        ln_lambdas = np.log(lambdas)
        f = interp1d(ln_lambdas, gcv_scores, kind='quadratic', fill_value='extrapolate')
        ln_dense = np.linspace(ln_lambdas[min_idx-1], ln_lambdas[min_idx+1], 100)
        scores_dense = f(ln_dense)
        best_ln = ln_dense[np.argmin(scores_dense)]
        opt_lambda = float(np.exp(best_ln))
    else:
        opt_lambda = float(lambdas[min_idx])

    return opt_lambda

def apply_tikhonov_filter_df(
    df: pd.DataFrame,
    lmbda: float | str = 'auto',
    method: str = 'gcv',
    npd: int = 16,
    lambda_min: float = 0.1,
    lambda_max: float = 100.0,
    num_lambdas: int = 50
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply Tikhonov filter to each column of a DataFrame with optional automatic lambda selection.

    Args:
        df (pd.DataFrame): Input DataFrame with numeric columns.
        lmbda (float or 'auto'): Regularization parameter. If 'auto', selects λ automatically.
        method (str): Method for auto-selection: 'gcv' (default) or 'l-curve' (not implemented here).
        npd (int): Padding size. Default 16.
        lambda_min (float): Min λ for search (if auto).
        lambda_max (float): Max λ for search (if auto).
        num_lambdas (int): Number of λ values to try.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (df_low, df_high)
    """
    if lmbda == 'auto':
        lambda_values = {}
        for col in df.columns:
            s = df[col].values
            try:
                λ = select_lambda_gcv_1d(
                    s, lambda_min=lambda_min, lambda_max=lambda_max,
                    num_lambdas=num_lambdas, npd=npd
                )
                lambda_values[col] = λ
            except Exception as e:
                print(f"Failed to auto-select λ for '{col}': {e}")
                lambda_values[col] = 10.0  # fallback
        print("Auto-selected λ per column:", lambda_values)
    else:
        lambda_values = {col: lmbda for col in df.columns}

    df_low = pd.DataFrame(index=df.index, columns=df.columns, dtype=np.float64)
    df_high = pd.DataFrame(index=df.index, columns=df.columns, dtype=np.float64)

    for col in df.columns:
        s = df[col].values
        slp, shp = tikhonov_filter_1d(s, lmbda=lambda_values[col], npd=npd)
        df_low[col] = slp
        df_high[col] = shp

    return df_low, df_high



def apply_tikhonov_filter_df_frft(
    df: pd.DataFrame,
    lmbda: float | str = 'auto',
    method: str = 'gcv',
    alpha:float=0.5,
    npd: int = 16,
    lambda_min: float = 0.1,
    lambda_max: float = 100.0,
    num_lambdas: int = 50
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply Tikhonov filter to each column of a DataFrame with optional automatic lambda selection.

    Args:
        df (pd.DataFrame): Input DataFrame with numeric columns.
        lmbda (float or 'auto'): Regularization parameter. If 'auto', selects λ automatically.
        method (str): Method for auto-selection: 'gcv' (default) or 'l-curve' (not implemented here).
        npd (int): Padding size. Default 16.
        lambda_min (float): Min λ for search (if auto).
        lambda_max (float): Max λ for search (if auto).
        num_lambdas (int): Number of λ values to try.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (df_low, df_high)
    """
    if lmbda == 'auto':
        lambda_values = {}
        for col in df.columns:
            s = df[col].values
            try:
                λ = select_lambda_gcv_1d(
                    s, lambda_min=lambda_min, lambda_max=lambda_max,
                    num_lambdas=num_lambdas, npd=npd
                )
                lambda_values[col] = λ
            except Exception as e:
                print(f"Failed to auto-select λ for '{col}': {e}")
                lambda_values[col] = 10.0  # fallback
        print("Auto-selected λ per column:", lambda_values)
    else:
        lambda_values = {col: lmbda for col in df.columns}

    df_low = pd.DataFrame(index=df.index, columns=df.columns, dtype=np.float64)
    df_high = pd.DataFrame(index=df.index, columns=df.columns, dtype=np.float64)

    for col in df.columns:
        s = df[col].values
        slp, shp = tikhonov_filter_1d_frft(s, lmbda=lambda_values[col],alpha=alpha, npd=npd)
        df_low[col] = slp
        df_high[col] = shp

    return df_low, df_high