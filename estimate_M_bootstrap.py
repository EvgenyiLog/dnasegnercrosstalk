
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.optimize import nnls
from sklearn.linear_model import LinearRegression

CHANNELS = ["A", "C", "G", "T"]

# ----------------------------
# 1) Поиск чистых пиков
# ----------------------------
def find_clean_peaks(df, channels=CHANNELS, purity_thr=0.9, min_height=None, min_distance=10):
    """
    df: DataFrame с 4 каналами (A/C/G/T)
    Возвращает DataFrame с колонками:
      - idx: индекс пика
      - label: класс пика (0..3)
      - purity: чистота пика
      - x0..x3: исходный 4-канальный вектор пика
    """
    X = df[channels].to_numpy(dtype=float)
    envelope = X.max(axis=1)

    peaks, props = find_peaks(envelope, height=min_height, distance=min_distance)
    rows = []
    for i in peaks:
        x = X[i]
        s = x.sum()
        if s <= 0:
            continue
        label = int(np.argmax(x))
        purity = float(x[label] / s)
        if purity >= purity_thr:
            rows.append({
                "idx": int(i),
                "label": label,
                "purity": purity,
                "x0": x[0], "x1": x[1], "x2": x[2], "x3": x[3],
            })

    return pd.DataFrame(rows)


# ----------------------------
# 2) Простая оценка M по средним чистых пиков
# ----------------------------
def estimate_M_mean(clean_peaks_df, channels=CHANNELS):
    """
    Столбец j = среднее нормированных чистых пиков класса j.
    M[:, j] соответствует красителю/каналу j.
    """
    M = np.zeros((4, 4), dtype=float)
    Xcols = ["x0", "x1", "x2", "x3"]

    for j in range(4):
        block = clean_peaks_df[clean_peaks_df["label"] == j][Xcols].to_numpy(dtype=float)
        if len(block) == 0:
            M[:, j] = np.eye(4)[:, j]
            continue
        block = block / block.sum(axis=1, keepdims=True)
        m = block.mean(axis=0)
        m = np.clip(m, 0, None)
        M[:, j] = m / m.sum()

    return M


# ----------------------------
# 3) sklearn-вариант (intercept-only regression)
# ----------------------------
def estimate_M_sklearn(clean_peaks_df):
    """
    То же самое, но через sklearn LinearRegression.
    На практике для чистых пиков это эквивалентно среднему.
    """
    M = np.zeros((4, 4), dtype=float)
    Xcols = ["x0", "x1", "x2", "x3"]

    for j in range(4):
        block = clean_peaks_df[clean_peaks_df["label"] == j][Xcols].to_numpy(dtype=float)
        if len(block) == 0:
            M[:, j] = np.eye(4)[:, j]
            continue
        block = block / block.sum(axis=1, keepdims=True)

        Xreg = np.ones((len(block), 1), dtype=float)
        reg = LinearRegression(fit_intercept=False)
        reg.fit(Xreg, block)
        m = reg.coef_.ravel()   # shape (4,)
        m = np.clip(m, 0, None)
        M[:, j] = m / m.sum()

    return M


# ----------------------------
# 4) Итеративная Li & Speed-подобная оценка M
# ----------------------------
def _normalize_columns(M, eps=1e-12):
    M = np.asarray(M, dtype=float)
    M = np.clip(M, 0, None)
    for j in range(M.shape[1]):
        s = M[:, j].sum()
        if s > eps:
            M[:, j] /= s
        else:
            M[:, j] = np.eye(M.shape[0])[:, j]
    return M


def estimate_M_li_speed(data, n_iter=30, min_height=200,
                        min_distance=10, min_purity=0.5,
                        init_M=None, verbose=True,
                        ridge=1e-8):
    """
    data: array shape (N_scans, 4), baseline-corrected
    Идея: найти пики -> на E-шаге назначить пик классу -> на M-шаге обновить столбцы.
    Это ближе к итеративной калибровке, чем к строгому EM.
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 2 or data.shape[1] != 4:
        raise ValueError("data must have shape (N_scans, 4)")

    M = np.eye(4, dtype=float) if init_M is None else _normalize_columns(init_M)

    envelope = data.max(axis=1)
    peak_pos, _ = find_peaks(envelope, height=min_height, distance=min_distance)
    if len(peak_pos) == 0:
        raise ValueError("No peaks found")

    peak_I = np.clip(data[peak_pos, :], 0, None)
    norms = peak_I.sum(axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    peak_normalized = peak_I / norms

    if verbose:
        print(f"Найдено пиков: {len(peak_pos)}")

    for iteration in range(n_iter):
        M_reg = M + ridge * np.eye(4)
        try:
            M_inv = np.linalg.inv(M_reg)
        except np.linalg.LinAlgError:
            M_inv = np.linalg.pinv(M_reg)

        concentrations = (M_inv @ peak_I.T).T
        concentrations = np.clip(concentrations, 0, None)

        assignments = np.argmax(concentrations, axis=1)
        conc_sums = concentrations.sum(axis=1)
        conc_sums[conc_sums == 0] = 1.0
        purities = concentrations.max(axis=1) / conc_sums

        M_new = np.zeros((4, 4), dtype=float)
        for j in range(4):
            mask = (assignments == j) & (purities >= min_purity)
            if mask.sum() < 3:
                mask = (assignments == j)
            if mask.sum() < 1:
                M_new[:, j] = M[:, j]
                continue
            M_new[:, j] = peak_normalized[mask].mean(axis=0)

        M_new = _normalize_columns(M_new)
        change = np.max(np.abs(M_new - M))
        M = M_new

        if verbose and (iteration < 3 or iteration % 5 == 0):
            print(f"Итерация {iteration + 1}: max Δ = {change:.6e}")
        if change < 1e-6:
            if verbose:
                print(f"Сходимость на итерации {iteration + 1}")
            break

    return M


# ----------------------------
# 5) NNLS-деконволюция сигналов уже по известной M
# ----------------------------
def deconvolve_nnls(data, M):
    """
    data: (N_scans, 4)
    Возвращает оценки концентраций (N_scans, 4) при фиксированной M.
    """
    data = np.asarray(data, dtype=float)
    M = np.asarray(M, dtype=float)
    out = np.zeros_like(data, dtype=float)
    for i in range(data.shape[0]):
        out[i], _ = nnls(M, np.clip(data[i], 0, None))
    return out


# ----------------------------
# 6) Bootstrap для стабильности M
# ----------------------------
def bootstrap_M_from_clean_peaks(clean_peaks_df, B=500, seed=0):
    rng = np.random.default_rng(seed)
    Xcols = ["x0", "x1", "x2", "x3"]
    M_boot = np.zeros((B, 4, 4), dtype=float)

    for b in range(B):
        rows = []
        for j in range(4):
            sub = clean_peaks_df[clean_peaks_df["label"] == j]
            if len(sub) == 0:
                continue
            idx = rng.integers(0, len(sub), size=len(sub))
            rows.append(sub.iloc[idx])
        sample = pd.concat(rows, ignore_index=True) if rows else clean_peaks_df.iloc[:0].copy()
        M_boot[b] = estimate_M_mean(sample)

    M_hat = estimate_M_mean(clean_peaks_df)
    se = M_boot.std(axis=0, ddof=1)
    lo = np.percentile(M_boot, 2.5, axis=0)
    hi = np.percentile(M_boot, 97.5, axis=0)
    return M_hat, se, lo, hi, M_boot
