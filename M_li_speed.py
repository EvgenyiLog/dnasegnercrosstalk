import numpy as np
from scipy.signal import find_peaks


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
    Итеративная оценка матрицы M по 4 каналам.

    data: array-like, shape (N_scans, 4)
          baseline-corrected signal
    n_iter: максимум итераций
    min_height: порог для поиска пиков по envelope
    min_distance: минимальное расстояние между пиками
    min_purity: минимальная чистота пика для M-step
    init_M: начальная матрица 4x4; если None -> I
    ridge: малый стабилизатор для обратимости
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 2 or data.shape[1] != 4:
        raise ValueError("data must have shape (N_scans, 4)")

    # --- initialization ---
    if init_M is None:
        M = np.eye(4, dtype=float)
    else:
        M = _normalize_columns(init_M)

    # --- peak finding: one-time ---
    envelope = data.max(axis=1)
    peak_pos, _ = find_peaks(envelope, height=min_height, distance=min_distance)
    if len(peak_pos) == 0:
        raise ValueError("No peaks found. Lower min_height or min_distance.")

    peak_I = data[peak_pos, :]                  # (N_peaks, 4)
    peak_I_pos = np.clip(peak_I, 0, None)       # no negative intensities

    norms = peak_I_pos.sum(axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    peak_normalized = peak_I_pos / norms

    if verbose:
        print(f"Найдено пиков: {len(peak_pos)}")

    # --- iterations ---
    for iteration in range(n_iter):
        # E-step-like: estimate concentrations under current M
        M_reg = M + ridge * np.eye(4)
        try:
            M_inv = np.linalg.inv(M_reg)
        except np.linalg.LinAlgError:
            M_inv = np.linalg.pinv(M_reg)

        concentrations = (M_inv @ peak_I_pos.T).T
        concentrations = np.clip(concentrations, 0, None)

        assignments = np.argmax(concentrations, axis=1)
        conc_sums = concentrations.sum(axis=1)
        conc_sums[conc_sums == 0] = 1.0
        purities = concentrations.max(axis=1) / conc_sums

        # M-step: update each column from clean peaks assigned to that dye
        M_new = np.zeros((4, 4), dtype=float)

        for j in range(4):
            mask = (assignments == j) & (purities >= min_purity)

            # fallback if too few clean peaks
            if mask.sum() < 3:
                mask = (assignments == j)

            if mask.sum() < 1:
                M_new[:, j] = M[:, j]
                continue

            block = peak_normalized[mask]
            m_j = block.mean(axis=0)
            M_new[:, j] = m_j

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


def bootstrap_M_li_speed(data, B=200, seed=0, **kwargs):
    """
    Bootstrap для оценки устойчивости M.
    На каждой итерации заново оценивает M по ресэмплированным пикам.
    """
    rng = np.random.default_rng(seed)
    data = np.asarray(data, dtype=float)

    envelope = data.max(axis=1)
    peak_pos, _ = find_peaks(envelope, height=kwargs.get("min_height", 200),
                              distance=kwargs.get("min_distance", 10))
    peak_I = np.clip(data[peak_pos, :], 0, None)
    n = len(peak_I)

    if n == 0:
        raise ValueError("No peaks found for bootstrap.")

    M_boot = np.zeros((B, 4, 4), dtype=float)

    for b in range(B):
        idx = rng.integers(0, n, size=n)
        sample = np.zeros_like(data)
        sample[peak_pos] = peak_I[idx]
        M_boot[b] = estimate_M_li_speed(sample, verbose=False, **kwargs)

    M_hat = estimate_M_li_speed(data, verbose=False, **kwargs)
    se = M_boot.std(axis=0, ddof=1)
    lo = np.percentile(M_boot, 2.5, axis=0)
    hi = np.percentile(M_boot, 97.5, axis=0)
    return M_hat, se, lo, hi, M_boot