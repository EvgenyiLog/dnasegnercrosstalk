import numpy as np
import pandas as pd
from scipy.optimize import least_squares

def levenberg_orthogonalizer(df: pd.DataFrame, lam: float = 10**14.7) -> np.ndarray:
    """
    Строит матрицу преобразования M для ортогонализации признаков DataFrame 
    с фиксированной единичной диагональю, используя регуляризованный метод 
    Левенберга–Марквардта (least_squares из SciPy).

    Параметры
    ----------
    df : pd.DataFrame
        Таблица данных размером (n, d), где n — число наблюдений, d — число признаков.
    lam : float, optional (default=1e-2)
        Коэффициент регуляризации (левенберговский множитель).
        Чем больше lam, тем сильнее M притягивается к единичной матрице.

    Возвращает
    ----------
    M_opt : np.ndarray
        Матрица размера (d, d) с диагональю, равной 1.
        При умножении на X.T (где X=df.values) строки результата становятся 
        максимально близкими к ортогональным.
    """

    X = df.values  # n x d
    n, d = X.shape

    def residuals(m_flat, X, lam):
        """Вектор невязок: ортогональность + регуляризация"""
        M = m_flat.reshape(d, d)
        np.fill_diagonal(M, 1.0)  # фиксируем диагональ = 1

        Y = M @ X.T  # d x n
        G = Y @ Y.T  # матрица Грама

        # 1) отклонение от ортогональности
        res1 = (G - np.eye(d)).ravel()
        # 2) регуляризация (M ~ I)
        res2 = (M - np.eye(d)).ravel()

        return np.concatenate([res1, lam * res2])

    # начальное приближение — единичная матрица
    m0 = np.eye(d).ravel()

    # оптимизация
    res = least_squares(residuals, m0, args=(X, lam))
    M_opt = res.x.reshape(d, d)
    np.fill_diagonal(M_opt, 1.0)

    return M_opt