import pingouin as pg
import dcor
import numpy as np
import pandas as pd
from typing import List,Union,Optional
def robust_corr_matrix(
    data: Union[pd.DataFrame,np.ndarray],
    methods: Union[str, List[str]] = 'all',
    annot: bool = True,
    pval_threshold: Optional[float] = 0.05,
    symmetric: bool = True,
    **kwargs
) -> Union[pd.DataFrame, dict]:
    """
    Расширенная матрица корреляций с поддержкой устойчивых (robust) и нелинейных методов.

    Исправлено: корректная обработка dcor (возвращает float, а не Series).
    """
    available_methods = {
        # Стандартные методы
        'pearson': lambda x, y: pg.corr(x, y, method='pearson', **kwargs),
        'spearman': lambda x, y: pg.corr(x, y, method='spearman', **kwargs),
        'kendall': lambda x, y: pg.corr(x, y, method='kendall', **kwargs),
        
        # Устойчивые (robust) методы
        'bicor': lambda x, y: pg.corr(x, y, method='bicor', **kwargs),
        'percbend': lambda x, y: pg.corr(x, y, method='percbend', **kwargs),
        'shepherd': lambda x, y: pg.corr(x, y, method='shepherd', **kwargs),
        'skipped': lambda x, y: pg.corr(x, y, method='skipped', **kwargs),
        
        # Нелинейная корреляция (dcor возвращает float, а не Series)
        'dcor': lambda x, y: {'r': dcor.distance_correlation(x, y), 'p-val': np.nan}
    }

    if methods == 'all':
        methods = ['pearson', 'spearman', 'kendall', 'bicor', 'percbend', 'shepherd', 'skipped']
    elif isinstance(methods, str):
        methods = [methods]

    results = {}
    for method in methods:
        if method not in available_methods:
            raise ValueError(f"Метод '{method}' не поддерживается. Доступные: {list(available_methods.keys())}")

        corr_func = available_methods[method]
        if isinstance(data, pd.DataFrame):
           cols = data.columns
           n = len(cols)
        else:
            # For numpy arrays (and similar)
            n = data.shape[1]
            cols = [f"V{i}" for i in range(n)]
        corr_matrix = np.zeros((n, n))
        pval_matrix = np.full((n, n), np.nan)

        for i in range(n):
            for j in range(n):
                if i == j:
                    corr_matrix[i, j] = 1.0
                    pval_matrix[i, j] = 0.0
                elif symmetric and i > j:
                    corr_matrix[i, j] = corr_matrix[j, i]
                    pval_matrix[i, j] = pval_matrix[j, i]
                else:
                    if isinstance(data, pd.DataFrame):
                        x = data[cols[i]].values
                        y = data[cols[j]].values
                    else:
                        x = data[:, i]
                        y = data[:, j]
                    res = corr_func(x, y)
                    # Исправление для dcor (res может быть dict или Series)
                    if isinstance(res, dict):
                        corr_matrix[i, j] = res['r']
                        pval_matrix[i, j] = res.get('p-val', np.nan)
                    else:  # Для pingouin.corr (возвращает DataFrame)
                        corr_matrix[i, j] = res['r'].iloc[0]
                        if 'p-val' in res:
                            pval_matrix[i, j] = res['p-val'].iloc[0]

        corr_df = pd.DataFrame(corr_matrix, columns=cols, index=cols)
        pval_df = pd.DataFrame(pval_matrix, columns=cols, index=cols)

        if annot:
            annot_df = corr_df.copy().astype(str)
            if not pval_df.isnull().all().all():
                for i in range(n):
                    for j in range(n):
                        if i != j and not np.isnan(pval_df.iloc[i, j]):
                            sig = '*' if pval_df.iloc[i, j] < pval_threshold else ''
                            annot_df.iloc[i, j] = f"{corr_df.iloc[i, j]:.2f}{sig}"
            corr_df = annot_df

        if symmetric:
            results[method] = corr_df
        else:
            results[method] = {'r': corr_df, 'pval': pval_df}

    return results if len(results) > 1 else results[methods[0]]