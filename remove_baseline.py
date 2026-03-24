import pandas as pd
from pybaselines import Baseline

def remove_baseline(df: pd.DataFrame, methods: list, **kwargs):
    """
    Убирает базовую линию из каждой колонки DataFrame всеми выбранными методами.

    Параметры
    ---------
    df : pd.DataFrame
        Входные данные (каждая колонка — сигнал).
    methods : list[str]
        Список методов pybaselines (например: ["asls", "airpls", "imodpoly"]).
    **kwargs : dict
        Параметры, передаваемые в соответствующую функцию pybaselines.

    Возвращает
    ----------
    corrected_df : pd.DataFrame
        Сигналы после коррекции (те же имена колонок, что у исходного df).
    baselines_df : pd.DataFrame
        Оцененные базовые линии (имена колонок = "<col>_<method>").
    """
    corrected_data = {}
    baseline_data = {}

    for col in df.columns:
        y = df[col].values
        # Для каждой колонки храним исправленный сигнал
        corrected_col = y.copy().astype(float)  # float, чтобы поддерживать NaN

        for method in methods:
            # Создаём объект Baseline
            baseline_fitter = Baseline(x_data=None)  # x_data=None если равномерно
            method_func = getattr(baseline_fitter, method)

            # Получаем параметры для метода
            method_kwargs = kwargs.get(method, {})

            # Выполняем подгонку базовой линии
            baseline = method_func(y, **method_kwargs)[0]  # [0] — это baseline

            # Сохраняем базовую линию
            baseline_data[f"{col}_{method}"] = baseline

            # Вычитаем из сигнала (накапливаем коррекцию)
            corrected_col -= baseline

        # Сохраняем окончательный исправленный сигнал
        corrected_data[col] = corrected_col

    corrected_df = pd.DataFrame(corrected_data, index=df.index)
    baselines_df = pd.DataFrame(baseline_data, index=df.index)

    return corrected_df, baselines_df