"""Создание модели по имени из конфига.
"""


def build_model(cfg):
    name = cfg["model"]["name"]
    # Параметры лежат по имени модели: model.params.<name>
    params = dict((cfg["model"].get("params") or {}).get(name) or {})
    seed = cfg["seed"]

    if name == "ridge":
        from sklearn.linear_model import Ridge

        return Ridge(**params)

    if name == "lasso":
        from sklearn.linear_model import Lasso

        return Lasso(max_iter=50_000, random_state=seed, **params)

    if name == "elasticnet":
        from sklearn.linear_model import ElasticNet

        return ElasticNet(max_iter=50_000, random_state=seed, **params)

    if name == "rf":
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(n_jobs=-1, random_state=seed, **params)

    if name == "histgb":
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(random_state=seed, **params)

    if name == "lgbm":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(n_jobs=-1, verbose=-1, random_state=seed, **params)

    if name == "xgb":
        from xgboost import XGBRegressor

        return XGBRegressor(n_jobs=-1, tree_method="hist", random_state=seed, **params)

    if name == "catboost":
        from catboost import CatBoostRegressor

        return CatBoostRegressor(verbose=0, allow_writing_files=False, random_seed=seed, **params)

    raise ValueError(
        f"Неизвестная модель '{name}'. Доступные: ridge, lasso, elasticnet, rf, histgb, lgbm, xgb, catboost"
    )
