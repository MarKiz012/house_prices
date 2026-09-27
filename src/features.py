"""Признаки: доменные для House Prices + обычный препроцессинг (пропуски, кодирование)."""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

# Оценки из описания данных (data_description.txt). NaN там почти везде значит
# «этого нет» (нет подвала, гаража, камина), поэтому он становится 0, а не пропуском.
QUALITY = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1}
ORDINAL_MAPS = {
    **{
        c: QUALITY
        for c in [
            "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC",
            "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC",
        ]
    },
    "BsmtExposure": {"Gd": 4, "Av": 3, "Mn": 2, "No": 1},
    "BsmtFinType1": {"GLQ": 6, "ALQ": 5, "BLQ": 4, "Rec": 3, "LwQ": 2, "Unf": 1},
    "BsmtFinType2": {"GLQ": 6, "ALQ": 5, "BLQ": 4, "Rec": 3, "LwQ": 2, "Unf": 1},
    "GarageFinish": {"Fin": 3, "RFn": 2, "Unf": 1},
    "Functional": {"Typ": 7, "Min1": 6, "Min2": 5, "Mod": 4, "Maj1": 3, "Maj2": 2, "Sev": 1, "Sal": 0},
}

# Площади, у которых NaN в тесте означает «нет подвала/гаража/облицовки», т.е. 0.
ZERO_IF_MISSING = [
    "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
    "BsmtFullBath", "BsmtHalfBath", "GarageCars", "GarageArea",
]

# Скошенные вправо площади: для линейных моделей берём логарифм.
# Деревьям монотонное преобразование безразлично, так что вреда нет.
LOG_COLUMNS = ["LotArea", "LotFrontage", "GrLivArea", "1stFlrSF", "TotalSF", "MasVnrArea", "TotalPorchSF"]


def add_features(df):
    """Доменные признаки House Prices.

    Функция не запоминает ничего про данные (никакой статистики по выборке),
    поэтому её безопасно применять к train и test сразу — утечки не будет.
    """
    df = df.copy()

    # Код типа дома записан числом, но это категория: 60 не «больше» 20.
    if "MSSubClass" in df:
        df["MSSubClass"] = "MS" + df["MSSubClass"].astype(str)

    for column, mapping in ORDINAL_MAPS.items():
        if column in df:
            df[column] = df[column].map(mapping).fillna(0)

    for column in ZERO_IF_MISSING:
        if column in df:
            df[column] = df[column].fillna(0)

    # Нет гаража -> года постройки гаража нет; разумнее взять год дома, чем медиану.
    if {"GarageYrBlt", "YearBuilt"} <= set(df):
        df["GarageYrBlt"] = df["GarageYrBlt"].fillna(df["YearBuilt"])

    if {"TotalBsmtSF", "1stFlrSF", "2ndFlrSF"} <= set(df):
        df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]

    if {"FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"} <= set(df):
        df["TotalBath"] = df["FullBath"] + df["BsmtFullBath"] + 0.5 * (df["HalfBath"] + df["BsmtHalfBath"])

    porches = ["OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "WoodDeckSF"]
    if set(porches) <= set(df):
        df["TotalPorchSF"] = df[porches].sum(axis=1)

    if {"YrSold", "YearBuilt", "YearRemodAdd"} <= set(df):
        df["HouseAge"] = (df["YrSold"] - df["YearBuilt"]).clip(lower=0)
        df["RemodAge"] = (df["YrSold"] - df["YearRemodAdd"]).clip(lower=0)
        df["IsRemodeled"] = (df["YearRemodAdd"] != df["YearBuilt"]).astype(int)

    # Качество x площадь — один из самых сильных признаков: хороший большой дом дорог вдвойне.
    if {"OverallQual", "GrLivArea"} <= set(df):
        df["QualLivArea"] = df["OverallQual"] * df["GrLivArea"]

    for new, source in [
        ("HasGarage", "GarageArea"), ("HasBsmt", "TotalBsmtSF"), ("Has2ndFlr", "2ndFlrSF"),
        ("HasFireplace", "Fireplaces"), ("HasPool", "PoolArea"),
    ]:
        if source in df:
            df[new] = (df[source].fillna(0) > 0).astype(int)

    for column in LOG_COLUMNS:
        if column in df:
            df[column] = np.log1p(df[column].clip(lower=0))

    # Utilities почти константа (в train одно отличное значение, в тесте ни одного).
    return df.drop(columns=[c for c in ("Utilities",) if c in df])


def make_preprocessor(X, cfg):
    """ColumnTransformer: заполнение пропусков + кодирование категорий.

    Колонки определяются по типу данных
    """
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if cfg["features"]["scale"]:
        numeric_steps.append(("scaler", StandardScaler()))

    if cfg["features"]["encoding"] == "onehot":
        encoder = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=5, sparse_output=False)
    else:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    categorical_steps = [
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoder", encoder),
    ]

    print(f"признаков: {len(numeric)} числовых, {len(categorical)} категориальных")
    return ColumnTransformer(
        [
            ("num", Pipeline(numeric_steps), numeric),
            ("cat", Pipeline(categorical_steps), categorical),
        ],
        remainder="drop",
    )


def prepare(X, X_test, cfg):
    """Применяет доменные признаки к train и test (если он есть)."""
    if not cfg["features"]["engineering"]:
        return X, X_test
    return add_features(X), (None if X_test is None else add_features(X_test))
