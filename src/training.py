"""Кросс-валидация, метрики и сохранение результатов."""

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline

from src.data import load_data, make_folds
from src.features import add_features, make_preprocessor, prepare
from src.models import build_model


def compute_metrics(y_true, pred):
    """rmse_log — метрика Kaggle (RMSE между логарифмами цен), по ней и сравниваем."""
    pred = np.clip(pred, 1, None)  # на всякий случай: логарифм от отрицательной цены не взять
    return {
        "rmse_log": root_mean_squared_error(np.log(y_true), np.log(pred)),
        "rmse": root_mean_squared_error(y_true, pred),
        "mae": mean_absolute_error(y_true, pred),
        "r2": r2_score(y_true, pred),
    }


def build_pipeline(preprocessor, cfg):
    """Препроцессинг + модель. При target_log модель учится на log1p(цены),
    а predict сразу возвращает цену в долларах — снаружи про логарифм думать не нужно."""
    pipeline = Pipeline([("prep", clone(preprocessor)), ("model", build_model(cfg))])
    if cfg.get("target_log", True):
        pipeline = TransformedTargetRegressor(
            regressor=pipeline, func=np.log1p, inverse_func=np.expm1, check_inverse=False
        )
    return pipeline


def train(cfg):
    """Обучает модель по фолдам и сохраняет всё в outputs/<запуск>/."""
    X, y, X_test, test_ids = load_data(cfg)
    X, X_test = prepare(X, X_test, cfg)

    preprocessor = make_preprocessor(X, cfg)
    folds = make_folds(X, y, cfg)

    oof = np.zeros(len(X))
    test_log_pred = np.zeros(len(X_test)) if X_test is not None else None
    fold_metrics = []
    models = []

    for fold, (train_idx, valid_idx) in enumerate(folds, start=1):
        # build_pipeline клонирует препроцессор -> каждый фолд учит его заново, только на своих данных
        pipeline = build_pipeline(preprocessor, cfg)
        pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])

        oof[valid_idx] = pipeline.predict(X.iloc[valid_idx])
        if X_test is not None:
            # усредняем в логарифмах — так же, как считается метрика
            test_log_pred += np.log1p(np.clip(pipeline.predict(X_test), 0, None)) / len(folds)

        scores = compute_metrics(y.iloc[valid_idx], oof[valid_idx])
        fold_metrics.append(scores)
        models.append(pipeline)
        print(f"fold {fold}/{len(folds)}: " + ", ".join(f"{k}={v:.4f}" for k, v in scores.items()))

    oof_metrics = compute_metrics(y, oof)
    print("\nOOF: " + ", ".join(f"{k}={v:.4f}" for k, v in oof_metrics.items()))
    for name in oof_metrics:
        values = [m[name] for m in fold_metrics]
        print(f"  {name}: {np.mean(values):.4f} +- {np.std(values):.4f} по фолдам")

    test_pred = None if test_log_pred is None else np.expm1(test_log_pred)
    run_dir = save_run(cfg, models, oof, y, test_pred, test_ids, oof_metrics, fold_metrics)
    print(f"\nРезультаты: {run_dir}")
    return oof_metrics


def save_run(cfg, models, oof, y, test_pred, test_ids, oof_metrics, fold_metrics):
    run_name = f"{datetime.now():%Y-%m-%d_%H-%M-%S}_{cfg['model']['name']}"
    run_dir = Path(cfg["output_dir"]) / run_name
    (run_dir / "models").mkdir(parents=True, exist_ok=True)

    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {"config": cfg, "oof": oof_metrics, "folds": fold_metrics},
            f,
            indent=2,
            ensure_ascii=False,
        )

    pd.DataFrame({"y_true": y, "pred": oof}).to_csv(run_dir / "oof.csv", index=False)
    for i, model in enumerate(models):
        joblib.dump(model, run_dir / "models" / f"fold_{i}.joblib")

    if test_pred is not None:
        submission = pd.DataFrame({cfg["data"]["id_column"]: test_ids, cfg["data"]["target"]: test_pred})
        submission.to_csv(run_dir / "submission.csv", index=False)

    return run_dir


def predict(cfg, run_dir=None, input_path=None):
    """Предсказание сохранёнными моделями (по умолчанию — последний запуск)."""
    output_dir = Path(cfg["output_dir"])
    if run_dir is None:
        runs = sorted(p for p in output_dir.glob("*") if (p / "models").is_dir())
        if not runs:
            raise FileNotFoundError(f"В {output_dir} нет ни одного обученного запуска")
        run_dir = runs[-1]
    run_dir = Path(run_dir)
    print(f"Использую запуск: {run_dir}")

    # Данные готовим так же, как при обучении этого запуска, — берём его конфиг.
    with open(run_dir / "metrics.json", encoding="utf-8") as f:
        run_cfg = json.load(f).get("config", cfg)

    models = [joblib.load(p) for p in sorted((run_dir / "models").glob("fold_*.joblib"))]

    frame = pd.read_csv(input_path or run_cfg["data"]["test_path"])
    id_column, target = run_cfg["data"]["id_column"], run_cfg["data"]["target"]
    ids = frame[id_column] if id_column in frame.columns else pd.Series(range(len(frame)))
    X = frame.drop(columns=[c for c in (id_column, target) if c in frame.columns])
    if run_cfg["features"]["engineering"]:
        X = add_features(X)

    pred = np.expm1(np.mean([np.log1p(np.clip(m.predict(X), 0, None)) for m in models], axis=0))
    submission = pd.DataFrame({id_column: ids, target: pred})
    path = run_dir / "submission.csv"
    submission.to_csv(path, index=False)
    print(f"Сабмишн сохранён: {path}")
    return submission
