from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train and benchmark regression models.")
    p.add_argument(
        "--config",
        required=True,
        help="Path to YAML config, e.g. configs/regression_tender.yaml",
    )
    return p.parse_args()


def _load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError("Config must be a YAML mapping")
    return raw


def _build_preprocessor(df: pd.DataFrame) -> Tuple[ColumnTransformer, list, list]:
    cat_cols = [c for c in df.columns if df[c].dtype == "object"]
    num_cols = [c for c in df.columns if c not in cat_cols]

    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    pre = ColumnTransformer(
        transformers=[
            ("num", numeric, num_cols),
            ("cat", categorical, cat_cols),
        ],
        remainder="drop",
    )
    return pre, num_cols, cat_cols


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": _rmse(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _make_models(cfg: Dict[str, Any]) -> Dict[str, Any]:
    mflags = cfg.get("models", {}) or {}
    tcfg = cfg.get("training", {}) or {}

    models: Dict[str, Any] = {}
    if mflags.get("linear", True):
        models["linear_regression"] = LinearRegression()
    if mflags.get("ridge", True):
        models["ridge"] = Ridge(alpha=float(tcfg.get("ridge_alpha", 1.0)), random_state=int(cfg["data"]["random_state"]))
    if mflags.get("lasso", True):
        models["lasso"] = Lasso(alpha=float(tcfg.get("lasso_alpha", 0.001)), random_state=int(cfg["data"]["random_state"]), max_iter=20000)
    if mflags.get("svr", True):
        models["svr_rbf"] = SVR(C=float(tcfg.get("svr_c", 10.0)), epsilon=float(tcfg.get("svr_epsilon", 0.1)))
    if mflags.get("decision_tree", True):
        models["decision_tree"] = DecisionTreeRegressor(
            random_state=int(cfg["data"]["random_state"]),
            max_depth=int(tcfg.get("decision_tree_max_depth", 12)),
        )
    if mflags.get("random_forest", True):
        models["random_forest"] = RandomForestRegressor(
            random_state=int(cfg["data"]["random_state"]),
            n_estimators=int(tcfg.get("random_forest_n_estimators", 400)),
            max_depth=int(tcfg.get("random_forest_max_depth", 18)),
            n_jobs=-1,
        )
    if mflags.get("xgboost", True):
        try:
            from xgboost import XGBRegressor  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "xgboost is enabled but could not be imported. "
                "On macOS this is commonly fixed by installing OpenMP runtime: `brew install libomp`. "
                f"Original error: {e}"
            )
        models["xgboost"] = XGBRegressor(
            random_state=int(cfg["data"]["random_state"]),
            n_estimators=int(tcfg.get("xgboost_n_estimators", 600)),
            max_depth=int(tcfg.get("xgboost_max_depth", 8)),
            learning_rate=float(tcfg.get("xgboost_learning_rate", 0.05)),
            subsample=float(tcfg.get("xgboost_subsample", 0.9)),
            colsample_bytree=float(tcfg.get("xgboost_colsample_bytree", 0.9)),
            reg_lambda=1.0,
            objective="reg:squarederror",
            n_jobs=-1,
        )
    return models


def main() -> None:
    args = _parse_args()
    cfg = _load_config(args.config)

    data_cfg = cfg.get("data", {}) or {}
    csv_path = str(data_cfg.get("csv_path"))
    target = str(data_cfg.get("target"))
    drop_columns = list(data_cfg.get("drop_columns", []) or [])
    test_size = float(data_cfg.get("test_size", 0.2))
    random_state = int(data_cfg.get("random_state", 42))

    out_cfg = cfg.get("output", {}) or {}
    artifacts_dir = Path(str(out_cfg.get("artifacts_dir", "artifacts/regression")))
    reports_dir = Path(str(out_cfg.get("reports_dir", "reports")))

    df = pd.read_csv(csv_path)
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not in CSV columns")

    y = df[target].astype(float).to_numpy()
    X = df.drop(columns=[target] + [c for c in drop_columns if c in df.columns])

    pre, num_cols, cat_cols = _build_preprocessor(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    models = _make_models(cfg)

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "metrics").mkdir(parents=True, exist_ok=True)

    results: Dict[str, Any] = {
        "dataset": {"csv_path": csv_path, "rows": int(len(df)), "target": target},
        "split": {"test_size": test_size, "random_state": random_state},
        "features": {"numeric": num_cols, "categorical": cat_cols, "n_features_raw": int(X.shape[1])},
        "models": {},
    }

    for name, model in models.items():
        pipe = Pipeline(steps=[("preprocess", pre), ("model", model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        metrics = _evaluate(y_test, preds)
        results["models"][name] = {
            "metrics": metrics,
            "artifact_path": str(artifacts_dir / f"{name}.joblib"),
        }

        joblib.dump(pipe, artifacts_dir / f"{name}.joblib")

    out_metrics = reports_dir / "metrics" / "metrics.json"
    with open(out_metrics, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)

    # Simple console summary
    ranked = sorted(
        ((k, v["metrics"]["rmse"]) for k, v in results["models"].items()),
        key=lambda x: x[1],
    )
    print(f"Wrote metrics: {out_metrics}")
    print("RMSE ranking (lower is better):")
    for k, rmse in ranked:
        print(f"- {k}: {rmse:.4f}")


if __name__ == "__main__":
    main()

