from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]

FEATURE_COLUMNS = [f"E{i}" for i in range(1, 30)]

MODEL_PATH = ROOT / "API" / "models" / "logistic_regression.joblib"
METRICS_PATH = ROOT / "API" / "models" / "lr_metrics.json"
MLFLOW_DB = ROOT / "MLflow" / "mlflow.db"
PLOTS_DIR = ROOT / "MLflow" / "artifacts" / "lr_plots"
DEFAULT_DATA = ROOT / "HDFS_v1" / "preprocessed"
DEFAULT_SAMPLE = 50_000


def load_dataset(data_dir: Path, sample: int) -> tuple[pd.DataFrame, pd.Series]:
    event_occ = pd.read_csv(data_dir / "Event_occurrence_matrix.csv")
    labels = pd.read_csv(data_dir / "anomaly_label.csv")

    if "Label" in event_occ.columns:
        event_occ = event_occ.drop(columns=["Label"])

    df = event_occ.merge(labels, on="BlockId", how="inner")
    y = (df["Label"] == "Anomaly").astype(int)

    drop = [c for c in ("BlockId", "Label", "Type") if c in df.columns]
    X = df.drop(columns=drop)

    if "E1" in X.columns:
        X["E1"] = pd.to_numeric(X["E1"], errors="coerce").fillna(0).astype(int)

    for col in FEATURE_COLUMNS:
        if col not in X.columns:
            X[col] = 0
    X = X[FEATURE_COLUMNS].fillna(0)

    if sample and sample < len(X):
        idx, _ = train_test_split(
            X.index, train_size=sample, stratify=y, random_state=42
        )
        X = X.loc[idx]
        y = y.loc[idx]
        print(f"Sample: {sample} blocks (из {len(df)})")
    else:
        print(f"dataset: {len(X)} blocks")

    return X, y


def train(data_dir: Path, sample: int, log_mlflow: bool) -> None:
    X, y = load_dataset(data_dir, sample)
    print(f"Anomalies: {y.sum()} / {len(y)} ({100 * y.mean():.2f}%)")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=5000, class_weight="balanced")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": "logistic_regression",
        "f1": float(f1_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "threshold": 0.5,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "classification_report": classification_report(
            y_test, y_pred, output_dict=True
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "feature_columns": FEATURE_COLUMNS,
            "threshold": 0.5,
        },
        MODEL_PATH,
    )
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved: {MODEL_PATH}")
    print(f"f1={metrics['f1']:.4f}  roc_auc={metrics['roc_auc']:.4f}")

    if log_mlflow:
        _log_mlflow(model, metrics, y_test, y_pred, y_proba, sample)


def _log_mlflow(model, metrics, y_test, y_pred, y_proba, sample) -> None:
    import mlflow
    import mlflow.sklearn

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    cm_path = PLOTS_DIR / "confusion_matrix.png"
    roc_path = PLOTS_DIR / "roc_curve.png"

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.tight_layout()
    plt.savefig(cm_path, dpi=120)
    plt.close()

    RocCurveDisplay.from_predictions(y_test, y_proba)
    plt.tight_layout()
    plt.savefig(roc_path, dpi=120)
    plt.close()

    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
    mlflow.set_experiment("LR_Experiment")

    with mlflow.start_run(run_name="lr_hdfs"):
        mlflow.log_params({
            "model": "LogisticRegression",
            "sample": sample or "full",
            "features": "E1-E29 (Event_occurrence_matrix)",
        })
        mlflow.log_metrics({
            k: metrics[k] for k in ("f1", "precision", "recall", "roc_auc")
        })
        mlflow.log_artifact(str(cm_path), "plots")
        mlflow.log_artifact(str(roc_path), "plots")
        mlflow.sklearn.log_model(model, "model")

    print(f"MLflow: {MLFLOW_DB}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    p.add_argument(
        "--sample",
        type=int,
        default=DEFAULT_SAMPLE,
        help=f"Сколько блоков взять (0 = все). По умолчанию {DEFAULT_SAMPLE}",
    )
    p.add_argument("--mlflow", action="store_true")
    args = p.parse_args()
    train(args.data_dir, args.sample, args.mlflow)


if __name__ == "__main__":
    main()
