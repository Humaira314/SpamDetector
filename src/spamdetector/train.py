from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from .pipeline import build_pipeline


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "label" not in df.columns or "text" not in df.columns:
        df = df.rename(columns={df.columns[0]: "label", df.columns[1]: "text"})
    df["label"] = df["label"].astype(str).str.lower().str.strip()
    df["text"] = df["text"].astype(str)
    return df[["label", "text"]]


def compute_metrics(y_true, y_pred, labels=("ham", "spam")) -> Dict[str, object]:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None
    )
    precision_spam, recall_spam, f1_spam, _ = precision_recall_fscore_support(
        y_true, y_pred, pos_label="spam", average="binary"
    )

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "spam": {
            "precision": precision_spam,
            "recall": recall_spam,
            "f1": f1_spam,
        },
        "per_label": {
            label: {
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
                "support": int(support[idx]),
            }
            for idx, label in enumerate(labels)
        },
        "confusion_matrix": cm.tolist(),
        "labels": list(labels),
    }


def train_model(
    data_path: str,
    model_type: str = "nb",
    test_size: float = 0.2,
    random_state: int = 42,
    use_stemming: bool = False,
    use_stopwords: bool = True,
    use_nltk_stopwords: bool = True,
    calibrate_svm: bool = True,
) -> Tuple[object, Dict[str, object]]:
    df = load_dataset(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"],
    )

    pipeline = build_pipeline(
        model=model_type,
        use_stemming=use_stemming,
        use_stopwords=use_stopwords,
        use_nltk_stopwords=use_nltk_stopwords,
        calibrate_svm=calibrate_svm,
    )
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)
    return pipeline, metrics


def train_and_select(
    data_path: str,
    test_size: float = 0.2,
    random_state: int = 42,
    use_stemming: bool = False,
    use_stopwords: bool = True,
    use_nltk_stopwords: bool = True,
    calibrate_svm: bool = True,
) -> Tuple[str, Dict[str, object], Dict[str, object]]:
    results = {}
    models = {}

    for model_type in ("nb", "svm"):
        model, metrics = train_model(
            data_path=data_path,
            model_type=model_type,
            test_size=test_size,
            random_state=random_state,
            use_stemming=use_stemming,
            use_stopwords=use_stopwords,
            use_nltk_stopwords=use_nltk_stopwords,
            calibrate_svm=calibrate_svm,
        )
        results[model_type] = metrics
        models[model_type] = model

    best_model = max(results.items(), key=lambda item: item[1]["spam"]["f1"])[0]
    return best_model, results, models


def save_model(model: object, output_path: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output)


def save_metrics(metrics: Dict[str, object], output_path: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def load_model(model_path: str) -> object:
    return joblib.load(model_path)


def evaluate_model(model: object, data_path: str) -> Dict[str, object]:
    df = load_dataset(data_path)
    y_pred = model.predict(df["text"])
    return compute_metrics(df["label"], y_pred)
