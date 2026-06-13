import re
from collections import Counter

import joblib
import pandas as pd

FEATURE_COLUMNS = [f"E{i}" for i in range(1, 30)]
_THRESHOLD = 0.5

_artifacts = joblib.load("models/logistic_regression.joblib")
_model = _artifacts["model"]
_scaler = _artifacts["scaler"]

_EVENT_RULES = [
    (r"Exception in receiveBlock|IOException", "E3"),
    (r"writeBlock.*[Ee]xception|SocketTimeoutException", "E20"),
    (r"Connection refused|Failed to transfer", "E11"),
    (r"PacketResponder", "E11"),
    (r"Receiving block", "E5"),
    (r"Received block", "E6"),
    (r"Served block", "E9"),
    (r"Verification succeeded", "E4"),
    (r"addStoredBlock|Redundant addStoredBlock", "E26"),
    (r"BLOCK\* NameSystem", "E22"),
    (r"ERROR", "E3"),
    (r"WARN", "E21"),
]


def _event_id(log: dict) -> str:
    if log.get("event_id"):
        return log["event_id"]
    text = f"{log.get('level', '')} {log.get('message', '')}"
    for pattern, eid in _EVENT_RULES:
        if re.search(pattern, text, re.IGNORECASE):
            return eid
    return "E5"


def predict_from_logs(logs: list[dict]) -> dict:
    counts = Counter(_event_id(l) for l in logs)
    X = pd.DataFrame([[counts.get(f"E{i}", 0) for i in range(1, 30)]], columns=FEATURE_COLUMNS)
    proba = float(_model.predict_proba(_scaler.transform(X))[0, 1])
    return {
        "score": proba,
        "is_anomaly": proba >= _THRESHOLD,
        "threshold": _THRESHOLD,
        "num_events": len(logs),
    }
