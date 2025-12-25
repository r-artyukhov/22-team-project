import re
from pathlib import Path
from typing import Optional

import joblib


class LogAnomalyDetector:
    """Isolation Forest модель для детекции аномалий в HDFS логах."""

    # Регулярные выражения для нормализации
    IP_RE = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
    HEX_RE = re.compile(r"\b[0-9a-fA-F]{6,}\b")
    PATH_RE = re.compile(r"(?:/[^ \t\n\r\f\v]+)+")
    NUM_RE = re.compile(r"\b\d+\b")
    BLK_RE = re.compile(r"\bblk_[\-\d]+\b")

    def __init__(self, model_path: str = "models/isolation_forest.joblib"):
        self.model_path = Path(model_path)
        self._load_model()

    def _load_model(self):
        """Загрузка модели из joblib файла."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        artifacts = joblib.load(self.model_path)
        self.model = artifacts["model"]
        self.vectorizer = artifacts["vectorizer"]
        self.threshold = artifacts["threshold"]

    def normalize_message(self, s: str) -> str:
        """Нормализация лог-сообщения."""
        s = s.lower()
        s = self.BLK_RE.sub(" <blk> ", s)
        s = self.IP_RE.sub(" <ip> ", s)
        s = self.PATH_RE.sub(" <path> ", s)
        s = self.HEX_RE.sub(" <hex> ", s)
        s = self.NUM_RE.sub(" <num> ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def tokenize_log_entry(self, message: str, component: str = "", level: str = "") -> str:
        """Преобразование лог-записи в токен."""
        msg = self.normalize_message(message)
        comp = component.split("$")[0].lower() if component else ""
        lvl = level.lower() if level else ""
        return f"{comp}_{lvl}__ {msg}"

    def predict(self, tokenized_block: str) -> dict:
        """
        Предсказание для токенизированной последовательности логов.

        Args:
            tokenized_block: Строка с токенами событий, разделёнными " . "

        Returns:
            dict с полями: score, is_anomaly, threshold
        """
        X = self.vectorizer.transform([tokenized_block])
        score = float(self.model.score_samples(X)[0])
        is_anomaly = score <= self.threshold

        return {
            "score": score,
            "is_anomaly": bool(is_anomaly),
            "threshold": self.threshold
        }

    def predict_from_logs(self, logs: list[dict]) -> dict:
        """
        Предсказание для списка лог-записей.

        Args:
            logs: Список словарей с ключами: message, component (опц.), level (опц.)

        Returns:
            dict с полями: score, is_anomaly, threshold, num_events
        """
        if not logs:
            return {
                "score": None,
                "is_anomaly": None,
                "threshold": self.threshold,
                "num_events": 0,
                "error": "Empty log sequence"
            }

        tokens = []
        for log in logs:
            message = log.get("message", "")
            component = log.get("component", "")
            level = log.get("level", "")
            token = self.tokenize_log_entry(message, component, level)
            tokens.append(token)

        tokenized_block = " . ".join(tokens)
        result = self.predict(tokenized_block)
        result["num_events"] = len(logs)

        return result


# Глобальный экземпляр модели
ml_model: Optional[LogAnomalyDetector] = None


def get_ml_model() -> LogAnomalyDetector:
    """Получение экземпляра модели (ленивая загрузка)."""
    global ml_model
    if ml_model is None:
        ml_model = LogAnomalyDetector()
    return ml_model
