from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

ANOMALY_PREDICTIONS = Counter(
    "anomaly_predictions_total",
    "LR predictions",
    ["result"],
)


def metrics_endpoint() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
