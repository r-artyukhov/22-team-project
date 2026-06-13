"""HDFS JSON → поля для модели. Поддерживает LogHub-формат и короткий demo-формат."""


def parse_log(raw: dict) -> dict:
    return {
        "message": raw.get("Content") or raw.get("content") or raw.get("message") or "",
        "level": raw.get("Level") or raw.get("level") or "",
        "component": raw.get("Component") or raw.get("component") or "",
        "event_id": raw.get("EventId") or raw.get("event_id"),
    }


def parse_logs(raw_logs: list[dict]) -> list[dict]:
    return [parse_log(r) for r in raw_logs]
