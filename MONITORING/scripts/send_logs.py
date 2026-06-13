import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from itertools import cycle
from pathlib import Path
import random


def load_payload(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def send(url: str, payload: dict) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        raise urllib.error.URLError(f"HTTP {e.code}: {body}") from e


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--interval", type=float, default=2.0)
    p.add_argument("--rounds", type=int, default=0)
    p.add_argument("--anomaly", type=float, default=0.1)
    args = p.parse_args()

    ROOT = Path(__file__).resolve().parents[2]
    files = [
        ROOT / "API/test_logs/raw/normal_logs.json",
        ROOT / "API/test_logs/raw/anomaly_logs.json",
    ]

    payloads = [(f.name, load_payload(f)) for f in files]
    n = 0
    for name, payload in cycle(payloads):
        n += 1
        if args.rounds and n >= args.rounds:
            break

        chosen = random.choices(
            payloads, weights=[1 - args.anomaly, args.anomaly], k=1
        )[0]
        name, payload = chosen
        try:
            result = send("http://localhost:8080/forward", payload)
            label = "A" if result["is_anomaly"] else "N"
            print(f"[{n}] {name} → {label}  {result['score']:.3f}")
        except urllib.error.URLError as e:
            print(f"[ошибка: {e}", file=sys.stderr)
            sys.exit(1)

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
