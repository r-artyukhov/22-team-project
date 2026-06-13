import argparse
import json
import re
from pathlib import Path

BLK = re.compile(r"blk_[0-9-]+")


def parse_hdfs_line(line: str) -> dict | None:
    parts = line.strip().split(" ", 4)
    if len(parts) < 5:
        return None
    date, time, pid, level, rest = parts
    if ":" not in rest:
        return None
    component, content = rest.split(":", 1)
    content = content.strip()
    if not BLK.search(content):
        return None
    return {
        "Date": date,
        "Time": time,
        "Pid": pid,
        "Level": level,
        "Component": component.strip(),
        "Content": content,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("log_file", type=Path)
    p.add_argument("-o", "--output", type=Path, required=True)
    p.add_argument("-n", "--max-lines", type=int, default=100)
    args = p.parse_args()

    logs = []
    with args.log_file.open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            row = parse_hdfs_line(line)
            if row:
                logs.append(row)
            if len(logs) >= args.max_lines:
                break

    block_id = BLK.search(logs[0]["Content"]).group(0) if logs else None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {"block_id": block_id, "logs": logs}, indent=2, ensure_ascii=False
        ),
        encoding="utf-8",
    )
    print(f"Saved {len(logs)} lines → {args.output}")


if __name__ == "__main__":
    main()
