import ast
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset


# Configuration
@dataclass
class Config:
    csv_path: str = "hdfs_sessions.csv"
    output_dir: str = "deeplog_output"
    seed: int = 42

    # path model
    history_size: int = 20
    top_g: int = 5
    top_p: float = 0.997
    event_embed_dim: int = 16
    path_hidden_dim: int = 64
    path_num_layers: int = 2
    path_dropout: float = 0.1
    path_lr: float = 1e-3
    path_batch_size: int = 1024
    path_epochs: int = 1

    # timing model
    time_hidden_dim: int = 64
    time_num_layers: int = 1
    time_dropout: float = 0.1
    time_lr: float = 5e-4
    time_batch_size: int = 1024
    time_epochs: int = 3
    timing_threshold_quantile: float = 0.992

    # data split among normal sessions only
    train_ratio: float = 0.8
    valid_ratio: float = 0.1

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# Utility functions
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_list_cell(x: str):
    """
    Robust parsing for cells like:
      "[E5,E22,E5]"
      "[0.0, 1.0, 0.0]"
    """
    if isinstance(x, list):
        return x
    s = str(x).strip()
    if not s:
        return []

    # Try direct literal_eval first.
    try:
        return ast.literal_eval(s)
    except Exception:
        pass

    # Fallback for tokens without quotes, e.g. [E5,E22,E9]
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].strip()
        if not inner:
            return []
        parts = [p.strip() for p in inner.split(",")]
        parsed = []
        for p in parts:
            if not p:
                continue
            try:
                parsed.append(float(p))
            except ValueError:
                parsed.append(p)
        return parsed

    raise ValueError(f"Cannot parse list cell: {x}")


def safe_label_to_binary(label: str) -> int:
    return 0 if str(label).strip().lower() == "success" else 1


# Data preparation
class SessionRecord:
    def __init__(self, block_id: str, label: str, events: List[str], times: List[float], latency: float):
        self.block_id = block_id
        self.label = label
        self.events = events
        self.times = times
        self.latency = latency


def load_sessions(csv_path: str) -> List[SessionRecord]:
    df = pd.read_csv(csv_path)
    required = {"BlockId", "Label", "Features", "TimeInterval"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    sessions: List[SessionRecord] = []
    for _, row in df.iterrows():
        events = parse_list_cell(row["Features"])
        times = parse_list_cell(row["TimeInterval"])
        latency = float(row["Latency"]) if "Latency" in df.columns and pd.notna(row["Latency"]) else float(sum(times))

        if len(events) < 2:
            continue
        # In many HDFS session files, len(TimeInterval) can be len(events)-1.
        # We align it to len(events) by prepending 0.0 if needed.
        if len(times) == len(events) - 1:
            times = [0.0] + [float(t) for t in times]
        elif len(times) != len(events):
            # fallback: pad or trim to match
            times = [float(t) for t in times]
            if len(times) < len(events):
                times = times + [0.0] * (len(events) - len(times))
            else:
                times = times[: len(events)]

        events = [str(e).strip() for e in events]
        times = [float(t) for t in times]

        sessions.append(
            SessionRecord(
                block_id=str(row["BlockId"]),
                label=str(row["Label"]),
                events=events,
                times=times,
                latency=latency,
            )
        )
    return sessions


class Vocab:
    PAD = "<PAD>"
    UNK = "<UNK>"

    def __init__(self, tokens: Sequence[str]):
        uniq = [self.PAD, self.UNK] + sorted(set(tokens))
        self.stoi = {t: i for i, t in enumerate(uniq)}
        self.itos = {i: t for t, i in self.stoi.items()}

    def encode(self, token: str) -> int:
        return self.stoi.get(token, self.stoi[self.UNK])

    def decode(self, idx: int) -> str:
        return self.itos[idx]

    def __len__(self) -> int:
        return len(self.stoi)


def split_normal_sessions(sessions: List[SessionRecord], cfg: Config):
    normal = [s for s in sessions if safe_label_to_binary(s.label) == 0]
    random.shuffle(normal)
    n = len(normal)
    n_train = int(n * cfg.train_ratio)
    n_valid = int(n * cfg.valid_ratio)
    train = normal[:n_train]
    valid = normal[n_train : n_train + n_valid]
    test_norm = normal[n_train + n_valid :]
    abnormal = [s for s in sessions if safe_label_to_binary(s.label) == 1]
    test = test_norm + abnormal
    return train, valid, test


# Path anomaly dataset/model
class PathWindowDataset(Dataset):
    def __init__(self, sessions: List[SessionRecord], vocab: Vocab, history_size: int):
        self.samples = []
        self.history_size = history_size
        pad_id = vocab.encode(Vocab.PAD)

        for session in sessions:
            encoded = [vocab.encode(e) for e in session.events]
            for i in range(1, len(encoded)):
                left = max(0, i - history_size)
                hist = encoded[left:i]
                if len(hist) < history_size:
                    hist = [pad_id] * (history_size - len(hist)) + hist
                target = encoded[i]
                self.samples.append((hist, target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        hist, target = self.samples[idx]
        return torch.tensor(hist, dtype=torch.long), torch.tensor(target, dtype=torch.long)


class PathLSTM(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        # print(x)
        emb = self.embedding(x)
        out, _ = self.lstm(emb)
        last = out[:, -1, :]
        logits = self.fc(last)
        return logits


# Timing anomaly dataset/model
class TimeWindowDataset(Dataset):
    def __init__(self, sessions: List[SessionRecord], vocab: Vocab, history_size: int):
        self.samples = []
        pad_id = vocab.encode(Vocab.PAD)
        for session in sessions:
            ev = [vocab.encode(e) for e in session.events]
            tm = list(session.times)
            for i in range(1, len(ev)):
                left = max(0, i - history_size)
                hist_e = ev[left:i]
                hist_t = tm[left:i]
                if len(hist_e) < history_size:
                    pad_len = history_size - len(hist_e)
                    hist_e = [pad_id] * pad_len + hist_e
                    hist_t = [0.0] * pad_len + hist_t
                next_event = ev[i]
                target_time = float(tm[i])
                self.samples.append((hist_e, hist_t, next_event, target_time))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        hist_e, hist_t, next_event, target_time = self.samples[idx]
        return (
            torch.tensor(hist_e, dtype=torch.long),
            torch.tensor(hist_t, dtype=torch.float32),
            torch.tensor(next_event, dtype=torch.long),
            torch.tensor(target_time, dtype=torch.float32),
        )


class TimeLSTM(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim + 1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, hist_events, hist_times, next_event):
        # print(hist_times)
        emb_hist = self.embedding(hist_events)
        x = torch.cat([emb_hist, hist_times.unsqueeze(-1)], dim=-1)
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        emb_next = self.embedding(next_event)
        pred = self.head(torch.cat([last, emb_next], dim=-1)).squeeze(-1)
        return pred


# Training helpers
def train_path_model(model, train_loader, valid_loader, cfg: Config):
    device = cfg.device
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.path_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
    criterion = nn.CrossEntropyLoss()

    best_state = None
    best_valid = float("inf")

    for epoch in range(cfg.path_epochs):
        model.train()
        train_losses = []
        for hist, target in train_loader:
            hist = hist.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            logits = model(hist)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()

        model.eval()
        valid_losses = []
        with torch.no_grad():
            for hist, target in valid_loader:
                hist = hist.to(device)
                target = target.to(device)
                logits = model(hist)
                loss = criterion(logits, target)
                valid_losses.append(loss.item())

        mean_valid = float(np.mean(valid_losses)) if valid_losses else float("inf")
        print(f"[Path] epoch={epoch+1:02d} train_loss={np.mean(train_losses):.6f} valid_loss={mean_valid:.6f}")
        if mean_valid < best_valid:
            best_valid = mean_valid
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def train_time_model(model, train_loader, valid_loader, cfg: Config):
    device = cfg.device
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.time_lr)
    criterion = nn.MSELoss()

    best_state = None
    best_valid = float("inf")

    for epoch in range(cfg.time_epochs):
        break
        model.train()
        train_losses = []
        for hist_e, hist_t, next_e, target_t in train_loader:
            hist_e = hist_e.to(device)
            hist_t = hist_t.to(device)
            next_e = next_e.to(device)
            target_t = target_t.to(device)

            optimizer.zero_grad()
            pred = model(hist_e, hist_t, next_e)
            loss = criterion(pred, target_t)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        valid_losses = []
        with torch.no_grad():
            for hist_e, hist_t, next_e, target_t in valid_loader:
                hist_e = hist_e.to(device)
                hist_t = hist_t.to(device)
                next_e = next_e.to(device)
                target_t = target_t.to(device)
                pred = model(hist_e, hist_t, next_e)
                loss = criterion(pred, target_t)
                valid_losses.append(loss.item())

        mean_valid = float(np.mean(valid_losses)) if valid_losses else float("inf")
        print(f"[Time] epoch={epoch+1:02d} train_loss={np.mean(train_losses):.6f} valid_loss={mean_valid:.6f}")
        if mean_valid < best_valid:
            best_valid = mean_valid
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# Threshold calibration
def collect_timing_errors(model: TimeLSTM, sessions: List[SessionRecord], vocab: Vocab, cfg: Config) -> np.ndarray:
    device = cfg.device
    model.eval()
    pad_id = vocab.encode(Vocab.PAD)
    errors = []

    with torch.no_grad():
        for session in sessions:
            ev = [vocab.encode(e) for e in session.events]
            tm = list(session.times)
            for i in range(1, len(ev)):
                left = max(0, i - cfg.history_size)
                hist_e = ev[left:i]
                hist_t = tm[left:i]
                if len(hist_e) < cfg.history_size:
                    pad_len = cfg.history_size - len(hist_e)
                    hist_e = [pad_id] * pad_len + hist_e
                    hist_t = [0.0] * pad_len + hist_t

                hist_e_t = torch.tensor(hist_e, dtype=torch.long, device=device).unsqueeze(0)
                hist_t_t = torch.tensor(hist_t, dtype=torch.float32, device=device).unsqueeze(0)
                next_e_t = torch.tensor([ev[i]], dtype=torch.long, device=device)
                target_t = float(tm[i])
                pred_t = float(model(hist_e_t, hist_t_t, next_e_t).item())
                err = (pred_t - target_t) ** 2
                errors.append(err)

    return np.array(errors, dtype=np.float64)


# Detection / evaluation
def detect_session(
    session: SessionRecord,
    path_model: PathLSTM,
    time_model: TimeLSTM,
    vocab: Vocab,
    cfg: Config,
    timing_threshold: float,
) -> Dict:
    device = cfg.device
    pad_id = vocab.encode(Vocab.PAD)
    path_model.eval()
    time_model.eval()

    ev = [vocab.encode(e) for e in session.events]
    tm = list(session.times)
    step_results = []

    with torch.no_grad():
        for i in range(1, len(ev)):
            left = max(0, i - cfg.history_size)
            hist_e = ev[left:i]
            hist_t = tm[left:i]
            if len(hist_e) < cfg.history_size:
                pad_len = cfg.history_size - len(hist_e)
                hist_e = [pad_id] * pad_len + hist_e
                hist_t = [0.0] * pad_len + hist_t

            # path anomaly
            hist_e_t = torch.tensor(hist_e, dtype=torch.long, device=device).unsqueeze(0)
            logits = path_model(hist_e_t)
            probs = torch.softmax(logits, dim=-1).squeeze(0)
            sorted_probs, sorted_ids = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=0)
            nucleus_mask = cumulative - sorted_probs < cfg.top_p
            nucleus_ids = sorted_ids[nucleus_mask].tolist()
            path_anomaly = ev[i] not in nucleus_ids

            # timing anomaly
            hist_t_t = torch.tensor(hist_t, dtype=torch.float32, device=device).unsqueeze(0)
            next_e_t = torch.tensor([ev[i]], dtype=torch.long, device=device)
            pred_time = float(time_model(hist_e_t, hist_t_t, next_e_t).item())
            err = (pred_time - float(tm[i])) ** 2
            # time_anomaly = err > timing_threshold
            time_anomaly = False

            step_results.append(
                {
                    "step_index": i,
                    "actual_event": vocab.decode(ev[i]),
                    "path_anomaly": bool(path_anomaly),
                    "actual_time": float(tm[i]),
                    "pred_time": pred_time,
                    "time_mse": err,
                    "time_anomaly": bool(time_anomaly),
                    "combined_anomaly": bool(path_anomaly or time_anomaly),
                }
            )


    session_pred = int(any(x["combined_anomaly"] for x in step_results))
    return {
        "BlockId": session.block_id,
        "true_label": safe_label_to_binary(session.label),
        "pred_label": session_pred,
        "latency": session.latency,
        "num_steps": len(step_results),
        "num_path_anomalies": sum(int(x["path_anomaly"]) for x in step_results),
        "num_time_anomalies": sum(int(x["time_anomaly"]) for x in step_results),
        "step_results": step_results,
    }


# Main pipeline
def main(cfg: Config):
    set_seed(cfg.seed)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading sessions...")
    sessions = load_sessions(cfg.csv_path)
    if not sessions:
        raise ValueError("No usable sessions found in the CSV.")

    print(f"Total sessions: {len(sessions)}")
    train_sessions, valid_sessions, test_sessions = split_normal_sessions(sessions, cfg)
    print(f"Normal train: {len(train_sessions)} | normal valid: {len(valid_sessions)} | test(all): {len(test_sessions)}")

    # Build vocabulary only from normal training data.
    train_tokens = [e for s in train_sessions for e in s.events]
    vocab = Vocab(train_tokens)
    print(f"Vocab size: {len(vocab)}")

    # Normalize time values using only normal training data.
    train_times = np.array([np.log1p(t) for s in train_sessions for t in s.times], dtype=np.float32)
    time_mean = float(train_times.mean())
    time_std = float(train_times.std() + 1e-8)

    for group in [train_sessions, valid_sessions, test_sessions]:
        for s in group:
            s.times = [float((np.log1p(t) - time_mean) / time_std) for t in s.times]

    # Datasets / loaders
    path_train_ds = PathWindowDataset(train_sessions, vocab, cfg.history_size)
    path_valid_ds = PathWindowDataset(valid_sessions, vocab, cfg.history_size)
    time_train_ds = TimeWindowDataset(train_sessions, vocab, cfg.history_size)
    time_valid_ds = TimeWindowDataset(valid_sessions, vocab, cfg.history_size)

    path_train_loader = DataLoader(path_train_ds, batch_size=cfg.path_batch_size, shuffle=True)
    path_valid_loader = DataLoader(path_valid_ds, batch_size=cfg.path_batch_size, shuffle=False)
    time_train_loader = DataLoader(time_train_ds, batch_size=cfg.time_batch_size, shuffle=True)
    time_valid_loader = DataLoader(time_valid_ds, batch_size=cfg.time_batch_size, shuffle=False)

    # Models
    path_model = PathLSTM(
        vocab_size=len(vocab),
        embed_dim=cfg.event_embed_dim,
        hidden_dim=cfg.path_hidden_dim,
        num_layers=cfg.path_num_layers,
        dropout=cfg.path_dropout,
    )
    time_model = TimeLSTM(
        vocab_size=len(vocab),
        embed_dim=cfg.event_embed_dim,
        hidden_dim=cfg.time_hidden_dim,
        num_layers=cfg.time_num_layers,
        dropout=cfg.time_dropout,
    )

    print("Training path model...")
    path_model = train_path_model(path_model, path_train_loader, path_valid_loader, cfg)

    print("Training time model...")
    time_model = train_time_model(time_model, time_train_loader, time_valid_loader, cfg)

    print("Calibrating timing threshold on normal validation data...")
    valid_errors = collect_timing_errors(time_model, valid_sessions, vocab, cfg)
    timing_threshold = float(np.quantile(valid_errors, cfg.timing_threshold_quantile))
    print(f"Timing threshold (quantile={cfg.timing_threshold_quantile}): {timing_threshold:.6f}")

    # Evaluate on test sessions.
    print("Running detection...")
    session_outputs = []
    y_true, y_pred = [], []
    for s in test_sessions:
        result = detect_session(s, path_model, time_model, vocab, cfg, timing_threshold)
        session_outputs.append(result)
        y_true.append(result["true_label"])
        y_pred.append(result["pred_label"])

    print("Block-level evaluation:")
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=["Success", "Anomaly"], digits=4))

    # false_negatives = [
    #     x for x in session_outputs
    #     if x["true_label"] == 1 and x["pred_label"] == 0
    # ]

    # print(f"False negatives: {len(false_negatives)}")

    # for x in false_negatives[:10]:
    #     print("BlockId:", x["BlockId"])
    #     print("true_label:", x["true_label"], "pred_label:", x["pred_label"])
    #     print("num_steps:", x["num_steps"])
    #     print("num_path_anomalies:", x["num_path_anomalies"])
    #     print("num_time_anomalies:", x["num_time_anomalies"])

    # Save artifacts.
    torch.save(path_model.state_dict(), out_dir / "path_model.pt")
    torch.save(time_model.state_dict(), out_dir / "time_model.pt")
    with open(out_dir / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab.stoi, f, ensure_ascii=False, indent=2)
    with open(out_dir / "normalization.json", "w", encoding="utf-8") as f:
        json.dump({"time_mean": time_mean, "time_std": time_std, "timing_threshold": timing_threshold}, f, indent=2)
    with open(out_dir / "session_predictions.json", "w", encoding="utf-8") as f:
        json.dump(session_outputs, f, ensure_ascii=False, indent=2)

    print(f"Saved outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    cfg = Config(
        csv_path="/Users/liyakul/project/output_with_eof.csv",
        output_dir="deeplog_output",
    )
    main(cfg)
