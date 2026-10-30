"""Fine-tuning a transformer (XLM-R / RoBERTa / HerBERT) for user-level classification.

Design (kept simple and leak-free):
  * the unit of the split is the USER (train / val / test are disjoint sets of users);
  * each user's text is cut into word-chunks (<= max_len tokens), the model is trained on chunks
    carrying the user's label, and chunk probabilities are aggregated per user (mean or max);
  * the decision threshold is chosen on the validation users, never on the test set;
  * ROC-AUC / average precision are reported too, because they do not depend on the threshold
    (important for zero-shot transfer EN -> PL, where calibration usually shifts).

torch / transformers are imported lazily so the helpers can be tested without them.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split


@dataclass
class TConfig:
    model_name: str = "xlm-roberta-base"  # EN only: "roberta-base"; PL only: "allegro/herbert-base-cased"
    max_len: int = 256
    words_per_chunk: int = 120  # ~160-200 subword tokens, fits max_len
    max_chunks_per_user: int = 20
    batch_size: int = 16
    lr: float = 2e-5
    epochs: int = 3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    seed: int = 42
    agg: str = "mean"  # "mean" | "max"


# ---------------------------------------------------------------- pure helpers (no torch)

def split_users(users: pd.DataFrame, val_frac: float = 0.15, test_frac: float = 0.15, seed: int = 42):
    """Stratified, disjoint train/val/test split over users (one row per user)."""
    if users["user_id"].duplicated().any():
        raise ValueError("users must have one row per user_id")
    train_val, test = train_test_split(users, test_size=test_frac, stratify=users["label"], random_state=seed)
    rel = val_frac / (1.0 - test_frac)
    train, val = train_test_split(train_val, test_size=rel, stratify=train_val["label"], random_state=seed)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def chunk_users(users: pd.DataFrame, words_per_chunk: int = 120, max_chunks: int | None = 20) -> pd.DataFrame:
    """Cut each user's text into word windows. If a user has more than max_chunks, pick chunks
    evenly spaced over their whole history (deterministic, avoids using only the earliest text)."""
    rows = []
    for r in users.itertuples(index=False):
        words = str(r.text).split()
        chunks = [" ".join(words[i : i + words_per_chunk]) for i in range(0, len(words), words_per_chunk)]
        if not chunks:
            chunks = [""]
        if max_chunks and len(chunks) > max_chunks:
            idx = np.unique(np.linspace(0, len(chunks) - 1, max_chunks).round().astype(int))
            chunks = [chunks[i] for i in idx]
        for c in chunks:
            rows.append({"user_id": r.user_id, "label": int(r.label), "text": c})
    return pd.DataFrame(rows, columns=["user_id", "label", "text"])


def aggregate_scores(chunks: pd.DataFrame, probs, how: str = "mean") -> pd.DataFrame:
    """Chunk-level probabilities -> one score per user."""
    if how not in ("mean", "max"):
        raise ValueError("how must be 'mean' or 'max'")
    df = chunks[["user_id", "label"]].copy()
    df["p"] = np.asarray(probs, dtype=float)
    g = df.groupby("user_id", sort=True)
    score = g["p"].mean() if how == "mean" else g["p"].max()
    return pd.DataFrame({"label": g["label"].first().astype(int), "score": score}).reset_index()


def best_threshold(y, score) -> float:
    """Threshold maximising F1 on the given (validation) users."""
    y, score = np.asarray(y), np.asarray(score)
    best_t, best_f1 = 0.5, -1.0
    for t in np.unique(score):
        f1 = f1_score(y, (score >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t


def user_metrics(y, score, threshold: float) -> dict:
    y, score = np.asarray(y), np.asarray(score)
    pred = (score >= threshold).astype(int)
    two_classes = len(np.unique(y)) == 2
    return {
        "n_users": int(len(y)),
        "positives": int(y.sum()),
        "threshold": float(threshold),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, score)) if two_classes else float("nan"),
        "avg_precision": float(average_precision_score(y, score)) if two_classes else float("nan"),
    }


# ---------------------------------------------------------------- torch part (lazy imports)

def _set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _predict_chunks(model, tok, texts: list[str], cfg: TConfig, device) -> np.ndarray:
    import torch

    model.eval()
    out = []
    bs = cfg.batch_size * 2
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            enc = tok(texts[i : i + bs], truncation=True, max_length=cfg.max_len,
                      padding=True, return_tensors="pt").to(device)
            out.append(torch.softmax(model(**enc).logits, dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(out) if out else np.array([])


def score_users(model, tok, users: pd.DataFrame, cfg: TConfig, device) -> pd.DataFrame:
    chunks = chunk_users(users, cfg.words_per_chunk, cfg.max_chunks_per_user)
    probs = _predict_chunks(model, tok, chunks["text"].tolist(), cfg, device)
    return aggregate_scores(chunks, probs, cfg.agg)


def fit(train_users: pd.DataFrame, val_users: pd.DataFrame, cfg: TConfig, device: str | None = None):
    """Fine-tune and return (model, tokenizer, threshold, val_metrics, history).

    The best epoch is selected by user-level validation ROC-AUC; the threshold by validation F1.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    _set_seed(cfg.seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name, num_labels=2).to(device)

    tr = chunk_users(train_users, cfg.words_per_chunk, cfg.max_chunks_per_user)
    texts = tr["text"].tolist()
    y = torch.tensor(tr["label"].to_numpy(), dtype=torch.long)
    n = len(texts)
    counts = np.bincount(tr["label"].to_numpy(), minlength=2)
    weights = torch.tensor(n / (2.0 * np.maximum(counts, 1)), dtype=torch.float, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total = math.ceil(n / cfg.batch_size) * cfg.epochs
    sched = get_linear_schedule_with_warmup(opt, int(cfg.warmup_ratio * total), total)
    gen = torch.Generator().manual_seed(cfg.seed)

    best_auc, best_state, best_val, history = -1.0, None, None, []
    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(n, generator=gen).tolist()
        running = 0.0
        for i in range(0, n, cfg.batch_size):
            idx = perm[i : i + cfg.batch_size]
            enc = tok([texts[j] for j in idx], truncation=True, max_length=cfg.max_len,
                      padding=True, return_tensors="pt").to(device)
            loss = loss_fn(model(**enc).logits, y[idx].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            running += float(loss) * len(idx)

        val = score_users(model, tok, val_users, cfg, device)
        auc = roc_auc_score(val["label"], val["score"]) if val["label"].nunique() == 2 else float("nan")
        history.append({"epoch": epoch + 1, "train_loss": running / n, "val_roc_auc": float(auc)})
        if best_state is None or (not math.isnan(auc) and auc > best_auc):
            best_auc, best_val = auc, val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    thr = best_threshold(best_val["label"], best_val["score"])
    return model, tok, thr, user_metrics(best_val["label"], best_val["score"], thr), history
