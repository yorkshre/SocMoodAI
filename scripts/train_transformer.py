"""Train on one dataset, evaluate on a held-out test split and optionally zero-shot on others.

  # English training, held-out English test, zero-shot Polish
  python scripts/train_transformer.py --train-data data/en.csv --eval-data pl=data/pl.csv \
      --model xlm-roberta-base --epochs 3 --out runs/xlmr_en
"""
import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dd.data import load_posts  # noqa: E402
from dd.features import build_user_table  # noqa: E402
from dd.transformer import TConfig, fit, score_users, split_users, user_metrics  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-data", required=True)
    ap.add_argument("--eval-data", action="append", default=[], metavar="NAME=CSV",
                    help="extra datasets evaluated with the threshold chosen on validation (repeatable)")
    ap.add_argument("--model", default="xlm-roberta-base")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--agg", choices=["mean", "max"], default="mean")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    ap.add_argument("--save-model", action="store_true")
    a = ap.parse_args()

    cfg = TConfig(model_name=a.model, epochs=a.epochs, max_len=a.max_len, batch_size=a.batch_size,
                  lr=a.lr, agg=a.agg, seed=a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    users = build_user_table(load_posts(a.train_data))
    train, val, test = split_users(users, seed=cfg.seed)
    print(f"users: train={len(train)} val={len(val)} test={len(test)}")

    model, tok, thr, val_metrics, history = fit(train, val, cfg)
    device = next(model.parameters()).device

    results = {"config": asdict(cfg), "history": history, "val": val_metrics, "test": None, "zero_shot": {}}
    s = score_users(model, tok, test, cfg, device)
    results["test"] = user_metrics(s["label"], s["score"], thr)
    for spec in a.eval_data:
        name, path = spec.split("=", 1)
        eu = build_user_table(load_posts(path))
        s = score_users(model, tok, eu, cfg, device)
        results["zero_shot"][name] = user_metrics(s["label"], s["score"], thr)

    (out / "results.json").write_text(json.dumps(results, indent=2))
    split_ids = {k: v["user_id"].astype(str).tolist() for k, v in zip(("train", "val", "test"), (train, val, test))}
    (out / "splits.json").write_text(json.dumps(split_ids))
    if a.save_model:
        model.save_pretrained(out / "model")
        tok.save_pretrained(out / "model")
    print(json.dumps({k: results[k] for k in ("val", "test", "zero_shot")}, indent=2))


if __name__ == "__main__":
    main()
