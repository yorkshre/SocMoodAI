"""python scripts/run_baseline.py --data data/posts.csv [--per-lang]"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dd.baseline import ablation  # noqa: E402
from dd.data import load_posts  # noqa: E402
from dd.features import build_user_table  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--per-lang", action="store_true", help="report PL and EN separately")
    a = ap.parse_args()

    users = build_user_table(load_posts(a.data))
    print(f"users={len(users)}  positives={users['label'].sum()}  langs={users['lang'].value_counts().to_dict()}")
    print("\n== all ==\n", ablation(users, a.splits, a.seed).round(3))
    if a.per_lang:
        for lang, sub in users.groupby("lang"):
            if sub["label"].nunique() == 2 and len(sub) >= 2 * a.splits:
                print(f"\n== {lang} ==\n", ablation(sub, a.splits, a.seed).round(3))


if __name__ == "__main__":
    main()
