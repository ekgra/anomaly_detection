"""
Shared vectorisation module for log lines.

Implements a consistent HashingVectorizer configuration used across
training and inference so features are identical everywhere.

CLI can transform a log file into a sparse feature matrix and optionally
dump the vectoriser config (via joblib) for later reuse.

python scripts/vectorise.py --input logs/train_normal_100k.log --out models/features/train_X.pkl --dump-vectorizer models/vectoriser.pkl --dim 2048

"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Optional

from sklearn.feature_extraction.text import HashingVectorizer
import joblib


# Default vector space dimensionality; must match across all components
VEC_DIM = 2048


def get_vectorizer(vec_dim: int = VEC_DIM) -> HashingVectorizer:
    """Return a configured HashingVectorizer.

    Notes:
    - alternate_sign=False to keep non-negative features (better for some models)
    - norm=None to leave magnitudes as-is
    - Stateless: no fitting required; hash is stable given same config
    """

    return HashingVectorizer(
        n_features=vec_dim,
        alternate_sign=False,
        norm=None,
    )


def vectorize_lines(lines: Iterable[str], vec: Optional[HashingVectorizer] = None, *, vec_dim: int = VEC_DIM):
    """Vectorize an iterable of log lines into a sparse matrix (CSR)."""
    if vec is None:
        vec = get_vectorizer(vec_dim)
    return vec.transform(lines)


def _read_lines(path: Optional[Path]) -> List[str]:
    if path is None or str(path) == "-":
        return [ln.rstrip("\n") for ln in sys.stdin]
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return [ln.rstrip("\n") for ln in f]


def save_vectorizer(vec: HashingVectorizer, out_path: Path) -> None:
    """Persist the vectorizer via joblib (produces .pkl)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vec, out_path)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Vectorize log lines using HashingVectorizer")
    p.add_argument("--input", "-i", type=str, default="-", help="Input log file path or '-' for stdin")
    p.add_argument("--out", "-o", type=str, default=None, help="Output .pkl path to dump sparse matrix via joblib")
    p.add_argument("--dim", type=int, default=VEC_DIM, help="Vector dimension (n_features)")
    p.add_argument("--dump-vectorizer", type=str, default=None, help="Optional path to save vectoriser.pkl via joblib")

    args = p.parse_args(argv)

    vec = get_vectorizer(args.dim)
    lines = _read_lines(Path(args.input) if args.input else None)

    X = vectorize_lines(lines, vec)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(X, out_path)

    if args.dump_vectorizer:
        save_vectorizer(vec, Path(args.dump_vectorizer))

    # Print a small summary for interactive runs
    print(f"Vectorized {X.shape[0]} lines into {X.shape[1]}-dim features (CSR)")
    if not args.out:
        print("Hint: pass --out features.pkl to persist the matrix")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

