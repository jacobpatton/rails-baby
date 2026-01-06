import argparse
import json
from pathlib import Path


REPLACEMENTS = {
    # Common mojibake sequences seen when UTF-8 punctuation is decoded as Windows-1252/Latin-1.
    "ÔÇô": "-",  # en dash
    "ÔÇö": "-",  # em dash
    "ÔÇª": "...",  # ellipsis
    "ÔÇÖ": "'",  # right single quote
    "ÔÇŒ": '"',  # left double quote (approx)
    "ÔÇ": '"',  # right double quote (approx)
    "ÔåÆ": " -> ",  # arrow-like separator in this repo outputs
}


def normalize_string(value: str) -> str:
    for bad, good in REPLACEMENTS.items():
        value = value.replace(bad, good)
    # Final safety: keep it ASCII-only for paste-ready launch copy.
    value = value.encode("ascii", errors="ignore").decode("ascii")
    # Collapse any accidental double spaces from replacements.
    value = " ".join(value.split())
    return value


def walk(value):
    if isinstance(value, str):
        return normalize_string(value)
    if isinstance(value, list):
        return [walk(v) for v in value]
    if isinstance(value, dict):
        return {k: walk(v) for k, v in value.items()}
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True)
    parser.add_argument("--out", dest="out_path", required=True)
    parser.add_argument("--indent", type=int, default=2)
    args = parser.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)

    obj = json.loads(in_path.read_text(encoding="utf-8"))
    obj = walk(obj)
    out_path.write_text(
        json.dumps(obj, ensure_ascii=True, indent=args.indent) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

