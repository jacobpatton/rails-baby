import argparse
import json
import re
from pathlib import Path


def read_text(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16")
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw.decode("utf-8-sig", errors="replace")
    return raw.decode("utf-8", errors="replace")


def extract_last_json(text: str):
    return extract_last_json_matching(text, must_have_keys=[])


def extract_last_json_matching(text: str, must_have_keys: list[str]):
    decoder = json.JSONDecoder()
    candidates = [m.start() for m in re.finditer(r"(\{|\[)", text)]
    for start in reversed(candidates):
        try:
            obj, end = decoder.raw_decode(text[start:])
        except Exception:
            continue
        rest = text[start + end :]
        # Accept the last JSON value in the file, even if followed by non-JSON noise
        # (e.g., "tokens used ..."), as long as no later JSON start exists.
        if re.search(r"(\{|\[)", rest) is None:
            if must_have_keys:
                if not isinstance(obj, dict):
                    continue
                if any(k not in obj for k in must_have_keys):
                    continue
            return obj
    raise ValueError("No JSON object/array found that ends at EOF.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True)
    parser.add_argument("--out", dest="out_path", required=True)
    parser.add_argument(
        "--must-have-keys",
        default="",
        help="Comma-separated list of keys that must exist on the extracted JSON object.",
    )
    args = parser.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)

    text = read_text(in_path)
    must_have_keys = [k for k in args.must_have_keys.split(",") if k.strip()]
    obj = extract_last_json_matching(text, must_have_keys=must_have_keys)
    out_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
