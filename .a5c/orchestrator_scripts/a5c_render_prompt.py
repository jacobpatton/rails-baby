import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--context-json", default="")
    parser.add_argument("--context-file", default="")
    args = parser.parse_args()

    template = Path(args.template).read_text(encoding="utf-8")
    if args.context_file:
        context = json.loads(Path(args.context_file).read_text(encoding="utf-8-sig"))
    else:
        context = json.loads(args.context_json)

    rendered = template.replace("{{task}}", args.task).replace(
        "{{context}}", json.dumps(context, ensure_ascii=False, indent=2)
    )
    Path(args.out).write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
