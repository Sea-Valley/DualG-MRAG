"""Randomly sample N questions from the MMQA dev jsonl to build MMQA1000.json."""

import argparse
import json
import random
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Randomly select N samples from the MMQA dev jsonl."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input MMQA dev jsonl file path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("MMQA1000.json"),
        help="Output json file path.",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=1000,
        help="Number of samples to select.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    records = load_jsonl(args.input)
    if len(records) < args.num:
        raise ValueError(
            f"Input has only {len(records)} records, cannot sample {args.num}."
        )

    sampled = random.sample(records, args.num)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(sampled, f, ensure_ascii=False, indent=2)

    print(f"Sampled {args.num} records from {args.input} -> {args.output}")


if __name__ == "__main__":
    main()
