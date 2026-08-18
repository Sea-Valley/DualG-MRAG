"""Reservoir-sample N items with split=="val" from the WebQA train_val JSON.

Streams the file with ijson so only N items are held in memory.
"""

import argparse
import json
import os
import random
from typing import Optional


def reservoir_sample_val(
    input_path: str, n: int, seed: Optional[int] = None
) -> tuple[dict, int]:
    try:
        import ijson
    except ImportError as exc:
        raise SystemExit("Please install ijson first: pip install ijson") from exc

    if seed is not None:
        random.seed(seed)

    reservoir: list[tuple[str, dict]] = []
    val_count = 0

    with open(input_path, "rb") as f:
        for guid, item in ijson.kvitems(f, ""):
            if item.get("split") != "val":
                continue
            val_count += 1
            if len(reservoir) < n:
                reservoir.append((guid, item))
            else:
                j = random.randrange(val_count)
                if j < n:
                    reservoir[j] = (guid, item)

    return dict(reservoir), val_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Randomly sample N val-split items from the WebQA train_val JSON."
    )
    parser.add_argument(
        "--input", "-i", type=str, required=True, help="WebQA train_val JSON path"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="WebQA1000.json", help="Output JSON path"
    )
    parser.add_argument("--num", "-n", type=int, default=1000, help="Number of samples")
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    print(f"Sampling {args.num} items from val split, input: {args.input}")
    sampled, total_val = reservoir_sample_val(args.input, args.num, seed=args.seed)
    print(f"val total: {total_val}, sampled: {len(sampled)}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(sampled, f, ensure_ascii=False, indent=2)
    print(f"Written: {args.output}")


if __name__ == "__main__":
    main()
