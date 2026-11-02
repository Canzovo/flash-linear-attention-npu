"""Sample an unchanged ATK accuracy subset and record how to reproduce it."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import secrets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        parser.error("The sample must not overwrite the candidate pool.")
    source_bytes = args.source.read_bytes()
    cases = json.loads(source_bytes)
    if not isinstance(cases, list) or not 1 <= args.count <= len(cases):
        parser.error("count must be between 1 and the candidate pool size.")
    if len({case["id"] for case in cases}) != len(cases):
        parser.error("Candidate case IDs must be unique.")
    seed = secrets.randbits(64) if args.seed is None else args.seed
    positions = sorted(random.Random(seed).sample(range(len(cases)), args.count))
    selected = [cases[position] for position in positions]
    # Keep original ATK IDs, payloads, input seeds, ranges and comparison rules.
    output_bytes = (json.dumps(selected, ensure_ascii=False) + "\n").encode("utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(output_bytes)
    manifest = {
        "candidate_count": len(cases),
        "sample_count": len(selected),
        "sampling_seed": seed,
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "sample_sha256": hashlib.sha256(output_bytes).hexdigest(),
        "selected_case_ids": [case["id"] for case in selected],
        "payloads_unchanged": True,
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
