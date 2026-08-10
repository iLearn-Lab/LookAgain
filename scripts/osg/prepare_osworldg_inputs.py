#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def str_to_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def run_converter(
    *,
    suite_root: Path,
    dataset_json: Path,
    image_dir: Path,
    instruction_key: str,
    converted_json: Path,
    include_refusal: bool,
    strict_image_check: bool,
    max_samples: str,
    sample_ids: str,
) -> None:
    args = [
        sys.executable,
        str(suite_root / "data_transfer" / "run_multiturn_data_transfer.py"),
        "--benchmark",
        "osworld-g",
        "--dataset-json",
        str(dataset_json),
        "--image-dir",
        str(image_dir),
        "--instruction-key",
        instruction_key,
        "--converted-json",
        str(converted_json),
        "--prepare-only",
    ]
    if include_refusal:
        args.append("--include-refusal")
    if strict_image_check:
        args.append("--strict-image-check")
    if max_samples:
        args.extend(["--max-samples", max_samples])
    if sample_ids:
        args.extend(["--sample-ids", sample_ids])

    converted_json.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(args, check=True)


def is_refusal(sample: dict[str, Any]) -> bool:
    elements = sample.get("elements") or []
    if not elements:
        return False
    el = elements[0]
    if (el.get("task_type") or "").lower() == "refusal":
        return True
    if (el.get("data_type") or "").lower() == "refusal":
        return True
    if el.get("bbox") in (None, []):
        return True
    return False


def write_refusal_only(src: Path, dst: Path) -> None:
    with src.open("r", encoding="utf-8") as fh:
        samples = json.load(fh)
    refusal = [sample for sample in samples if is_refusal(sample)]

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as fh:
        json.dump(refusal, fh, ensure_ascii=False, indent=2)

    print(f"[prepare] refusal-only converted JSON: {dst} | total={len(samples)} | refusal={len(refusal)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare OSWorld-G inputs for SFT tool-call evaluation.")
    parser.add_argument("--suite-root", required=True, type=Path)
    parser.add_argument("--dataset-json", required=True, type=Path)
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--instruction-key", default="instruction")
    parser.add_argument("--full-input-json", required=True, type=Path)
    parser.add_argument("--refusal-full-json", required=True, type=Path)
    parser.add_argument("--refusal-input-json", required=True, type=Path)
    parser.add_argument("--strict-image-check", default="0")
    parser.add_argument("--max-samples", default="")
    parser.add_argument("--sample-ids", default="")
    parser.add_argument("--refusal-max-samples", default="")
    parser.add_argument("--refusal-sample-ids", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suite_root = args.suite_root.expanduser().resolve()
    dataset_json = args.dataset_json.expanduser().resolve()
    image_dir = args.image_dir.expanduser().resolve()
    strict_image_check = str_to_bool(args.strict_image_check)

    run_converter(
        suite_root=suite_root,
        dataset_json=dataset_json,
        image_dir=image_dir,
        instruction_key=args.instruction_key,
        converted_json=args.full_input_json,
        include_refusal=False,
        strict_image_check=strict_image_check,
        max_samples=args.max_samples,
        sample_ids=args.sample_ids,
    )
    print(f"[prepare] full converted JSON: {args.full_input_json}")

    run_converter(
        suite_root=suite_root,
        dataset_json=dataset_json,
        image_dir=image_dir,
        instruction_key=args.instruction_key,
        converted_json=args.refusal_full_json,
        include_refusal=True,
        strict_image_check=strict_image_check,
        max_samples=args.refusal_max_samples,
        sample_ids=args.refusal_sample_ids,
    )
    print(f"[prepare] full-with-refusal converted JSON: {args.refusal_full_json}")
    write_refusal_only(args.refusal_full_json, args.refusal_input_json)


if __name__ == "__main__":
    main()
