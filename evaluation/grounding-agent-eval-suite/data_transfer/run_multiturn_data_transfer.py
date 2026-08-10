#!/usr/bin/env python3
"""Convert local GUI grounding benchmarks to SFT tool-call input JSON."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


SUITE_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = SUITE_ROOT.parents[1]

BENCHMARK_NAMES = ("osworld-g", "screenspot-pro", "ui-vision", "mmbench-gui", "venus-bench-gd")
VENUS_DEFAULT_CATEGORY_FILES = (
    "element_grounding.json",
    "functional_grounding.json",
    "reason_grounding.json",
    "spatial_grounding.json",
    "visual_grounding.json",
)

UI_TASKSET_FILES = {
    "element_basic": Path("annotations/element_grounding/element_grounding_basic.json"),
    "element_functional": Path("annotations/element_grounding/element_grounding_functional.json"),
    "element_spatial": Path("annotations/element_grounding/element_grounding_spatial.json"),
    "layout": Path("annotations/layout_grounding/layout_grounding.json"),
}
UI_TASKSET_ALIASES = {
    "basic": "element_basic",
    "functional": "element_functional",
    "spatial": "element_spatial",
    "element": "element_basic,element_functional,element_spatial",
    "element_grounding": "element_basic,element_functional,element_spatial",
    "layout_grounding": "layout",
    "all": "element_basic,element_functional,element_spatial,layout",
}


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def benchmark_root(name: str) -> Path:
    return BENCHMARK_ROOT / name


def default_converted_json(name: str) -> Path:
    return SUITE_ROOT / "data_transfer" / ".generated" / f"{name.replace('-', '_')}_multiturn_input.json"


def get_image_size(path: Path, fallback: Iterable[Any] | None = None) -> tuple[int, int] | None:
    if fallback is not None:
        values = list(fallback)
        if len(values) == 2:
            try:
                width, height = int(values[0]), int(values[1])
                if width > 0 and height > 0:
                    return width, height
            except (TypeError, ValueError):
                pass
    if not path.is_file():
        return None
    with Image.open(path) as img:
        return int(img.width), int(img.height)


def normalize_xyxy(values: Iterable[Any], image_size: tuple[int, int]) -> list[float]:
    coords = [float(v) for v in values]
    if len(coords) != 4:
        raise ValueError(f"bbox must have 4 values, got {coords!r}")
    if max(abs(v) for v in coords) <= 1.5:
        x1, y1, x2, y2 = coords
    else:
        width, height = float(image_size[0]), float(image_size[1])
        x1, y1, x2, y2 = coords[0] / width, coords[1] / height, coords[2] / width, coords[3] / height
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


def normalize_xywh(values: Iterable[Any], image_size: tuple[int, int]) -> list[float]:
    coords = [float(v) for v in values]
    if len(coords) != 4:
        raise ValueError(f"xywh bbox must have 4 values, got {coords!r}")
    x, y, w, h = coords
    if max(abs(v) for v in coords) <= 1.5:
        return [x, y, x + w, y + h]
    width, height = float(image_size[0]), float(image_size[1])
    return [x / width, y / height, (x + w) / width, (y + h) / height]


def normalize_polygon_bbox(values: Iterable[Any], image_size: tuple[int, int]) -> list[float]:
    coords = [float(v) for v in values]
    if len(coords) < 4 or len(coords) % 2:
        raise ValueError(f"polygon coordinates must be even-length x/y values, got {coords!r}")
    if max(abs(v) for v in coords) <= 1.5:
        return coords
    width, height = float(image_size[0]), float(image_size[1])
    out: list[float] = []
    for idx, value in enumerate(coords):
        out.append(value / width if idx % 2 == 0 else value / height)
    return out


def make_sample(
    *,
    image_path: Path,
    instruction: str,
    bbox_norm01: list[float] | None,
    benchmark: str,
    source_id: str,
    task_type: str,
    image_size: tuple[int, int],
    data_type: str = "bbox",
    extra_sample: dict[str, Any] | None = None,
    extra_element: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sample_extra = extra_sample or {}
    element_extra = extra_element or {}
    return {
        "img_filename": str(image_path),
        "data_source": benchmark,
        "image_size": [image_size[0], image_size[1]],
        "app_name": str(sample_extra.get("app_name") or sample_extra.get("application") or sample_extra.get("platform") or benchmark),
        "grounding_type": task_type,
        # Reuse runner-preserved Venus metadata slots so output rows retain stable source ids.
        "venus_id": source_id,
        "venus_category": task_type,
        "venus_label": str(sample_extra.get("label") or sample_extra.get("ui_type") or sample_extra.get("element_type") or ""),
        "elements": [
            {
                "instruction": instruction,
                "bbox": bbox_norm01,
                "data_type": data_type,
                "task_type": task_type,
                "venus_id": source_id,
                "venus_category": task_type,
                "venus_label": str(element_extra.get("label") or sample_extra.get("label") or ""),
            }
        ],
    }


def convert_osworld_g(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = benchmark_root("osworld-g")
    dataset_json = (args.dataset_json and Path(args.dataset_json).expanduser()) or (root / "OSWorld-G_refined.json")
    image_dir = (args.image_dir and Path(args.image_dir).expanduser()) or (root / "images")
    records = json.loads(dataset_json.resolve().read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"OSWorld-G dataset must be a list: {dataset_json}")

    allow_ids = set(_split_csv(args.sample_ids))
    samples: list[dict[str, Any]] = []
    skipped_refusal = 0
    skipped_bad = 0
    skipped_missing_image = 0
    skipped_by_filter = 0
    per_type: Counter[str] = Counter()
    missing_images: list[str] = []

    for index, item in enumerate(records):
        if not isinstance(item, dict):
            skipped_bad += 1
            continue
        source_id = str(item.get("id") or f"osworld-g:{index}")
        if allow_ids and source_id not in allow_ids and str(index) not in allow_ids:
            skipped_by_filter += 1
            continue
        instruction = str(item.get(args.instruction_key) or item.get("instruction") or "").strip()
        image_rel = str(item.get("image_path") or "").strip()
        image_path = (image_dir / image_rel).resolve() if image_rel else Path("")
        image_size = get_image_size(image_path, item.get("image_size"))
        box_type = str(item.get("box_type") or "bbox")
        coords = item.get("box_coordinates")
        if not instruction or not image_rel or image_size is None:
            if image_rel and image_size is None:
                skipped_missing_image += 1
                missing_images.append(str(image_path))
            else:
                skipped_bad += 1
            continue

        bbox_norm: list[float] | None
        if box_type == "refusal":
            if not args.include_refusal:
                skipped_refusal += 1
                continue
            bbox_norm = None
        else:
            try:
                if box_type == "polygon":
                    bbox_norm = normalize_polygon_bbox(coords, image_size)
                else:
                    bbox_norm = normalize_xywh(coords, image_size)
            except Exception:
                skipped_bad += 1
                continue

        samples.append(
            make_sample(
                image_path=image_path,
                instruction=instruction,
                bbox_norm01=bbox_norm,
                benchmark="osworld-g",
                source_id=source_id,
                task_type=box_type,
                image_size=image_size,
                data_type=box_type,
                extra_sample={
                    "app_name": ",".join(str(x) for x in item.get("GUI_types") or []),
                    "label": box_type,
                },
            )
        )
        per_type[box_type] += 1
        if args.max_samples is not None and len(samples) >= max(0, args.max_samples):
            break

    if missing_images and args.strict_image_check:
        raise FileNotFoundError(f"Missing {len(missing_images)} OSWorld-G images, first={missing_images[0]}")
    return samples, {
        "benchmark": "osworld-g",
        "dataset_json": str(dataset_json.resolve()),
        "image_dir": str(image_dir.resolve()),
        "num_records": len(records),
        "num_samples": len(samples),
        "num_skipped_refusal": skipped_refusal,
        "num_skipped_bad_record": skipped_bad,
        "num_skipped_missing_image": skipped_missing_image,
        "num_skipped_by_filter": skipped_by_filter,
        "per_type_counts": dict(sorted(per_type.items())),
        "include_refusal": bool(args.include_refusal),
    }


def convert_screenspot_pro(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = benchmark_root("screenspot-pro")
    annotations_dir = (args.annotations_dir and Path(args.annotations_dir).expanduser()) or (root / "annotations")
    image_dir = (args.image_dir and Path(args.image_dir).expanduser()) or (root / "images")
    split = args.split or "overall"
    if split == "overall":
        ann_files = sorted(annotations_dir.glob("*.json"))
    else:
        ann_files = []
        for raw in _split_csv(split):
            path = Path(raw).expanduser()
            ann_files.append(path if path.is_absolute() else annotations_dir / f"{raw.removesuffix('.json')}.json")
    if not ann_files:
        raise FileNotFoundError(f"No ScreenSpot-Pro annotation files under {annotations_dir}")

    allow_ids = set(_split_csv(args.sample_ids))
    samples: list[dict[str, Any]] = []
    missing_images: list[str] = []
    skipped_bad = 0
    skipped_missing_image = 0
    skipped_by_filter = 0
    per_split: Counter[str] = Counter()

    total_records = 0
    for ann_file in ann_files:
        records = json.loads(ann_file.resolve().read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError(f"ScreenSpot-Pro annotation must be a list: {ann_file}")
        total_records += len(records)
        for index, item in enumerate(records):
            if not isinstance(item, dict):
                skipped_bad += 1
                continue
            source_id = str(item.get("id") or f"{ann_file.stem}:{index}")
            if allow_ids and source_id not in allow_ids and str(index) not in allow_ids:
                skipped_by_filter += 1
                continue
            image_rel = str(item.get("img_filename") or "").strip()
            image_path = (image_dir / image_rel).resolve() if image_rel else Path("")
            image_size = get_image_size(image_path, item.get("img_size"))
            instruction = str(item.get(args.instruction_key) or item.get("instruction") or "").strip()
            bbox = item.get("bbox")
            if not image_rel or image_size is None or not instruction or not isinstance(bbox, list):
                if image_rel and image_size is None:
                    skipped_missing_image += 1
                    missing_images.append(str(image_path))
                else:
                    skipped_bad += 1
                continue
            try:
                bbox_norm = normalize_xyxy(bbox, image_size)
            except Exception:
                skipped_bad += 1
                continue
            task_type = str(item.get("ui_type") or ann_file.stem)
            samples.append(
                make_sample(
                    image_path=image_path,
                    instruction=instruction,
                    bbox_norm01=bbox_norm,
                    benchmark="screenspot-pro",
                    source_id=source_id,
                    task_type=task_type,
                    image_size=image_size,
                    extra_sample={
                        "application": item.get("application"),
                        "platform": item.get("platform"),
                        "ui_type": item.get("ui_type"),
                    },
                )
            )
            per_split[ann_file.stem] += 1
            if args.max_samples is not None and len(samples) >= max(0, args.max_samples):
                break
        if args.max_samples is not None and len(samples) >= max(0, args.max_samples):
            break

    if missing_images and args.strict_image_check:
        raise FileNotFoundError(f"Missing {len(missing_images)} ScreenSpot-Pro images, first={missing_images[0]}")
    return samples, {
        "benchmark": "screenspot-pro",
        "annotations_dir": str(annotations_dir.resolve()),
        "image_dir": str(image_dir.resolve()),
        "split": split,
        "annotation_files": [str(path.resolve()) for path in ann_files],
        "num_records": total_records,
        "num_samples": len(samples),
        "num_skipped_bad_record": skipped_bad,
        "num_skipped_missing_image": skipped_missing_image,
        "num_skipped_by_filter": skipped_by_filter,
        "per_split_counts": dict(sorted(per_split.items())),
    }


def expand_ui_task_sets(value: str | None) -> list[str]:
    names: list[str] = []
    for raw in _split_csv(value or "all"):
        expanded = UI_TASKSET_ALIASES.get(raw, raw)
        for item in _split_csv(expanded):
            if item not in UI_TASKSET_FILES:
                raise ValueError(f"Unsupported UI-Vision task set: {raw!r}")
            if item not in names:
                names.append(item)
    return names


def convert_ui_vision_record(
    item: dict[str, Any],
    *,
    root: Path,
    image_dir: Path,
    taskset: str,
    index: int,
    instruction_key: str,
) -> dict[str, Any] | None:
    image_rel = str(item.get("image_path") or item.get("img_filename") or "").strip()
    if not image_rel or "bbox" not in item:
        return None
    if Path(image_rel).parent == Path(".") and taskset.startswith("element_"):
        image_rel = f"element_grounding/{image_rel}"
    image_path = (image_dir / image_rel).resolve()
    image_size = get_image_size(image_path, item.get("image_size"))
    if image_size is None:
        return None
    instruction = str(
        item.get(instruction_key)
        or item.get("prompt_to_evaluate")
        or item.get("instruction")
        or item.get("prompt")
        or ""
    ).strip()
    if not instruction and taskset == "layout":
        explanation = str(item.get("explanation") or "").strip()
        instruction = explanation
    if not instruction:
        return None
    bbox_norm = normalize_xyxy(item["bbox"], image_size)
    source_id = f"{taskset}:{Path(image_rel).stem}:{index}"
    task_type = taskset
    element_type = str(item.get("element_type") or item.get("category") or ("layout_region" if taskset == "layout" else "element"))
    return make_sample(
        image_path=image_path,
        instruction=instruction,
        bbox_norm01=bbox_norm,
        benchmark="ui-vision",
        source_id=source_id,
        task_type=task_type,
        image_size=image_size,
        data_type=element_type,
        extra_sample={
            "platform": item.get("platform"),
            "element_type": element_type,
        },
    )


def convert_ui_vision(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = benchmark_root("ui-vision")
    image_dir = (args.image_dir and Path(args.image_dir).expanduser()) or (root / "images")
    if args.dataset_json:
        dataset_files = [Path(p).expanduser().resolve() for p in _split_csv(args.dataset_json)]
        tasksets = [f"custom_{idx}" for idx, _ in enumerate(dataset_files)]
    else:
        tasksets = expand_ui_task_sets(args.ui_task_sets)
        dataset_files = [(root / UI_TASKSET_FILES[name]).resolve() for name in tasksets]

    allow_ids = set(_split_csv(args.sample_ids))
    samples: list[dict[str, Any]] = []
    missing_or_bad = 0
    skipped_by_filter = 0
    total_records = 0
    per_taskset: Counter[str] = Counter()

    for taskset, dataset_file in zip(tasksets, dataset_files, strict=True):
        records = json.loads(dataset_file.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError(f"UI-Vision annotation must be a list: {dataset_file}")
        total_records += len(records)
        for index, item in enumerate(records):
            if not isinstance(item, dict):
                missing_or_bad += 1
                continue
            sample = convert_ui_vision_record(
                item,
                root=root,
                image_dir=image_dir,
                taskset=taskset,
                index=index,
                instruction_key=args.instruction_key,
            )
            if sample is None:
                missing_or_bad += 1
                continue
            source_id = str(sample["venus_id"])
            if allow_ids and source_id not in allow_ids and str(index) not in allow_ids:
                skipped_by_filter += 1
                continue
            samples.append(sample)
            per_taskset[taskset] += 1
            if args.max_samples is not None and len(samples) >= max(0, args.max_samples):
                break
        if args.max_samples is not None and len(samples) >= max(0, args.max_samples):
            break

    if missing_or_bad and args.strict_image_check:
        raise FileNotFoundError(f"UI-Vision has {missing_or_bad} missing/bad records")
    return samples, {
        "benchmark": "ui-vision",
        "image_dir": str(image_dir.resolve()),
        "dataset_files": [str(path) for path in dataset_files],
        "task_sets": tasksets,
        "num_records": total_records,
        "num_samples": len(samples),
        "num_missing_or_bad_records": missing_or_bad,
        "num_skipped_by_filter": skipped_by_filter,
        "per_taskset_counts": dict(sorted(per_taskset.items())),
    }


def resolve_mmbench_image_path(image_path: str, platform: str, image_dir: Path) -> Path:
    raw = Path(image_path)
    if raw.is_absolute():
        return raw

    candidates = [
        image_dir / platform / image_path,
        image_dir / "offline_images" / platform / image_path,
        image_dir / image_path,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[0].resolve()


def normalize_mmbench_xyxy_bbox(bbox: list[Any], image_size: Any) -> list[float]:
    values = [float(v) for v in bbox]
    if len(values) != 4:
        raise ValueError(f"bbox must have 4 numbers, got {values}")
    if max(abs(v) for v in values) <= 1.5:
        return values

    if not isinstance(image_size, list | tuple) or len(image_size) != 2:
        raise ValueError("pixel bbox requires image_size=[width,height]")
    width, height = float(image_size[0]), float(image_size[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid image_size={image_size!r}")
    return [values[0] / width, values[1] / height, values[2] / width, values[3] / height]


def convert_mmbench_gui(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = benchmark_root("mmbench-gui")
    dataset_json = (args.dataset_json and Path(args.dataset_json).expanduser()) or (root / "L2_annotations.json")
    image_dir = (args.image_dir and Path(args.image_dir).expanduser()) or (root / "images")
    records = json.loads(dataset_json.resolve().read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"MMBench-GUI annotation must be a JSON list: {dataset_json}")

    allow_ids = set(_split_csv(args.sample_ids))
    samples: list[dict[str, Any]] = []
    missing_images: list[str] = []
    skipped_bad = 0
    skipped_by_filter = 0

    for offset, item in enumerate(records):
        if not isinstance(item, dict):
            skipped_bad += 1
            continue
        mmbench_index = item.get("index", offset)
        sample_key = f"l2:{mmbench_index}"
        if allow_ids and sample_key not in allow_ids and str(mmbench_index) not in allow_ids:
            skipped_by_filter += 1
            continue

        image_rel = str(item.get("image_path") or "").strip()
        instruction = str(item.get(args.instruction_key) or item.get("instruction") or "").strip()
        bbox = item.get("bbox")
        if not image_rel or not instruction or not isinstance(bbox, list) or len(bbox) != 4:
            skipped_bad += 1
            continue

        platform = str(item.get("platform") or "unknown_platform").strip()
        abs_image = resolve_mmbench_image_path(image_rel, platform, image_dir)
        if not abs_image.is_file():
            missing_images.append(str(abs_image))

        samples.append(
            {
                "img_filename": str(abs_image),
                "data_source": "mmbench-gui:L2",
                "mmbench_gui_index": mmbench_index,
                "mmbench_gui_image_path": image_rel,
                "mmbench_gui_platform": platform,
                "image_size": item.get("image_size"),
                "elements": [
                    {
                        "instruction": instruction,
                        "bbox": normalize_mmbench_xyxy_bbox(bbox, item.get("image_size")),
                        "data_type": item.get("data_type"),
                        "task_type": item.get("grounding_type"),
                        "app_name": item.get("app_name"),
                        "grounding_type": item.get("grounding_type"),
                        "mmbench_gui_index": mmbench_index,
                    }
                ],
            }
        )
        if args.max_samples is not None and len(samples) >= max(0, args.max_samples):
            break

    if missing_images and args.strict_image_check:
        preview = "\n".join(missing_images[:10])
        raise FileNotFoundError(f"Missing {len(missing_images)} MMBench-GUI images, first entries:\n{preview}")

    return samples, {
        "benchmark": "mmbench-gui",
        "dataset_json": str(dataset_json.resolve()),
        "image_dir": str(image_dir.resolve()),
        "num_records": len(records),
        "num_samples": len(samples),
        "num_skipped_bad_record": skipped_bad,
        "num_skipped_by_filter": skipped_by_filter,
        "num_missing_images": len(missing_images),
        "first_missing_image": missing_images[0] if missing_images else None,
    }


def resolve_venus_instruction_files(value: str | None, instruction_dir: Path) -> list[Path]:
    if not value:
        return [(instruction_dir / name).resolve() for name in VENUS_DEFAULT_CATEGORY_FILES]

    out: list[Path] = []
    for raw in _split_csv(value):
        path = Path(raw).expanduser()
        if path.is_dir():
            out.extend((path / name).resolve() for name in VENUS_DEFAULT_CATEGORY_FILES)
            continue
        if path.is_absolute() or path.exists():
            out.append(path.resolve())
            continue
        if path.suffix == "" and "/" not in raw and "\\" not in raw:
            for candidate in (
                instruction_dir / f"{raw}.json",
                instruction_dir / f"{raw}_grounding.json",
            ):
                if candidate.is_file():
                    out.append(candidate.resolve())
                    break
            else:
                out.append((instruction_dir / f"{raw}.json").resolve())
            continue
        candidate = instruction_dir / raw
        out.append(candidate.resolve() if candidate.is_file() else path.resolve())
    return out


def load_venus_meta_index(meta_json: Path) -> dict[str, dict[str, str]]:
    if not meta_json.is_file():
        return {}
    records = json.loads(meta_json.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        return {}
    index: dict[str, dict[str, str]] = {}
    for item in records:
        if not isinstance(item, dict):
            continue
        image = item.get("img_filename")
        if not image:
            continue
        index[str(image)] = {
            "language": str(item.get("language") or "unknown_lang"),
            "platform": str(item.get("platform") or "unknown_platform"),
            "application": str(item.get("application") or "unknown_app"),
        }
    return index


def resolve_venus_image_path(image_path: str, image_dir: Path) -> Path:
    raw = Path(image_path)
    if raw.is_absolute():
        return raw
    candidates = [
        image_dir / image_path,
        image_dir / "images" / image_path,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[0].resolve()


def normalize_venus_xyxy_bbox(bbox: list[Any], image_size: tuple[int, int]) -> list[float]:
    values = [float(v) for v in bbox]
    if len(values) != 4:
        raise ValueError(f"bbox must have 4 numbers, got {values}")
    if max(abs(v) for v in values) <= 1.5:
        return values
    width, height = float(image_size[0]), float(image_size[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid image_size={image_size!r}")
    x1, y1, x2, y2 = values
    return [x1 / width, y1 / height, x2 / width, y2 / height]


def convert_venus_bench_gd(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = benchmark_root("venus-bench-gd")
    instruction_dir = (args.instruction_dir and Path(args.instruction_dir).expanduser()) or (root / "instruction")
    image_dir = (args.image_dir and Path(args.image_dir).expanduser()) or (root / "images")
    meta_json = (args.meta_json and Path(args.meta_json).expanduser()) or (root / "meta.json")
    instruction_files = resolve_venus_instruction_files(args.instruction_json, instruction_dir)
    meta_index = load_venus_meta_index(meta_json.resolve())

    allow_ids = set(_split_csv(args.sample_ids))
    samples: list[dict[str, Any]] = []
    missing_images: list[str] = []
    skipped_no_bbox = 0
    skipped_bad_record = 0
    skipped_missing_image = 0
    skipped_by_filter = 0
    per_category: Counter[str] = Counter()
    total_records = 0

    for instruction_file in instruction_files:
        if not instruction_file.is_file():
            raise FileNotFoundError(f"Venus-Bench-GD instruction JSON not found: {instruction_file}")
        records = json.loads(instruction_file.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError(f"Venus-Bench-GD annotation must be a JSON list: {instruction_file}")
        total_records += len(records)
        dataset_name = instruction_file.stem
        for offset, item in enumerate(records):
            if not isinstance(item, dict):
                skipped_bad_record += 1
                continue

            category = str(item.get("category") or dataset_name).strip()
            venus_id = f"venus:{category}:{offset}"
            short_id = f"{category}:{offset}"
            if allow_ids and venus_id not in allow_ids and short_id not in allow_ids and str(offset) not in allow_ids:
                skipped_by_filter += 1
                continue

            image_rel = str(item.get("img_filename") or "").strip()
            instruction = item.get(args.instruction_key) or item.get("instruction") or item.get("instruction_zh")
            instruction = str(instruction).strip() if instruction is not None else ""
            bbox = item.get("bbox")
            if not image_rel or not instruction:
                skipped_bad_record += 1
                continue
            if not isinstance(bbox, list) or len(bbox) != 4:
                if args.include_refusal:
                    bbox = None
                else:
                    skipped_no_bbox += 1
                    continue

            abs_image = resolve_venus_image_path(image_rel, image_dir)
            image_size = get_image_size(abs_image)
            if image_size is None:
                missing_images.append(str(abs_image))
                skipped_missing_image += 1
                continue
            gt_bbox = None if bbox is None else normalize_venus_xyxy_bbox(bbox, image_size)

            meta = meta_index.get(image_rel, {})
            platform = meta.get("platform") or (image_rel.split("/", 1)[0] if "/" in image_rel else "unknown_platform")
            application = meta.get("application") or (image_rel.split("/")[1] if image_rel.count("/") >= 2 else "unknown_app")
            language = meta.get("language") or "unknown_lang"
            data_type = str(item.get("data_type") or "bbox").strip()
            label = str(item.get("label") or "").strip()

            samples.append(
                {
                    "img_filename": str(abs_image),
                    "data_source": "venus-bench-gd",
                    "venus_id": venus_id,
                    "venus_image_path": image_rel,
                    "venus_category": category,
                    "venus_label": label,
                    "venus_platform": platform,
                    "venus_application": application,
                    "venus_language": language,
                    "venus_data_type": data_type,
                    "image_size": [image_size[0], image_size[1]],
                    "elements": [
                        {
                            "instruction": instruction,
                            "bbox": gt_bbox,
                            "data_type": data_type,
                            "task_type": category,
                            "venus_id": venus_id,
                            "venus_category": category,
                            "venus_label": label,
                            "venus_platform": platform,
                            "venus_application": application,
                            "venus_language": language,
                            "venus_data_type": data_type,
                        }
                    ],
                }
            )
            per_category[category] += 1
            if args.max_samples is not None and len(samples) >= max(0, args.max_samples):
                break
        if args.max_samples is not None and len(samples) >= max(0, args.max_samples):
            break

    if missing_images and args.strict_image_check:
        preview = "\n".join(missing_images[:10])
        raise FileNotFoundError(f"Missing {len(missing_images)} Venus-Bench-GD images, first entries:\n{preview}")

    return samples, {
        "benchmark": "venus-bench-gd",
        "dataset_json": [str(path) for path in instruction_files],
        "image_dir": str(image_dir.resolve()),
        "meta_json": str(meta_json.resolve()),
        "instruction_key": args.instruction_key,
        "num_records": total_records,
        "num_samples": len(samples),
        "num_skipped_no_bbox": skipped_no_bbox,
        "num_skipped_bad_record": skipped_bad_record,
        "num_skipped_missing_image": skipped_missing_image,
        "num_skipped_by_filter": skipped_by_filter,
        "num_missing_images": len(missing_images),
        "first_missing_image": missing_images[0] if missing_images else None,
        "per_category_counts": dict(sorted(per_category.items())),
        "include_refusal": bool(args.include_refusal),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark", choices=BENCHMARK_NAMES, required=True)
    p.add_argument("--dataset-json", default=None)
    p.add_argument("--image-dir", default=None)
    p.add_argument("--annotations-dir", default=None)
    p.add_argument("--instruction-json", default=None)
    p.add_argument("--instruction-dir", default=None)
    p.add_argument("--meta-json", default=None)
    p.add_argument("--split", default="overall")
    p.add_argument("--ui-task-sets", default="all")
    p.add_argument("--instruction-key", default="instruction")
    p.add_argument("--converted-json", type=Path, default=None)
    p.add_argument("--prepare-only", action="store_true", help="Compatibility no-op; this converter only prepares data.")
    p.add_argument("--strict-image-check", action="store_true")
    p.add_argument("--include-refusal", action="store_true")
    p.add_argument("--sample-ids", default=None)
    p.add_argument("--max-samples", type=int, default=None)
    return p


def convert(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if args.benchmark == "osworld-g":
        return convert_osworld_g(args)
    if args.benchmark == "screenspot-pro":
        return convert_screenspot_pro(args)
    if args.benchmark == "ui-vision":
        return convert_ui_vision(args)
    if args.benchmark == "mmbench-gui":
        return convert_mmbench_gui(args)
    if args.benchmark == "venus-bench-gd":
        return convert_venus_bench_gd(args)
    raise ValueError(f"Unsupported benchmark: {args.benchmark}")


def main() -> None:
    args = build_arg_parser().parse_args()
    samples, summary = convert(args)
    converted_json = (args.converted_json or default_converted_json(args.benchmark)).expanduser().resolve()
    converted_json.parent.mkdir(parents=True, exist_ok=True)
    converted_json.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[prepare] benchmark={args.benchmark} converted={converted_json} samples={len(samples)}",
        file=sys.stderr,
        flush=True,
    )
    print(json.dumps({**summary, "converted_json": str(converted_json)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
