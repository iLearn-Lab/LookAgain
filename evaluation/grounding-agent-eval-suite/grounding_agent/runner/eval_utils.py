from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from grounding_agent.env import first_env as _first_env

GROUNDCUA_ANNOTATION_FIELDS = ("if_bbox_right", "if_inst_match", "if_fine_grained")
GROUNDCUA_SUBSET_SPECS: tuple[tuple[str, str, Callable[[dict[str, Any]], bool]], ...] = (
    ("fine_grained_yes", "if_fine_grained=yes", lambda row: row.get("if_fine_grained") == "yes"),
    (
        "exclude_ambiguous",
        "exclude if_bbox_right=ambiguous",
        lambda row: row.get("if_bbox_right") != "ambiguous",
    ),
    (
        "ambiguous_only",
        "if_bbox_right=ambiguous only",
        lambda row: row.get("if_bbox_right") == "ambiguous",
    ),
)


def _shared_run_stamp() -> str:
    for env_name in ("RUN_STAMP", "MERLIN_JOB_ID", "ARNOLD_TRIAL_ID", "SLURM_JOB_ID", "JOB_ID"):
        value = _first_env(env_name)
        if value:
            return value
    now = datetime.now()
    return f"{now.year}-{now.month:02d}-{now.day:02d}_{now.hour:02d}-{now.minute:02d}"


def make_default_run_dir(root: Path, args: argparse.Namespace, *, nnodes: int = 1) -> Path:
    run_name = args.run_name
    run_data = str(args.input_json).replace(".json", "")

    output_dir = _first_env("RUN_OUTPUT_DIR")
    if output_dir is not None:
        path = Path(output_dir)
        if path.is_absolute():
            return path
        return root / path

    now = datetime.now()
    yyyymm = f"{now.year}{now.month:02d}"
    if nnodes > 1:
        stamp = _shared_run_stamp()
    else:
        stamp = (
            f"{now.year}-{now.month:02d}-{now.day:02d}_"
            f"{now.hour:02d}-{now.minute:02d}-{now.second:02d}"
        )
    return root / "results" / run_name / run_data / yyyymm / stamp


def load_json(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("JSON root must be an array of samples")
    return data


def samples_have_groundcua_annotations(samples: list[dict[str, Any]]) -> bool:
    required = set(GROUNDCUA_ANNOTATION_FIELDS)
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        for el in sample.get("elements") or []:
            if isinstance(el, dict) and required.issubset(el.keys()):
                return True
    return False


def iter_grounding_tasks(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for si, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError(f"sample {si} must be an object")
        img = sample.get("img_filename")
        if not isinstance(img, str) or not img:
            raise ValueError(f"sample {si} missing string img_filename")
        elements = sample.get("elements")
        if not isinstance(elements, list):
            raise ValueError(f"sample {si} missing elements array")
        data_source = sample.get("data_source")
        for ei, el in enumerate(elements):
            if not isinstance(el, dict):
                raise ValueError(f"sample {si} element {ei} must be an object")
            instruction = el.get("instruction")
            if not isinstance(instruction, str) or not instruction.strip():
                continue
            task_id = str(el.get("custom_id") or f"sample_{si}_element_{ei}")
            rec: dict[str, Any] = {
                "sample_index": si,
                "element_index": ei,
                "task_id": task_id,
                "img_filename": img,
                "instruction": instruction,
                "gt_bbox": el.get("bbox"),
                "data_type": el.get("data_type"),
                "data_source": data_source,
            }
            if "task_type" in el:
                rec["task_type"] = el["task_type"]
            if "custom_id" in el:
                rec["custom_id"] = el["custom_id"]
            for field in GROUNDCUA_ANNOTATION_FIELDS:
                if field in el:
                    rec[field] = el[field]
            tasks.append(rec)
    return tasks


def expand_tasks_for_rollout(
    tasks: list[dict[str, Any]],
    rollout_n: int,
) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for task in tasks:
        for rollout_index in range(rollout_n):
            expanded.append({**task, "rollout_index": rollout_index})
    return expanded


def _check_center_in_gt(
    pred_bboxes: list[dict[str, Any]] | None,
    gt_bbox: Any,
) -> tuple[bool | None, list[float] | None, list[float] | None]:
    if not isinstance(gt_bbox, list) or len(gt_bbox) != 4:
        return None, None, None
    try:
        gx1, gy1, gx2, gy2 = [float(v) for v in gt_bbox]
    except (TypeError, ValueError):
        return None, None, None
    gx1, gy1, gx2, gy2 = min(gx1, gx2), min(gy1, gy2), max(gx1, gx2), max(gy1, gy2)
    gt_xyxy: list[float] = [gx1, gy1, gx2, gy2]

    if not pred_bboxes:
        return None, None, gt_xyxy

    bbox_2d = pred_bboxes[0].get("bbox_2d")
    if not isinstance(bbox_2d, list) or len(bbox_2d) != 4:
        return None, None, gt_xyxy

    try:
        cx = (float(bbox_2d[0]) + float(bbox_2d[2])) / 2.0 / 1000.0
        cy = (float(bbox_2d[1]) + float(bbox_2d[3])) / 2.0 / 1000.0
    except (TypeError, ValueError):
        return None, None, gt_xyxy

    inside = gx1 <= cx <= gx2 and gy1 <= cy <= gy2
    return inside, [cx, cy], gt_xyxy


def _make_json_safe(row: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for k, v in row.items():
        if k == "turns":
            safe_turns = []
            for turn in v or []:
                safe_turn: dict[str, Any] = {}
                for tk, tv in turn.items():
                    if tk == "model_content":
                        safe_turn[tk] = str(tv) if tv is not None else None
                    else:
                        safe_turn[tk] = tv
                safe_turns.append(safe_turn)
            safe[k] = safe_turns
        else:
            safe[k] = v
    return safe


def _result_path(output_dir: Path, task_id: str, rollout_id: str) -> Path:
    return output_dir / task_id / rollout_id / "result.json"


def _has_turn_error(result: dict[str, Any]) -> bool:
    return any(
        turn.get("error") is not None
        for turn in (result.get("turns") or [])
    )


def _load_cached_result(output_dir: Path, task_id: str, rollout_id: str) -> dict[str, Any] | None:
    path = _result_path(output_dir, task_id, rollout_id)
    if not path.exists():
        return None
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if _has_turn_error(result):
        return None
    return result


def _group_by_task_id(
    results: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in results:
        groups.setdefault(row.get("task_id", ""), []).append(row)
    return groups


def summarize_rollout_pass_metrics(
    results: list[dict[str, Any]],
    rollout_n: int,
) -> dict[str, Any]:
    groups = _group_by_task_id(results)
    n_eval = len(groups)
    pass1_hit = pass1_failed = passk_hit = passk_failed = 0

    for rows in groups.values():
        hits = [r.get("center_in_gt") for r in rows]
        if any(h is True for h in hits):
            passk_hit += 1
        elif all(h is None for h in hits):
            passk_failed += 1

        first_row = next((r for r in rows if r.get("rollout_index", 0) == 0), None)
        if first_row is None:
            pass1_failed += 1
        elif first_row.get("center_in_gt") is True:
            pass1_hit += 1
        elif first_row.get("center_in_gt") is None:
            pass1_failed += 1

    pass1_acc = pass1_hit / n_eval if n_eval else 0.0
    passk_acc = passk_hit / n_eval if n_eval else 0.0
    return {
        "pass_at_1_hits": pass1_hit,
        "pass_at_1_evaluable": n_eval,
        "pass_at_1_failed_no_pred": pass1_failed,
        "pass_at_1_accuracy": pass1_acc,
        "pass_at_k_hits": passk_hit,
        "pass_at_k_evaluable": n_eval,
        "pass_at_k_failed_no_pred": passk_failed,
        "pass_at_k_accuracy": passk_acc,
        "pass_at_k": rollout_n,
    }


def summarize_results(
    results: list[dict[str, Any]],
    *,
    unique_tasks: int | None = None,
    rollout_n: int = 1,
) -> dict[str, Any]:
    n_eval = len(results)
    n_hit = sum(1 for r in results if r.get("center_in_gt") is True)
    n_failed = sum(1 for r in results if r.get("center_in_gt") is None)
    acc = n_hit / n_eval if n_eval else 0.0
    turn_dist: Counter[int] = Counter(r.get("total_turns", 0) for r in results)
    summary: dict[str, Any] = {
        "center_in_gt_hits": n_hit,
        "center_in_gt_evaluable": n_eval,
        "center_in_gt_failed_no_pred": n_failed,
        "center_in_gt_accuracy": acc,
        "total_tasks": n_eval,
        "unique_tasks": unique_tasks if unique_tasks is not None else n_eval,
        "rollout_n": rollout_n,
        "turn_distribution": dict(sorted(turn_dist.items())),
    }
    summary.update(summarize_rollout_pass_metrics(results, rollout_n))
    return summary


def attach_groundcua_subset_metrics(
    summary: dict[str, Any],
    results: list[dict[str, Any]],
    *,
    rollout_n: int,
    enabled: bool,
) -> None:
    if not enabled:
        return
    subsets: dict[str, Any] = {}
    for key, label, predicate in GROUNDCUA_SUBSET_SPECS:
        filtered = [r for r in results if predicate(r)]
        subsets[key] = {"label": label, **summarize_rollout_pass_metrics(filtered, rollout_n)}
    summary["groundcua_subsets"] = subsets


def format_eval_summary(
    summary: dict[str, Any],
    *,
    workers: int,
    total_tasks: int,
) -> str:
    rollout_n = summary.get("rollout_n", 1)
    unique_tasks = summary.get("unique_tasks", total_tasks)
    lines = [
        (
            f"[eval] pass@1: {summary['pass_at_1_hits']}/"
            f"{summary['pass_at_1_evaluable']} "
            f"({100.0 * summary['pass_at_1_accuracy']:.2f}%) "
            f"| failed(no_pred)={summary['pass_at_1_failed_no_pred']} "
            f"| threads={workers} | total_tasks={total_tasks} "
            f"| unique_tasks={unique_tasks} | rollout_n={rollout_n}"
        ),
    ]
    if rollout_n > 1:
        lines.append(
            f"[eval] pass@{rollout_n}: {summary['pass_at_k_hits']}/"
            f"{summary['pass_at_k_evaluable']} "
            f"({100.0 * summary['pass_at_k_accuracy']:.2f}%) "
            f"| failed(no_pred)={summary['pass_at_k_failed_no_pred']}"
        )
    lines.append(
        f"[eval] center_in_gt (all): {summary['center_in_gt_hits']}/"
        f"{summary['center_in_gt_evaluable']} "
        f"({100.0 * summary['center_in_gt_accuracy']:.2f}%) "
        f"| failed(no_pred)={summary['center_in_gt_failed_no_pred']}"
    )
    subsets = summary.get("groundcua_subsets")
    if subsets:
        lines.append("[eval] GroundCUA subset pass metrics:")
        for key, _, _ in GROUNDCUA_SUBSET_SPECS:
            m = subsets[key]
            lines.append(
                f"  [{m['label']}] pass@1: {m['pass_at_1_hits']}/{m['pass_at_1_evaluable']} "
                f"({100.0 * m['pass_at_1_accuracy']:.2f}%)"
            )
            if rollout_n > 1:
                lines.append(
                    f"  [{m['label']}] pass@{rollout_n}: {m['pass_at_k_hits']}/{m['pass_at_k_evaluable']} "
                    f"({100.0 * m['pass_at_k_accuracy']:.2f}%)"
                )
    dist = summary.get("turn_distribution", {})
    dist_str = "  ".join(f"turn{k}={v}" for k, v in sorted(dist.items()))
    if dist_str:
        lines.append(f"[turns] distribution: {dist_str}")
    return "\n".join(lines) + "\n"


def print_eval_summary(
    summary: dict[str, Any], *, workers: int, total_tasks: int
) -> None:
    sys.stderr.write(format_eval_summary(summary, workers=workers, total_tasks=total_tasks))
    sys.stderr.flush()


def write_eval_summary_file(
    path: Path,
    summary: dict[str, Any],
    *,
    workers: int,
    total_tasks: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        format_eval_summary(summary, workers=workers, total_tasks=total_tasks),
        encoding="utf-8",
    )
