"""Runner for SFTToolCallGroundingAgent.

For each (image, instruction) pair in the input JSON the agent runs a locate/
confirm tool-calling loop against a vLLM-served OpenAI-compatible endpoint.

Results are written to:
    {output}/{task_id}/{rollout_id}/meta.json   – per-task trajectory + crops
    {output}/{task_id}/{rollout_id}/result.json – per-rollout grading
    {output}/results.json                       – flat per-rollout result list
    {output}/eval_summary.txt                   – pass@1 / pass@k metrics
    {output}/run_args.json                      – run configuration + eval dict

Usage::

    python grounding_agent/runner/run_sft_tool_call.py \\
        data/OSWorld-G_refined_grounding_no_refusal.json \\
        -o results/sft_osworld_g \\
        --workers 16 \\
        --api-base http://localhost:34575/v1 \\
        --model Qwen/Qwen3.5-27B \\
        --max-turns 5

Most of the data-loading / eval / summary helpers are imported from
``eval_utils`` to avoid duplication; this file only owns the SFT-specific
per-task entry point and CLI surface.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Allow `python grounding_agent/runner/run_sft_tool_call.py` from the eval-suite root.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from grounding_agent.agents.sft_tool_call_agent import SFTToolCallGroundingAgent
from grounding_agent.env import first_env as _first_env
from grounding_agent.env import load_project_env
from grounding_agent.runner.eval_utils import (
    _check_center_in_gt,
    _load_cached_result,
    _make_json_safe,
    _result_path,
    attach_groundcua_subset_metrics,
    expand_tasks_for_rollout,
    format_eval_summary,
    iter_grounding_tasks,
    load_json,
    make_default_run_dir,
    print_eval_summary,
    samples_have_groundcua_annotations,
    summarize_results,
    write_eval_summary_file,
)

load_project_env()

_verbose_print_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Per-task processing
# ---------------------------------------------------------------------------


def _turn_args_to_pred_bboxes(tool_args: Any) -> list[dict[str, Any]] | None:
    if not isinstance(tool_args, dict):
        return None

    bbox_2d = tool_args.get("bbox_2d") or tool_args.get("bbox")
    if isinstance(bbox_2d, list) and len(bbox_2d) == 4:
        try:
            return [{"bbox_2d": [float(v) for v in bbox_2d]}]
        except (TypeError, ValueError):
            return None

    if all(key in tool_args for key in ("x1", "y1", "x2", "y2")):
        try:
            return [{
                "bbox_2d": [
                    float(tool_args["x1"]),
                    float(tool_args["y1"]),
                    float(tool_args["x2"]),
                    float(tool_args["y2"]),
                ]
            }]
        except (TypeError, ValueError):
            return None

    return None


def _has_confirm_turn(turns: list[dict[str, Any]]) -> bool:
    return any(turn.get("tool_name") == "confirm" for turn in turns)


def _last_locate_pred_bboxes(turns: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    for turn in reversed(turns):
        if turn.get("tool_name") != "locate":
            continue
        pred_bboxes = _turn_args_to_pred_bboxes(turn.get("tool_args"))
        if pred_bboxes:
            return pred_bboxes
    return None


def _score_result(
    result: dict[str, Any],
    gt_bbox: Any,
) -> tuple[bool | None, list[float] | None, list[float] | None, str | None, list[dict[str, Any]] | None]:
    pred_bboxes = result.get("pred_bboxes")
    center_in_gt, pred_center_xy_norm01, gt_xyxy_norm01 = _check_center_in_gt(
        pred_bboxes,
        gt_bbox,
    )
    scoring_source = "pred_bboxes" if pred_bboxes else None
    scoring_pred_bboxes = pred_bboxes

    turns = result.get("turns") or []
    if (
        center_in_gt is None
        and not pred_bboxes
        and isinstance(turns, list)
        and not _has_confirm_turn(turns)
    ):
        last_locate_pred_bboxes = _last_locate_pred_bboxes(turns)
        if last_locate_pred_bboxes:
            fallback_center, fallback_center_xy, fallback_gt_xyxy = _check_center_in_gt(
                last_locate_pred_bboxes,
                gt_bbox,
            )
            if fallback_center is not None:
                center_in_gt = fallback_center
                pred_center_xy_norm01 = fallback_center_xy
                gt_xyxy_norm01 = fallback_gt_xyxy
                scoring_source = "last_locate"
                scoring_pred_bboxes = last_locate_pred_bboxes

    return (
        center_in_gt,
        pred_center_xy_norm01,
        gt_xyxy_norm01,
        scoring_source,
        scoring_pred_bboxes,
    )


def process_task(
    task: dict[str, Any],
    agent_kwargs: dict[str, Any],
    *,
    output_dir: Path,
    verbose: bool = False,
) -> dict[str, Any]:
    rollout_index = task.get("rollout_index", 0)
    task_id = task["task_id"]
    rollout_id = str(rollout_index)

    # Fresh agent per task -- independent vLLM context.
    agent = SFTToolCallGroundingAgent(**agent_kwargs, temp_dir=output_dir)

    result = agent.run(
        instruction=task["instruction"],
        image_path=task["img_filename"],
        task_id=task_id,
        rollout_id=rollout_id,
    )

    (
        center_in_gt,
        pred_center_xy_norm01,
        gt_xyxy_norm01,
        scoring_source,
        scoring_pred_bboxes,
    ) = _score_result(
        result,
        task.get("gt_bbox"),
    )

    if verbose:
        line = (
            f"[task] sample={task.get('sample_index')} element={task.get('element_index')} "
            f"rollout={rollout_index} status={result['status']} "
            f"center_in_gt={center_in_gt} turns={result['total_turns']} "
            f"scoring_source={scoring_source}"
        )
        if result["status"] == "failed":
            line += f" reason={result.get('feedback')!r}"
        with _verbose_print_lock:
            print(line, file=sys.stderr, flush=True)

    row = {
        **task,
        "status": result["status"],
        "feedback": result.get("feedback"),
        "locate_count": result.get("locate_count", 0),
        "total_turns": result["total_turns"],
        "pred_bboxes": result.get("pred_bboxes"),
        "turns": result.get("turns", []),
        "center_in_gt": center_in_gt,
        "pred_center_xy_norm01": pred_center_xy_norm01,
        "gt_xyxy_norm01": gt_xyxy_norm01,
        "scoring_source": scoring_source,
        "scoring_pred_bboxes": scoring_pred_bboxes,
        "last_locate_fallback_used": scoring_source == "last_locate",
        "task_output_dir": str(output_dir / task_id / rollout_id),
    }

    safe_row = _make_json_safe(row)
    rpath = _result_path(output_dir, task_id, rollout_id)
    rpath.parent.mkdir(parents=True, exist_ok=True)
    rpath.write_text(json.dumps(safe_row, ensure_ascii=False, indent=2), encoding="utf-8")
    return row


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run SFTToolCallGroundingAgent on a grounding JSON dataset. "
            "Multi-threaded, multi-rollout, pass@k aware, resumable."
        ),
    )
    p.add_argument(
        "input_json",
        type=Path,
        help="JSON array – each element has img_filename and elements[].instruction",
    )
    p.add_argument(
        "-o", "--output", type=Path, default=None,
        help=(
            "Root output dir (default: results/{run_name}/{dataset}/YYYYMM/{stamp} "
            f"under {_ROOT}; override with RUN_OUTPUT_DIR env). "
            "Per-task artifacts go to {output}/{task_id}/{rollout_id}/."
        ),
    )
    p.add_argument(
        "--run-name", type=str, default="SFT_Model_ToolCall",
        choices=["SFT_Model_ToolCall", "RL_Model_ToolCall", "Base_ToolCall"],
        help="Run label used in the default output path (default: SFT_Model).",
    )
    # Concurrency / resume / slicing
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--rollout-n", type=int, default=1)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--breakpoint-start", type=int, default=0)
    p.add_argument("--breakpoint-end", type=int, default=None)
    # vLLM endpoint
    p.add_argument(
        "--api-base", type=str, default="http://localhost:34575/v1",
        help="OpenAI-compatible vLLM endpoint",
    )
    p.add_argument(
        "--model", type=str, default=None,
        help="Model name (default: VLM_MODEL_NAME env, or Qwen/Qwen3.5-27B)",
    )
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--max-turns", type=int, default=5)
    p.add_argument(
        "--tool-choice", choices=["auto", "required", "none"], default="auto",
        help=(
            "tool_choice sent to vLLM (default: auto). 'auto' lets the "
            "qwen3_coder XML tool parser handle tool calls; 'required' forces "
            "guided JSON decoding which can conflict with the XML-trained model."
        ),
    )
    p.add_argument("--crop-size", type=int, default=512)
    p.add_argument("--timeout", type=float, default=60.0,
                   help="HTTP timeout per request in seconds (0 = no cap)")
    p.add_argument(
        "--disable-thinking", action="store_true",
        help="Pass enable_thinking=False (default: thinking on, matches SFT)",
    )
    # Output
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--print-results", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    if args.rollout_n < 1:
        raise SystemExit("--rollout-n must be >= 1")
    if args.breakpoint_start < 0:
        raise SystemExit("--breakpoint-start must be >= 0")
    if args.breakpoint_end is not None and args.breakpoint_end <= args.breakpoint_start:
        raise SystemExit("--breakpoint-end must be > --breakpoint-start when set")
    if args.rollout_n > 1 and args.temperature <= 0.0:
        print("[warn] --rollout-n > 1 with temperature <= 0; forcing temperature=1.0",
              file=sys.stderr)
        args.temperature = 1.0

    if args.output is None:
        output_dir = make_default_run_dir(_ROOT, args).resolve()
    else:
        output_dir = args.output.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load and slice ────────────────────────────────────────────────────────
    samples = load_json(args.input_json)
    has_groundcua = samples_have_groundcua_annotations(samples)
    tasks = iter_grounding_tasks(samples)

    n_flattened = len(tasks)
    bp_start = max(0, args.breakpoint_start)
    bp_end = args.breakpoint_end
    tasks = tasks[bp_start:bp_end] if bp_end is not None else tasks[bp_start:]
    n_after_slice = len(tasks)
    if args.limit is not None:
        tasks = tasks[: max(0, args.limit)]

    unique_tasks = len(tasks)
    run_tasks = expand_tasks_for_rollout(tasks, args.rollout_n)
    total_tasks = len(run_tasks)

    bp_end_str = str(bp_end) if bp_end is not None else "end"
    print(
        f"[run] flattened={n_flattened} slice=[{bp_start}:{bp_end_str}) "
        f"after_slice={n_after_slice} unique_tasks={unique_tasks} "
        f"rollout_n={args.rollout_n} total_tasks={total_tasks} output={output_dir}",
        file=sys.stderr,
    )

    # ── Agent kwargs ──────────────────────────────────────────────────────────
    model_name = args.model or _first_env("VLM_MODEL_NAME") or "Qwen/Qwen3.5-27B"
    timeout_kw = None if args.timeout == 0 else args.timeout
    agent_kwargs: dict[str, Any] = {
        "api_base": args.api_base,
        "model_name": model_name,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "max_turns": args.max_turns,
        "crop_size": args.crop_size,
        "enable_thinking": not args.disable_thinking,
        "tool_choice": args.tool_choice,
        "timeout": timeout_kw,
        "verbose": args.verbose,
    }

    # ── Resume ────────────────────────────────────────────────────────────────
    cached_results: list[dict[str, Any]] = []
    pending_tasks: list[dict[str, Any]] = []
    if args.resume:
        for task in run_tasks:
            cached = _load_cached_result(
                output_dir, task["task_id"], str(task.get("rollout_index", 0))
            )
            if cached is not None:
                cached_results.append(cached)
            else:
                pending_tasks.append(task)
        print(
            f"[resume] cached={len(cached_results)} pending={len(pending_tasks)} "
            f"(total={total_tasks})",
            file=sys.stderr,
        )
    else:
        pending_tasks = run_tasks

    # ── Run (threaded) ────────────────────────────────────────────────────────
    workers = max(1, args.workers)
    run_one = partial(
        process_task,
        agent_kwargs=agent_kwargs,
        output_dir=output_dir,
        verbose=args.verbose,
    )
    results: list[dict[str, Any]] = list(cached_results)
    n_correct = sum(1 for r in cached_results if r.get("center_in_gt") is True)
    n_error = sum(1 for r in cached_results if r.get("center_in_gt") is None)

    with tqdm(total=total_tasks, desc="SFT-Grounding", unit="task", dynamic_ncols=True) as pbar:
        if cached_results:
            n_done = len(cached_results)
            acc_str = f"{100.0 * n_correct / n_done:.1f}%" if n_done else "n/a"
            pbar.set_postfix(acc=acc_str, ok=n_correct, err=n_error, refresh=False)
            pbar.update(n_done)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(run_one, task): task for task in pending_tasks}
            for future in as_completed(futures):
                row = future.result()
                results.append(row)
                cig = row.get("center_in_gt")
                if cig is True:
                    n_correct += 1
                elif cig is None:
                    n_error += 1
                n_done = len(results)
                acc_str = f"{100.0 * n_correct / n_done:.1f}%" if n_done else "n/a"
                pbar.set_postfix(acc=acc_str, ok=n_correct, err=n_error, refresh=False)
                pbar.update(1)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    summary = summarize_results(results, unique_tasks=unique_tasks, rollout_n=args.rollout_n)
    attach_groundcua_subset_metrics(
        summary, results, rollout_n=args.rollout_n, enabled=has_groundcua
    )
    print_eval_summary(summary, workers=workers, total_tasks=total_tasks)

    # ── Persist ───────────────────────────────────────────────────────────────
    json_results = [_make_json_safe(r) for r in results]
    results_path = output_dir / "results.json"
    results_path.write_text(
        json.dumps(json_results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    eval_summary_path = output_dir / "eval_summary.txt"
    write_eval_summary_file(
        eval_summary_path, summary, workers=workers, total_tasks=total_tasks
    )

    run_args_payload: dict[str, Any] = {
        "input_json": str(args.input_json),
        "output": str(output_dir),
        "run_name": args.run_name,
        "workers": workers,
        "rollout_n": args.rollout_n,
        "api_base": args.api_base,
        "model": model_name,
        "temperature": args.temperature,
        "max_turns": args.max_turns,
        "max_tokens": args.max_tokens,
        "crop_size": args.crop_size,
        "timeout": args.timeout,
        "enable_thinking": not args.disable_thinking,
        "tool_choice": args.tool_choice,
        "breakpoint_start": bp_start,
        "breakpoint_end": bp_end,
        "unique_tasks": unique_tasks,
        "total_tasks": total_tasks,
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "eval": summary,
        "results_json": str(results_path),
        "eval_summary_txt": str(eval_summary_path),
    }
    (output_dir / "run_args.json").write_text(
        json.dumps(run_args_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        f"[output] output_dir={output_dir} "
        f"results_json={results_path} "
        f"eval_summary={eval_summary_path}",
        file=sys.stderr,
    )
    if args.print_results:
        print(json.dumps(json_results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
