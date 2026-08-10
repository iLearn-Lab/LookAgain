"""Grounding agent for the LookAgain
"""

from __future__ import annotations

import base64
import io
import json
import logging
import mimetypes
import uuid
from pathlib import Path
from typing import Any, TypedDict, cast

from openai import OpenAI
from PIL import Image

from .utils import compute_crop_region, draw_cirle_with_marker

logger = logging.getLogger(__name__)


SYSTEM_PROMPT_SFT = """# Role
- An agent with strong computer knowledge, designed to locate GUI elements on desktop computer given a screenshot an an instruction.
- The user provides tools to help you locate the GUI elements. You are allowed to do one tool call each turn before getting any tool response.
- You need to output the coordinate in relative coordinate [0, 1000].
- Your goal is to make the center of bounding box falls on the target element. As a bonus, you shuold try your best to make the bounding box cover all interactive area of the target element.

# Important Notes
- You should first use `locate` tool to get bounding box coordinates for the element. The bounding box area will be used to perform mouse action such as clicking, dragging, so it is important to make sure the coordinate is accurate and the bounding box shuold cover all area that is interactive.
- After `locate` tool is called, you will get an image patch around the bounding box, and you will have a red circle with a marker in its center to indicate where the coordinate is. Then, you will need to use `confirm` tool to check if the location is correct enough.
- If you use `confirm` with status "correct", the procedure ends and downstream system get the coordinates.
- It's possible for the user to come up with some instruction that is not applicable (no answer) in this screenshot, you should use `confirm` tool with status `failed` and write the reason in `feedback` field.
- If the instruction is applicable and the current location is not correct enough, continue to call `locate` with a better bounding bbox.
"""


LOCATE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "locate",
        "description": "locate the element by bounding box",
        "parameters": {
            "type": "object",
            "properties": {
                "x1": {
                    "type": "integer",
                    "description": "bounding box's top left corner x coordinate of the element. Should between 0-1000",
                },
                "y1": {
                    "type": "integer",
                    "description": "bounding box's top left corner y coordinate of the element. Should between 0-1000",
                },
                "x2": {
                    "type": "integer",
                    "description": "bounding box's bottom right corner x coordinate of the element. Should between 0-1000",
                },
                "y2": {
                    "type": "integer",
                    "description": "bounding box's bottom right corner y coordinate of the element. Should between 0-1000",
                },
            },
            "required": ["x1", "y1", "x2", "y2"],
        },
    },
}

CONFIRM_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "confirm",
        "description": "Confirm the location of the element or comfirm that the instruction is not applicable",
        "parameters": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["success", "failed"],
                    "description": "Status of the confirmation. Should be one of 'success', 'failed'",
                },
                "feedback": {
                    "type": "string",
                    "description": "Feedback of the confirmation. Should have this field when status=failed",
                },
            },
            "required": ["status"],
        },
    },
}

TOOLS: list[dict[str, Any]] = [LOCATE_TOOL, CONFIRM_TOOL]


class BBox2DItem(TypedDict):
    bbox_2d: list[float]
    label: str


class SFTToolCallTurnRecord(TypedDict):
    turn: int
    tool_name: str | None
    tool_args: dict[str, Any] | None
    tool_call_id: str | None
    model_content: str | None
    reasoning_content: str | None
    error: str | None
    crop_image_path: str | None


class SFTToolCallResult(TypedDict):
    """Result schema consumed by the SFT tool-call runner."""

    task_id: str
    status: str
    feedback: str | None
    x1: int | None
    y1: int | None
    x2: int | None
    y2: int | None
    locate_count: int
    total_turns: int
    pred_bboxes: list[BBox2DItem] | None
    turns: list[SFTToolCallTurnRecord]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _image_path_to_data_url(image_path: str) -> str:
    path = Path(image_path)
    mime, _ = mimetypes.guess_type(str(path))
    if not mime:
        mime = "image/png"
    b64 = base64.standard_b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _pil_to_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.standard_b64encode(buf.getvalue()).decode("ascii")


def _make_marked_crop(
    image_path: str,
    bbox_2d: list[float],
    crop_size: int,
    save_path: Path | None = None,
) -> tuple[Image.Image, str]:
    """Open the screenshot, draw a transparent red marker at the predicted bbox
    center, and crop ``crop_size`` x ``crop_size`` around it. Returns
    ``(PIL crop, data URL)``.
    bbox_2d is in the 0-1000 normalized space (what the model emits).
    """
    with Image.open(image_path) as img:
        img_with_marker = draw_cirle_with_marker(img, bbox_2d)
        iw, ih = img_with_marker.size
        x1p = float(bbox_2d[0]) / 1000.0 * iw
        y1p = float(bbox_2d[1]) / 1000.0 * ih
        x2p = float(bbox_2d[2]) / 1000.0 * iw
        y2p = float(bbox_2d[3]) / 1000.0 * ih
        left, top, right, bottom = compute_crop_region(
            [x1p, y1p, x2p, y2p],
            (iw, ih),
            crop_size,
            crop_size,
        )
        patch = img_with_marker.crop((left, top, right, bottom)).copy()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        patch.convert("RGB").save(save_path)

    return patch, _pil_to_data_url(patch)


def _wrap_thinking(reasoning: str | None, content: str | None) -> str:
    """Re-embed thinking into assistant.content so the Qwen3 chat_template parses
    it back out as proper <think>...</think> on the next turn."""
    reasoning = (reasoning or "").strip()
    content = content or ""
    if not reasoning:
        return content
    return f"<think>\n{reasoning}\n</think>\n{content}"


def _norm1000_to_pixels(
    x1: float, y1: float, x2: float, y2: float, w: int, h: int
) -> tuple[int, int, int, int]:
    return (
        int(float(x1) / 1000.0 * w),
        int(float(y1) / 1000.0 * h),
        int(float(x2) / 1000.0 * w),
        int(float(y2) / 1000.0 * h),
    )


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class SFTToolCallGroundingAgent:
    """locate/confirm agent for the Qwen3-VL/Qwen3.5 deployed on vLLM."""

    def __init__(
        self,
        api_base: str = "http://localhost:34575/v1",
        api_key: str = "EMPTY",
        model_name: str = "Qwen/Qwen3.5-27B",
        temperature: float = 0.0,
        max_tokens: int | None = None,
        max_turns: int = 5,
        crop_size: int = 512,
        system_prompt: str = SYSTEM_PROMPT_SFT,
        enable_thinking: bool = True,
        tool_choice: str = "required",
        timeout: float | None = 120.0,
        temp_dir: str | Path | None = None,
        verbose: bool = False,
        **_: Any,  # accept and ignore extra runner kwargs for swap-in compat
    ) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be >= 1")
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_turns = max_turns
        self.crop_size = crop_size
        self.system_prompt = system_prompt
        self.enable_thinking = enable_thinking
        self.tool_choice = tool_choice
        self.verbose = verbose
        self.temp_dir = (
            Path(temp_dir) if temp_dir is not None else Path("./results/sft_grounding")
        )

        client_kw: dict[str, Any] = {"base_url": api_base, "api_key": api_key}
        if timeout is not None:
            client_kw["timeout"] = timeout
        self._client = OpenAI(**client_kw)

    # ------------------------------------------------------------------
    # Chat completion
    # ------------------------------------------------------------------

    def _complete(self, messages: list[dict[str, Any]]) -> Any:
        create_kwargs: dict[str, Any] = dict(
            model=self.model_name,
            messages=messages,
            tools=TOOLS,
            tool_choice=self.tool_choice,
            temperature=self.temperature,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": self.enable_thinking}
            },
        )
        if self.max_tokens is not None:
            create_kwargs["max_tokens"] = self.max_tokens
        return self._client.chat.completions.create(**create_kwargs)

    # ------------------------------------------------------------------
    # Result builders
    # ------------------------------------------------------------------

    def _build_success(
        self,
        *,
        task_id: str,
        instruction: str,
        last_bbox_norm1000: list[float],
        last_pixel_xyxy: tuple[int, int, int, int],
        locate_count: int,
        turns: list[SFTToolCallTurnRecord],
    ) -> SFTToolCallResult:
        px1, py1, px2, py2 = last_pixel_xyxy
        pred_bboxes = [
            cast(
                BBox2DItem,
                {"bbox_2d": [float(v) for v in last_bbox_norm1000], "label": instruction},
            )
        ]
        return cast(
            SFTToolCallResult,
            {
                "task_id": task_id,
                "status": "success",
                "feedback": None,
                "x1": px1,
                "y1": py1,
                "x2": px2,
                "y2": py2,
                "locate_count": locate_count,
                "total_turns": len(turns),
                "pred_bboxes": pred_bboxes,
                "turns": turns,
            },
        )

    def _build_failed(
        self,
        *,
        task_id: str,
        feedback: str,
        last_pixel_xyxy: tuple[int, int, int, int] | None,
        locate_count: int,
        turns: list[SFTToolCallTurnRecord],
    ) -> SFTToolCallResult:
        if last_pixel_xyxy is not None:
            px1, py1, px2, py2 = last_pixel_xyxy
        else:
            px1 = py1 = px2 = py2 = None  # type: ignore[assignment]
        return cast(
            SFTToolCallResult,
            {
                "task_id": task_id,
                "status": "failed",
                "feedback": feedback,
                "x1": px1,
                "y1": py1,
                "x2": px2,
                "y2": py2,
                "locate_count": locate_count,
                "total_turns": len(turns),
                "pred_bboxes": None,
                "turns": turns,
            },
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        instruction: str,
        image_path: str,
        task_id: str | None = None,
        rollout_id: str = "0",
    ) -> SFTToolCallResult:
        image_path = str(Path(image_path).expanduser())
        if not task_id:
            task_id = str(uuid.uuid4())
        with Image.open(image_path) as img:
            screen_w, screen_h = img.size

        session_dir = (self.temp_dir / task_id / rollout_id).resolve()
        session_dir.mkdir(parents=True, exist_ok=True)
        traj_path = session_dir / "meta.json"
        traj: dict[str, Any] = {
            "task_id": task_id,
            "instruction": instruction,
            "image_path": image_path,
            "model_name": self.model_name,
            "traj": [],
        }
        traj_path.write_text(json.dumps(traj, ensure_ascii=False, indent=2), encoding="utf-8")

        screenshot_url = _image_path_to_data_url(image_path)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": screenshot_url}},
                    {"type": "text", "text": f"\n{instruction}"},
                ],
            },
        ]

        turn_records: list[SFTToolCallTurnRecord] = []
        locate_count = 0
        last_bbox_norm1000: list[float] | None = None
        last_pixel_xyxy: tuple[int, int, int, int] | None = None

        try:
            for turn in range(self.max_turns):
                record: SFTToolCallTurnRecord = {
                    "turn": turn + 1,
                    "tool_name": None,
                    "tool_args": None,
                    "tool_call_id": None,
                    "model_content": None,
                    "reasoning_content": None,
                    "error": None,
                    "crop_image_path": None,
                }

                try:
                    completion = self._complete(messages)
                except Exception as exc:  # noqa: BLE001
                    record["error"] = f"{type(exc).__name__}: {exc}"
                    turn_records.append(record)
                    return self._build_failed(
                        task_id=task_id,
                        feedback=f"Model API error: {exc}",
                        last_pixel_xyxy=last_pixel_xyxy,
                        locate_count=locate_count,
                        turns=turn_records,
                    )

                msg = completion.choices[0].message
                reasoning = getattr(msg, "reasoning", None) or getattr(
                    msg, "reasoning_content", None
                )
                tool_calls = msg.tool_calls or []
                record["model_content"] = msg.content
                record["reasoning_content"] = reasoning

                if not tool_calls:
                    record["error"] = "no tool_calls in completion"
                    turn_records.append(record)
                    if last_bbox_norm1000 is not None and last_pixel_xyxy is not None:
                        return self._build_success(
                            task_id=task_id,
                            instruction=instruction,
                            last_bbox_norm1000=last_bbox_norm1000,
                            last_pixel_xyxy=last_pixel_xyxy,
                            locate_count=locate_count,
                            turns=turn_records,
                        )
                    return self._build_failed(
                        task_id=task_id,
                        feedback="Model returned no tool call",
                        last_pixel_xyxy=last_pixel_xyxy,
                        locate_count=locate_count,
                        turns=turn_records,
                    )

                tc = tool_calls[0]
                tool_name = tc.function.name
                raw_args = tc.function.arguments
                try:
                    tool_args = (
                        json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args or {})
                    )
                except json.JSONDecodeError as exc:
                    record["error"] = f"bad tool arguments JSON: {exc}"
                    turn_records.append(record)
                    if last_bbox_norm1000 is not None and last_pixel_xyxy is not None:
                        return self._build_success(
                            task_id=task_id,
                            instruction=instruction,
                            last_bbox_norm1000=last_bbox_norm1000,
                            last_pixel_xyxy=last_pixel_xyxy,
                            locate_count=locate_count,
                            turns=turn_records,
                        )
                    return self._build_failed(
                        task_id=task_id,
                        feedback=f"Bad tool arguments JSON: {exc}",
                        last_pixel_xyxy=last_pixel_xyxy,
                        locate_count=locate_count,
                        turns=turn_records,
                    )
                record["tool_name"] = tool_name
                record["tool_args"] = tool_args
                record["tool_call_id"] = tc.id

                if self.verbose:
                    print(
                        f"[Turn {turn + 1}] {tool_name}({tool_args})"
                        + (f"  reasoning='{reasoning[:80]}...'" if reasoning else "")
                    )

                messages.append(
                    {
                        "role": "assistant",
                        "content": _wrap_thinking(reasoning, msg.content),
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tool_name,
                                    "arguments": raw_args
                                    if isinstance(raw_args, str)
                                    else json.dumps(tool_args, ensure_ascii=False),
                                },
                            }
                        ],
                    }
                )

                traj_entry: dict[str, Any] = {
                    "function_call": tool_name,
                    "args": tool_args,
                    "reasoning_content": reasoning,
                }
                traj["traj"].append(traj_entry)

                if tool_name == "locate":
                    locate_count += 1
                    try:
                        bbox_2d = [
                            float(tool_args["x1"]),
                            float(tool_args["y1"]),
                            float(tool_args["x2"]),
                            float(tool_args["y2"]),
                        ]
                    except (KeyError, TypeError, ValueError) as exc:
                        record["error"] = f"bad locate args: {exc}"
                        turn_records.append(record)
                        return self._build_failed(
                            task_id=task_id,
                            feedback=f"Bad locate args: {exc}",
                            last_pixel_xyxy=last_pixel_xyxy,
                            locate_count=locate_count,
                            turns=turn_records,
                        )
                    last_bbox_norm1000 = bbox_2d
                    last_pixel_xyxy = _norm1000_to_pixels(*bbox_2d, screen_w, screen_h)

                    crop_path = session_dir / f"crop_turn_{turn + 1}.png"
                    _, crop_url = _make_marked_crop(
                        image_path, bbox_2d, self.crop_size, save_path=crop_path
                    )
                    record["crop_image_path"] = str(crop_path)
                    traj_entry["crop_image_path"] = str(crop_path)

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": [
                                {"type": "text", "text": "Screenshot: "},
                                {"type": "image_url", "image_url": {"url": crop_url}},
                            ],
                        }
                    )
                    turn_records.append(record)
                    continue

                if tool_name == "confirm":
                    status = str(tool_args.get("status", "")).lower()
                    turn_records.append(record)
                    if status == "success":
                        if last_bbox_norm1000 is None or last_pixel_xyxy is None:
                            return self._build_failed(
                                task_id=task_id,
                                feedback="Confirmed success without any prior locate",
                                last_pixel_xyxy=None,
                                locate_count=locate_count,
                                turns=turn_records,
                            )
                        return self._build_success(
                            task_id=task_id,
                            instruction=instruction,
                            last_bbox_norm1000=last_bbox_norm1000,
                            last_pixel_xyxy=last_pixel_xyxy,
                            locate_count=locate_count,
                            turns=turn_records,
                        )
                    if status == "failed":
                        feedback = str(tool_args.get("feedback", "No feedback provided"))
                        return self._build_failed(
                            task_id=task_id,
                            feedback=feedback,
                            last_pixel_xyxy=last_pixel_xyxy,
                            locate_count=locate_count,
                            turns=turn_records,
                        )
                    return self._build_failed(
                        task_id=task_id,
                        feedback=f"Unknown confirm status: {status!r}",
                        last_pixel_xyxy=last_pixel_xyxy,
                        locate_count=locate_count,
                        turns=turn_records,
                    )

                record["error"] = f"unknown tool: {tool_name}"
                turn_records.append(record)
                return self._build_failed(
                    task_id=task_id,
                    feedback=f"Unknown tool: {tool_name}",
                    last_pixel_xyxy=last_pixel_xyxy,
                    locate_count=locate_count,
                    turns=turn_records,
                )

            return self._build_failed(
                task_id=task_id,
                feedback=f"Maximum turns ({self.max_turns}) reached without confirm",
                last_pixel_xyxy=last_pixel_xyxy,
                locate_count=locate_count,
                turns=turn_records,
            )
        finally:
            traj_path.write_text(
                json.dumps(traj, ensure_ascii=False, indent=2), encoding="utf-8"
            )
