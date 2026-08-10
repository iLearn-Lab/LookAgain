from __future__ import annotations

import math
from typing import List, Tuple

from PIL import Image, ImageDraw


def get_annotation_width(image_size: tuple[int, int]) -> int:
    width, height = image_size
    return max(2, int(round(min(width, height) * 0.004)))


def apply_transparent_point_marker(
    base_image: Image.Image,
    point_xy: list[float] | tuple[float, float] | None,
    annotation_width: int,
    draw_center_dot: bool = True,
    marker_alpha: float = 0.45,
) -> Image.Image:
    if point_xy is None:
        return base_image

    alpha_int = max(0, min(255, int(round(marker_alpha * 255))))
    point_x = float(point_xy[0])
    point_y = float(point_xy[1])
    working_image = base_image if base_image.mode == "RGBA" else base_image.convert("RGBA")
    marker_overlay = Image.new("RGBA", working_image.size, (0, 0, 0, 0))
    marker_draw = ImageDraw.Draw(marker_overlay)

    ring_radius = max(12, int(round(min(working_image.size) * 0.025)))
    base_stroke_width = max(1, int(round(annotation_width)))
    ring_width = max(base_stroke_width + 1, int(round(ring_radius * 0.18)))
    marker_draw.ellipse(
        (
            point_x - ring_radius,
            point_y - ring_radius,
            point_x + ring_radius,
            point_y + ring_radius,
        ),
        outline=(255, 0, 0, alpha_int),
        width=ring_width,
    )

    if draw_center_dot:
        center_radius = max(2, int(round(ring_radius * 0.15)))
        center_alpha = max(0, min(255, int(round(alpha_int * 0.9))))
        outline_alpha = max(0, min(255, int(round(alpha_int * 1.1))))
        marker_draw.ellipse(
            (
                point_x - center_radius,
                point_y - center_radius,
                point_x + center_radius,
                point_y + center_radius,
            ),
            fill=(255, 0, 0, center_alpha),
            outline=(255, 255, 255, outline_alpha),
            width=max(1, ring_width // 3),
        )

    return Image.alpha_composite(working_image, marker_overlay)


def draw_cirle_with_marker(
    pil_image: Image.Image,
    bbox: List[float | int],
    color: str = "red",
) -> Image.Image:
    del color
    image_with_circle = pil_image.copy()
    annotation_width = get_annotation_width(image_with_circle.size)
    x1, y1, x2, y2 = bbox
    width, height = image_with_circle.size
    x1 = float(x1) * width / 1000.0
    y1 = float(y1) * height / 1000.0
    x2 = float(x2) * width / 1000.0
    y2 = float(y2) * height / 1000.0
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    return apply_transparent_point_marker(
        image_with_circle,
        [center_x, center_y],
        annotation_width,
        draw_center_dot=True,
        marker_alpha=0.45,
    )


def compute_crop_region(
    bbox_pixel: List[float],
    image_size: Tuple[int, int],
    crop_width: int,
    crop_height: int,
) -> List[int]:
    image_width, image_height = image_size
    target_width = max(1, min(int(crop_width), int(image_width)))
    target_height = max(1, min(int(crop_height), int(image_height)))
    bbox_width = max(1, int(math.ceil(float(bbox_pixel[2]) - float(bbox_pixel[0]))))
    bbox_height = max(1, int(math.ceil(float(bbox_pixel[3]) - float(bbox_pixel[1]))))
    target_width = min(int(image_width), max(target_width, bbox_width))
    target_height = min(int(image_height), max(target_height, bbox_height))
    center_x = (float(bbox_pixel[0]) + float(bbox_pixel[2])) / 2.0
    center_y = (float(bbox_pixel[1]) + float(bbox_pixel[3])) / 2.0
    left = int(round(center_x - (target_width / 2.0)))
    top = int(round(center_y - (target_height / 2.0)))
    left = max(0, min(int(image_width) - target_width, left))
    top = max(0, min(int(image_height) - target_height, top))
    right = left + target_width
    bottom = top + target_height
    return [left, top, right, bottom]
