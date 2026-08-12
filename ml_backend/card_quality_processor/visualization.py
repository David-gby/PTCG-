from __future__ import annotations

from typing import Iterable, Mapping

import cv2
import numpy as np


def draw_outer_box_result(
    image: np.ndarray,
    points: Iterable[Iterable[float]] | None,
    confidence: float,
    error_code: str | None = None,
    message: str | None = None,
    method: str | None = None,
) -> np.ndarray:
    output = image.copy()
    success = points is not None and error_code is None
    color = (30, 220, 30) if success else (20, 30, 235)
    if points is not None:
        pts = np.rint(np.asarray(points, dtype=np.float32)).astype(np.int32).reshape(4, 2)
        cv2.polylines(output, [pts.reshape(-1, 1, 2)], True, color, max(2, round(output.shape[1] / 400)), cv2.LINE_AA)
        labels = ("TL", "TR", "BR", "BL")
        radius = max(5, round(output.shape[1] / 160))
        for label, (x, y) in zip(labels, pts):
            cv2.circle(output, (int(x), int(y)), radius, color, -1, cv2.LINE_AA)
            cv2.putText(
                output,
                label,
                (int(x) + radius + 3, int(y) - radius - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                max(0.55, output.shape[1] / 1400),
                color,
                2,
                cv2.LINE_AA,
            )
    lines = ["PASS" if success else "FAIL", f"confidence: {confidence:.3f}"]
    if method:
        lines.append(f"method: {method}")
    if error_code:
        lines.append(f"error: {error_code}")
    if message:
        lines.append(message[:90])
    font_scale = max(0.55, min(1.1, output.shape[1] / 1050.0))
    thickness = max(1, round(font_scale * 2))
    line_height = round(30 * font_scale + 8)
    panel_width = min(output.shape[1], max(330, round(output.shape[1] * 0.62)))
    panel_height = min(output.shape[0], 14 + line_height * len(lines))
    overlay = output.copy()
    cv2.rectangle(overlay, (0, 0), (panel_width, panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.72, output, 0.28, 0, output)
    y = round(27 * font_scale + 7)
    for line in lines:
        cv2.putText(output, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        y += line_height
    return output


def draw_outer_pose_result(
    image: np.ndarray,
    points: Iterable[Iterable[float]] | None,
    bbox: Iterable[float] | None,
    confidence: float,
    keypoint_confidence: Mapping[str, float] | None,
    error_code: str | None = None,
    message: str | None = None,
) -> np.ndarray:
    """Draw a YOLO Pose card bbox and TL/TR/BR/BL keypoints."""
    output = image.copy()
    success = points is not None and error_code is None
    color = (30, 220, 30) if success else (20, 30, 235)
    line_width = max(2, round(output.shape[1] / 400))
    if bbox is not None:
        bbox_values = np.rint(np.asarray(list(bbox), dtype=np.float32)).astype(np.int32).reshape(-1)
        if bbox_values.shape == (4,):
            x1, y1, x2, y2 = (int(value) for value in bbox_values)
            cv2.rectangle(output, (x1, y1), (x2, y2), color, line_width, cv2.LINE_AA)
    if points is not None:
        point_values = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        if point_values.shape == (4, 2) and np.isfinite(point_values).all():
            pts = np.rint(point_values).astype(np.int32)
            cv2.polylines(output, [pts.reshape(-1, 1, 2)], True, color, line_width, cv2.LINE_AA)
            labels = ("TL", "TR", "BR", "BL")
            radius = max(5, round(output.shape[1] / 160))
            for label, (x, y) in zip(labels, pts):
                cv2.circle(output, (int(x), int(y)), radius, color, -1, cv2.LINE_AA)
                point_conf = None
                if keypoint_confidence:
                    point_conf = keypoint_confidence.get(label.lower(), keypoint_confidence.get(label))
                label_text = label if point_conf is None else f"{label} {float(point_conf):.2f}"
                cv2.putText(
                    output,
                    label_text,
                    (int(x) + radius + 3, int(y) - radius - 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    max(0.50, output.shape[1] / 1500),
                    color,
                    2,
                    cv2.LINE_AA,
                )
    confidence_values = [float(value) for value in (keypoint_confidence or {}).values()]
    min_keypoint_confidence = min(confidence_values) if confidence_values else 0.0
    lines = [
        "PASS" if success else "FAIL",
        f"confidence: {float(confidence):.3f}",
        f"min keypoint confidence: {min_keypoint_confidence:.3f}",
    ]
    if error_code:
        lines.append(f"error: {error_code}")
    if message:
        lines.append(str(message)[:90])
    font_scale = max(0.52, min(1.05, output.shape[1] / 1100.0))
    thickness = max(1, round(font_scale * 2))
    line_height = round(30 * font_scale + 8)
    panel_width = min(output.shape[1], max(390, round(output.shape[1] * 0.72)))
    panel_height = min(output.shape[0], 14 + line_height * len(lines))
    overlay = output.copy()
    cv2.rectangle(overlay, (0, 0), (panel_width, panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.72, output, 0.28, 0, output)
    y = round(27 * font_scale + 7)
    for line in lines:
        cv2.putText(
            output,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        y += line_height
    return output
