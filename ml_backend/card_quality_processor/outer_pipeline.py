from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping

from .config import normalize_config
from .io_utils import read_image, write_image, write_json
from .outer_detection import detect_outer_box
from .rectification import rectify_card
from .visualization import draw_outer_box_result


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
CSV_FIELDS = [
    "image_id",
    "image_path",
    "success",
    "confidence",
    "method",
    "error_code",
    "selected_area_ratio",
    "aspect_ratio",
    "aspect_ratio_error",
    "edge_score",
    "final_score",
    "rectified_path",
    "outer_visualization_path",
    "message",
]


def _empty_outer(width: int = 0, height: int = 0, message: str = "Could not read image.") -> dict[str, Any]:
    return {
        "success": False,
        "points": None,
        "confidence": 0.0,
        "method": None,
        "error_code": "NO_CARD_CANDIDATE",
        "message": message,
        "metrics": {
            "image_width": width,
            "image_height": height,
            "candidate_count": 0,
            "selected_area_ratio": 0.0,
            "aspect_ratio": 0.0,
            "aspect_ratio_error": 1.0,
            "edge_score": 0.0,
            "center_score": 0.0,
            "geometry_score": 0.0,
            "straightness_score": 0.0,
            "final_score": 0.0,
        },
        "candidates": [],
    }


def process_outer_and_rectify(
    image_path: str | Path,
    output_dir: str | Path,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    image_id = image_path.stem
    image_output = output_dir / image_id
    debug_dir = image_output / "debug"
    image_output.mkdir(parents=True, exist_ok=True)
    # Reprocessing must not leave a stale success artifact after a new failure.
    for generated_name in ("outer_selected.jpg", "outer_failed.jpg", "rectified_card.jpg", "outer_result.json"):
        (image_output / generated_name).unlink(missing_ok=True)
    image = read_image(image_path)
    output_paths: dict[str, str | None] = {
        "outer_visualization_path": None,
        "rectified_path": None,
        "result_json_path": str(image_output / "outer_result.json"),
        "debug_dir": str(debug_dir),
    }
    if image is None:
        outer = _empty_outer(message=f"Could not read image: {image_path}")
        # A readable visualization cannot be produced for a corrupt file.
        write_json(image_output / "outer_result.json", outer)
        return {
            "image_path": str(image_path),
            "success": False,
            "outer_result": outer,
            "rectification_result": None,
            "output_paths": output_paths,
        }

    cfg = normalize_config(config)
    outer = detect_outer_box(image, cfg, save_debug=True, debug_dir=debug_dir)
    visualization = draw_outer_box_result(
        image,
        outer["points"],
        outer["confidence"],
        outer["error_code"],
        outer["message"],
        outer["method"],
    )
    visualization_name = "outer_selected.jpg" if outer["success"] else "outer_failed.jpg"
    visualization_path = write_image(image_output / visualization_name, visualization)
    output_paths["outer_visualization_path"] = str(visualization_path)

    rectification: dict[str, Any] | None = None
    if outer["success"]:
        rcfg = cfg["rectification"]
        rectification = rectify_card(
            image,
            outer["points"],
            (int(rcfg["output_width"]), int(rcfg["output_height"])),
            cfg,
            save_debug=False,
        )
        if rectification["success"]:
            rectified_path = write_image(image_output / "rectified_card.jpg", rectification["rectified_image"])
            output_paths["rectified_path"] = str(rectified_path)
            rectification["confidence"] = float(outer["confidence"])

    json_result = {
        "image_path": str(image_path),
        "image_id": image_id,
        "success": bool(outer["success"] and rectification and rectification["success"]),
        "outer_result": outer,
        "rectification_result": None
        if rectification is None
        else {key: value for key, value in rectification.items() if key != "rectified_image"},
        "output_paths": output_paths,
    }
    write_json(image_output / "outer_result.json", json_result)
    return {
        "image_path": str(image_path),
        "success": json_result["success"],
        "outer_result": outer,
        "rectification_result": rectification,
        "output_paths": output_paths,
    }


def _csv_row(result: dict[str, Any]) -> dict[str, Any]:
    outer = result["outer_result"]
    metrics = outer.get("metrics", {})
    paths = result.get("output_paths", {})
    return {
        "image_id": Path(result["image_path"]).stem,
        "image_path": result["image_path"],
        "success": result["success"],
        "confidence": outer.get("confidence", 0.0),
        "method": outer.get("method") or "",
        "error_code": outer.get("error_code") or "",
        "selected_area_ratio": metrics.get("selected_area_ratio", 0.0),
        "aspect_ratio": metrics.get("aspect_ratio", 0.0),
        "aspect_ratio_error": metrics.get("aspect_ratio_error", 1.0),
        "edge_score": metrics.get("edge_score", 0.0),
        "final_score": metrics.get("final_score", 0.0),
        "rectified_path": paths.get("rectified_path") or "",
        "outer_visualization_path": paths.get("outer_visualization_path") or "",
        "message": outer.get("message", ""),
    }


def batch_process_outer_dataset(
    input_dir: str | Path,
    output_dir: str | Path = "data/processed/outer_rectified",
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images = (
        sorted(path for path in input_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
        if input_dir.exists()
        else []
    )
    results = [process_outer_and_rectify(path, output_dir, config) for path in images]
    report_path = output_dir / "batch_outer_report.csv"
    with report_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(_csv_row(result) for result in results)
    confidences = [float(result["outer_result"].get("confidence", 0.0)) for result in results]
    successful = sum(bool(result["success"]) for result in results)
    return {
        "processed_count": len(results),
        "success_count": successful,
        "failure_count": len(results) - successful,
        "average_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
        "report_path": str(report_path),
        "output_dir": str(output_dir),
        "results": results,
    }
