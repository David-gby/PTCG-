from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


INSTALL_HINT = (
    "python -m pip install ultralytics -i https://mirrors.aliyun.com/pypi/simple/ "
    "--trusted-host mirrors.aliyun.com"
)


def configure_runtime_dirs(debug_dir: Path | None = None) -> None:
    debug_dir = debug_dir or ROOT / "training_outputs" / "runtime"
    yolo_config_dir = debug_dir / "ultralytics_config"
    matplotlib_dir = debug_dir / "matplotlib"
    yolo_config_dir.mkdir(parents=True, exist_ok=True)
    matplotlib_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(yolo_config_dir))
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_dir))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO segmentation for card inner-frame detection.")
    parser.add_argument(
        "--model",
        default=str(ROOT / "models" / "inner_frame_yolo_v3_base_candidate.pt"),
        help="YOLO segmentation model or checkpoint.",
    )
    parser.add_argument(
        "--data",
        default=str(ROOT / "training" / "data" / "inner_frame_seg" / "data.yaml"),
        help="Dataset YAML path.",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size.")
    parser.add_argument("--batch", type=int, default=4, help="Batch size.")
    parser.add_argument(
        "--project",
        default=str(ROOT / "training_outputs" / "inner_frame_seg"),
        help="Output project directory.",
    )
    parser.add_argument("--name", default="inner_frame_seg", help="Run name.")
    parser.add_argument("--device", default="", help="Training device, for example 0 or cpu.")
    parser.add_argument("--workers", type=int, default=0, help="Data loader workers.")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate.")
    parser.add_argument("--optimizer", default="auto", help="Ultralytics optimizer name.")
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--degrees", type=float, default=0.0)
    parser.add_argument("--translate", type=float, default=0.1)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--perspective", type=float, default=0.0)
    parser.add_argument("--hsv-v", type=float, default=0.4)
    parser.add_argument("--close-mosaic", type=int, default=10)
    return parser.parse_args(argv)


def _read_yaml_scalar(text: str, key: str, default: str) -> str:
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(f"{key}:"):
            value = line.split(":", 1)[1].strip()
            return value.strip("'\"") or default
    return default


def write_resolved_dataset_yaml(data_path: Path) -> Path:
    dataset_dir = data_path.resolve().parent
    runtime_yaml = data_path.resolve().parent / "resolved_dataset.yaml"
    runtime_yaml.parent.mkdir(parents=True, exist_ok=True)
    source_text = data_path.read_text(encoding="utf-8", errors="replace")
    dataset_path = _read_yaml_scalar(source_text, "path", ".")
    if dataset_path == ".":
        dataset_path = dataset_dir.as_posix()
    train_path = _read_yaml_scalar(source_text, "train", "train/images")
    val_path = _read_yaml_scalar(source_text, "val", "valid/images")
    runtime_yaml.write_text(
        "\n".join(
            [
                f"path: {json.dumps(dataset_path)}",
                f"train: {train_path}",
                f"val: {val_path}",
                "test: ''",
                "names:",
                "  0: inner_frame",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return runtime_yaml


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_runtime_dirs(Path(args.project).resolve() / "runtime")

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics is not installed. Install it with:")
        print(INSTALL_HINT)
        return 1

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Dataset config not found: {data_path}")
        return 1
    runtime_data_path = write_resolved_dataset_yaml(data_path)

    model = YOLO(args.model)
    train_options = dict(
        data=str(runtime_data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        workers=args.workers,
        patience=args.patience,
        seed=args.seed,
        deterministic=True,
        lr0=args.lr0,
        optimizer=args.optimizer,
        weight_decay=args.weight_decay,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        perspective=args.perspective,
        hsv_v=args.hsv_v,
        close_mosaic=args.close_mosaic,
        amp=True,
    )
    if args.device:
        train_options["device"] = args.device
    model.train(**train_options)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
