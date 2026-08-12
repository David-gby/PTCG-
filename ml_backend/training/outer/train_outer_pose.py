from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the PTCG outer-card YOLO Pose model")
    parser.add_argument("--data", type=Path, default=Path("datasets/card_outer_pose/data.yaml"))
    parser.add_argument("--model", default="yolo11n-pose.pt", help="Ultralytics pose checkpoint")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None, help="Optional device such as 0, cpu, or mps")
    parser.add_argument("--lr0", type=float, default=None, help="Optional initial learning rate")
    parser.add_argument("--lrf", type=float, default=None, help="Final learning-rate fraction")
    parser.add_argument("--optimizer", default="auto", help="Ultralytics optimizer, e.g. auto, SGD, AdamW")
    parser.add_argument("--patience", type=int, default=30, help="Early-stopping patience")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--cache", action="store_true", help="Cache images during training")
    parser.add_argument("--resume", action="store_true", help="Resume optimizer and epoch state from --model")
    parser.add_argument("--freeze", type=int, default=None, help="Freeze the first N model layers")
    parser.add_argument("--save-period", type=int, default=-1, help="Checkpoint interval in epochs")
    parser.add_argument("--output", type=Path, default=Path("models/outer_pose.pt"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports/outer_pose_train"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.data.is_file():
        print(f"Dataset configuration not found: {args.data}")
        print("Create the dataset first; see README.md for the required structure and annotation order.")
        return 2
    config_root = PROJECT_ROOT / "reports" / "ultralytics_config"
    config_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_root))
    matplotlib_root = PROJECT_ROOT / "reports" / "matplotlib"
    matplotlib_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_root))
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics is not installed. Install it with:")
        print("pip install ultralytics")
        return 2

    model = YOLO(args.model)
    train_kwargs = {
        "data": str(args.data),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "patience": args.patience,
        "seed": args.seed,
        "workers": args.workers,
        "close_mosaic": args.close_mosaic,
        "cache": args.cache,
        "optimizer": args.optimizer,
        "save_period": args.save_period,
        "project": str(args.reports_dir.resolve().parent),
        "name": args.reports_dir.name,
        "exist_ok": True,
        "resume": args.resume,
    }
    if args.device is not None:
        train_kwargs["device"] = args.device
    if args.lr0 is not None:
        train_kwargs["lr0"] = args.lr0
    if args.lrf is not None:
        train_kwargs["lrf"] = args.lrf
    if args.freeze is not None:
        train_kwargs["freeze"] = args.freeze
    results = model.train(**train_kwargs)
    save_dir = Path(getattr(results, "save_dir", args.reports_dir))
    best_model = save_dir / "weights" / "best.pt"
    if not best_model.is_file():
        print(f"Training completed but the best checkpoint was not found: {best_model}")
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_model, args.output)
    print(f"Outer pose model: {args.output.resolve()}")
    print(f"Training reports: {save_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
