from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train physical-card outer silhouette segmentation")
    parser.add_argument("--data", type=Path, default=Path("datasets/card_outer_seg_v2_replay/data.yaml"))
    parser.add_argument("--model", type=Path, default=Path("yolov8s-seg.pt"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--lr0", type=float, default=0.005)
    parser.add_argument("--optimizer", default="SGD")
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--degrees", type=float, default=0.0)
    parser.add_argument("--translate", type=float, default=0.1)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--perspective", type=float, default=0.0)
    parser.add_argument("--hsv-h", type=float, default=0.015)
    parser.add_argument("--hsv-s", type=float, default=0.5)
    parser.add_argument("--hsv-v", type=float, default=0.4)
    parser.add_argument("--mosaic", type=float, default=0.5)
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--copy-paste", type=float, default=0.0)
    parser.add_argument("--fliplr", type=float, default=0.5)
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--freeze", type=int, default=0)
    parser.add_argument("--warmup-epochs", type=float, default=2.0)
    parser.add_argument("--cos-lr", action="store_true")
    parser.add_argument("--save-period", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--output", type=Path, default=Path("models/outer_seg.pt"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports/outer_seg_train"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data_path = args.data if args.data.is_absolute() else PROJECT_ROOT / args.data
    model_path = args.model if args.model.is_absolute() else PROJECT_ROOT / args.model
    output_path = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    reports_dir = args.reports_dir if args.reports_dir.is_absolute() else PROJECT_ROOT / args.reports_dir
    if not data_path.is_file():
        print(f"Dataset configuration not found: {data_path}")
        return 2
    if not model_path.is_file():
        print(f"Pretrained segmentation model not found: {model_path}")
        return 2
    reports_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(reports_dir / "ultralytics_config"))
    os.environ.setdefault("MPLCONFIGDIR", str(reports_dir / "matplotlib"))
    try:
        from ultralytics import YOLO
        from ultralytics.utils.torch_utils import strip_optimizer
    except ImportError:
        print("ultralytics is not installed. Install it with: pip install ultralytics")
        return 2

    model = YOLO(str(model_path))
    result = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=0,
        project=str(reports_dir),
        name="run",
        exist_ok=True,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=0.10,
        weight_decay=args.weight_decay,
        patience=args.patience,
        seed=args.seed,
        deterministic=True,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        perspective=args.perspective,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        mosaic=args.mosaic,
        mixup=args.mixup,
        copy_paste=args.copy_paste,
        fliplr=args.fliplr,
        close_mosaic=args.close_mosaic,
        freeze=args.freeze,
        warmup_epochs=args.warmup_epochs,
        cos_lr=args.cos_lr,
        cache=False,
        amp=True,
        save_period=args.save_period,
        plots=True,
    )
    best_path = Path(result.save_dir) / "weights" / "best.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    strip_optimizer(best_path, output_path)
    summary = {
        "data": str(data_path),
        "initial_model": str(model_path),
        "output": str(output_path),
        "training_dir": str(result.save_dir),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "lr0": args.lr0,
        "optimizer": args.optimizer,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "augmentation": {
            "degrees": args.degrees,
            "translate": args.translate,
            "scale": args.scale,
            "perspective": args.perspective,
            "hsv_h": args.hsv_h,
            "hsv_s": args.hsv_s,
            "hsv_v": args.hsv_v,
            "mosaic": args.mosaic,
            "mixup": args.mixup,
            "copy_paste": args.copy_paste,
            "fliplr": args.fliplr,
            "close_mosaic": args.close_mosaic,
        },
        "freeze": args.freeze,
        "warmup_epochs": args.warmup_epochs,
        "cos_lr": args.cos_lr,
        "save_period": args.save_period,
        "seed": args.seed,
        "selection_note": "For deployment, compare saved checkpoints by physical corner error, not mAP alone.",
    }
    (reports_dir / "training_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
