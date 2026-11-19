"""Run one paper ablation experiment with the reported training settings.

The generated args.yaml and checkpoint train_args are written by Ultralytics
from the settings actually used for training. Historical run files are never
rewritten by this script.
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "ultralytics" / "cfg" / "models" / "11"

EXPERIMENTS = {
    1: "yolo11.yaml",
    2: "yolo11-C3k2_MCC.yaml",
    3: "yolo11-CAWD.yaml",
    4: "yolo11-Detect_SALSCD.yaml",
    5: "yolo11-MCC-CAWD.yaml",
    6: "yolo11-MCC-SALSCD.yaml",
    7: "yolo11-CAWD-SALSCD.yaml",
    8: "yolo11-MCC-CAWD-SALSCD.yaml",
    9: "CXS-YOLO.yaml",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, help="Path to the bird-nest dataset YAML")
    parser.add_argument("--experiment", type=int, choices=EXPERIMENTS, default=9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="0")
    parser.add_argument("--pretrained", type=Path, help="Optional common YOLO11n checkpoint for weight transfer")
    parser.add_argument("--project", type=Path, default=ROOT / "runs" / "paper_ablation")
    parser.add_argument("--name", help="Run directory name")
    parser.add_argument("--dry-run", action="store_true", help="Build the selected model without training")
    return parser.parse_args()


def main():
    args = parse_args()
    model_yaml = MODEL_DIR / EXPERIMENTS[args.experiment]
    model = YOLO(model_yaml)

    if args.pretrained:
        model.load(args.pretrained)

    if args.dry_run:
        parameters = sum(parameter.numel() for parameter in model.model.parameters())
        print(f"experiment={args.experiment} model={model_yaml.name} parameters={parameters}")
        return

    if args.data is None:
        raise SystemExit("--data is required unless --dry-run is used")

    run_name = args.name or f"exp{args.experiment}_{model_yaml.stem}_seed{args.seed}"
    model.train(
        data=str(args.data),
        imgsz=640,
        batch=8,
        epochs=300,
        optimizer="SGD",
        lr0=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        workers=4,
        seed=args.seed,
        deterministic=True,
        amp=False,
        close_mosaic=10,
        patience=0,
        pretrained=str(args.pretrained) if args.pretrained else False,
        project=str(args.project),
        name=run_name,
        exist_ok=False,
        plots=True,
    )


if __name__ == "__main__":
    main()
