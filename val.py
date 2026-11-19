"""Validate a CXS-YOLO checkpoint on the validation or the test split."""
import argparse

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="weights/CXS-YOLO.pt")
    parser.add_argument("--data", required=True, help="Path to dataset YAML")
    parser.add_argument("--split", default="val", choices=["val", "test"], help="Dataset split to evaluate")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    YOLO(args.weights).val(data=args.data, split=args.split, imgsz=args.imgsz, device=args.device)


if __name__ == "__main__":
    main()
