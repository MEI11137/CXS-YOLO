"""Run inference with a CXS-YOLO checkpoint."""
import argparse

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Image, video, directory, or stream source")
    parser.add_argument("--weights", default="weights/CXS-YOLO.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    YOLO(args.weights).predict(source=args.source, imgsz=args.imgsz, device=args.device, save=True)


if __name__ == "__main__":
    main()
