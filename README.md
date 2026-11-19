# CXS-YOLO

Reproducible code package for the manuscript *"A lightweight bird nest detection model based on YOLO11n for transmission lines"*.

## Environment

- Python 3.10.18
- PyTorch 2.9.0, CUDA 12.8, cuDNN 9.10.2
- Ultralytics 8.3.9 (included in this package)
- one NVIDIA GeForce RTX 4060 Laptop GPU (8 GB)

```bash
pip install -r requirements.txt
```

## Reproduction

Train one configuration (1 = YOLO11n baseline, 2 = +C3k2_MCC, 3 = +CAWD, 4 = +Detect_SALSCD, 5 = +MCC+CAWD, 6 = +MCC+SALSCD, 7 = +CAWD+SALSCD, 8 = +MCC+CAWD+SALSCD, 9 = CXS-YOLO):

```bash
python train.py --experiment 9 --data data.yaml
```

Evaluate the selected checkpoint on the held-out test split:

```bash
python val.py --weights runs/detect/exp/weights/best.pt --data data.yaml --split test
```

Run inference:

```bash
python detect.py --weights runs/detect/exp/weights/best.pt --source path/to/images
```
