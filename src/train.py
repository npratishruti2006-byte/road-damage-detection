# Reproduces exp01: YOLOv8n from COCO weights, 50 epochs, 768 px, batch 16.
# Usage (from the repository root):
#   python src/train.py --kit <competition folder> --work baseline_work
import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="baseline_work",
                    help="folder containing data.yaml and yolo_data/")
    ap.add_argument("--weights", default="weights/yolov8n_coco.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=768)
    ap.add_argument("--name", default="exp01_768")
    a = ap.parse_args()

    work = Path(a.work).resolve()
    model = YOLO(a.weights)
    model.train(
        data=str(work / "data.yaml"), epochs=a.epochs, imgsz=a.imgsz, batch=16,
        device=[0, 1], workers=2,
        project=str(work / "runs"), name=a.name, exist_ok=True,
        fliplr=0.5, flipud=0.0, patience=15, plots=True,
    )


if __name__ == "__main__":
    main()
