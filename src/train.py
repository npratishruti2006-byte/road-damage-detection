# src/train.py
# Reproduces the submitted model family (exp01: 768 px, 50 epochs, batch 16, seed 0).
# Builds the YOLO dataset from the competition kit, then trains.
#
# Usage (repository root):
#   python src/train.py --kit /path/to/competition --device 0,1
#
# exp01 was trained with device=[0, 1]; "0,1" is the same request as a string.
import argparse
import sys
from pathlib import Path

from ultralytics import YOLO


def prepare_dataset(kit, work):
    sys.path.insert(0, str(kit))
    from train_baseline import build_yolo_split, CLASS_NAMES

    data = work / "yolo_data"
    for split in ("train", "val"):
        n = build_yolo_split(kit, data, split)
        print(f"{split}: {n} images prepared")
    (work / "data.yaml").write_text(
        f"path: {data}\ntrain: images/train\nval: images/val\nnames:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASS_NAMES))
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit", required=True, help="competition kit folder")
    ap.add_argument("--work", default="baseline_work")
    ap.add_argument("--weights", default="weights/yolov8n_coco.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=768)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="exp01_768")
    ap.add_argument("--device", default=None, help='e.g. "0,1" for two GPUs, "0" for one')
    a = ap.parse_args()

    work = Path(a.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    prepare_dataset(Path(a.kit).resolve(), work)

    model = YOLO(a.weights)
    model.train(
        data=str(work / "data.yaml"), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
        device=a.device, workers=2, seed=a.seed,
        project=str(work / "runs"), name=a.name, exist_ok=True,
        fliplr=0.5, flipud=0.0, patience=15, plots=True,
    )


if __name__ == "__main__":
    main()
