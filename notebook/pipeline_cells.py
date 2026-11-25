# %% Install the pinned Ultralytics version
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.216"], check=True)
from ultralytics import YOLO
import ultralytics
print("ultralytics:", ultralytics.__version__)

# %% Restore the COCO-pretrained starting weights (approved by the README)
import shutil
from pathlib import Path

dst = Path("/kaggle/working/weights/yolov8n_coco.pt")
dst.parent.mkdir(parents=True, exist_ok=True)
if not dst.exists():
    YOLO("yolov8n.pt")  # downloads the official COCO checkpoint
    shutil.copy(Path("yolov8n.pt"), dst)
m = YOLO(str(dst))
print("classes:", len(m.names), "| starts with:", list(m.names.values())[:3])

# %% Check image counts and label files
import csv
from collections import Counter

KIT = Path("/kaggle/input/competitions/datasprint-2-road-damage-detection")
expected = {"train": 14691, "val": 2150, "test": 4200}
for split, n in expected.items():
    files = [p for p in (KIT / f"{split}_images").iterdir() if p.is_file()]
    print(f"{split}: files={len(files)} expected={n} extensions={dict(Counter(p.suffix.lower() for p in files))}")
for split in ["train", "val"]:
    with open(KIT / f"{split}_labels.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    print(f"{split} box rows: {len(rows)} | images with boxes: {len({r['image_id'] for r in rows})}")
    print("  class counts:", sorted(Counter(int(r["class_id"]) for r in rows).items()))

# %% Check that labels match images and boxes are valid
from PIL import Image

for split in ["train", "val"]:
    stems = {p.stem for p in (KIT / f"{split}_images").glob("*.jpg")}
    with open(KIT / f"{split}_labels.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    orphan_ids = {r["image_id"] for r in rows} - stems
    inverted = [r for r in rows if float(r["x_max"]) <= float(r["x_min"]) or float(r["y_max"]) <= float(r["y_min"])]
    sample_ids = sorted({r["image_id"] for r in rows})[:300]
    by_id = {}
    for r in rows:
        by_id.setdefault(r["image_id"], []).append(r)
    out_of_bounds = 0
    for iid in sample_ids:
        W, H = Image.open(KIT / f"{split}_images" / f"{iid}.jpg").size
        for r in by_id[iid]:
            if float(r["x_max"]) > W or float(r["y_max"]) > H or float(r["x_min"]) < 0 or float(r["y_min"]) < 0:
                out_of_bounds += 1
    print(f"{split}: orphan ids={len(orphan_ids)} | inverted={len(inverted)} | "
          f"out-of-bounds in first {len(sample_ids)} labelled images={out_of_bounds}")

# %% Build the YOLO dataset and data.yaml (uses the competition's own converter)
import sys
sys.path.insert(0, str(KIT))
from train_baseline import build_yolo_split, CLASS_NAMES

WORK = Path("/kaggle/working/baseline_work").resolve()
DATA = WORK / "yolo_data"
for s in ("train", "val"):
    print(f"{s}: {build_yolo_split(KIT, DATA, s)} images prepared")
(WORK / "data.yaml").write_text(
    f"path: {DATA}\ntrain: images/train\nval: images/val\nnames:\n"
    + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASS_NAMES))
)

# %% Exp00: baseline, 640 px, 50 epochs
from ultralytics import YOLO

model = YOLO("/kaggle/working/weights/yolov8n_coco.pt")
model.train(
    data=str(WORK / "data.yaml"), epochs=50, imgsz=640, batch=16,
    device=[0, 1], workers=2,
    project=str(WORK / "runs"), name="baseline", exist_ok=True,
    fliplr=0.5, flipud=0.0, patience=15, plots=True,
)
run = WORK / "runs" / "baseline"
flat = WORK / "flat_baseline"
flat.mkdir(exist_ok=True)
for f in ["best.pt", "last.pt"]:
    shutil.copy(run / "weights" / f, flat / f)
for f in ["results.csv", "args.yaml"]:
    shutil.copy(run / f, flat / f)
for p in run.glob("*.png"):
    shutil.copy(p, flat / p.name)

# %% Exp01: 768 px, 50 epochs, seed 0 (run in the session; reconstructed from logged arguments)
from ultralytics import YOLO

WORK = Path("/kaggle/working/baseline_work").resolve()
model = YOLO("/kaggle/working/weights/yolov8n_coco.pt")
model.train(
    data=str(WORK / "data.yaml"), epochs=50, imgsz=768, batch=16,
    device=[0, 1], workers=2,
    project=str(WORK / "runs"), name="exp01_768", exist_ok=True,
    fliplr=0.5, flipud=0.0, patience=15, plots=True,
)

# %% Exp01: copy outputs to a flat folder, then back up and verify
import hashlib, json
from datetime import datetime

run = WORK / "runs" / "exp01_768"
flat = WORK / "flat_exp01_768"
out = Path("/kaggle/working/experiments/exp01_yolov8n_768")
REQUIRED = ["best.pt", "last.pt", "results.csv", "args.yaml"]
flat.mkdir(exist_ok=True)
out.mkdir(parents=True, exist_ok=True)
shutil.copy(run / "weights" / "best.pt", flat / "best.pt")
shutil.copy(run / "weights" / "last.pt", flat / "last.pt")
shutil.copy(run / "results.csv", flat / "results.csv")
shutil.copy(run / "args.yaml", flat / "args.yaml")
for p in run.glob("*.png"):
    shutil.copy(p, flat / p.name)
for p in flat.iterdir():
    shutil.copy(p, out / p.name)

def sha256(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()

config = {"model": "yolov8n", "weights_init": "yolov8n_coco.pt", "epochs": 50, "imgsz": 768,
          "batch": 16, "fliplr": 0.5, "flipud": 0.0, "patience": 15, "devices": [0, 1]}
(out / "config.json").write_text(json.dumps(config, indent=2))
manifest = {"experiment": out.name, "created": datetime.now().isoformat(timespec="seconds"),
            "files": {f: {"bytes": (out / f).stat().st_size, "sha256": sha256(out / f)} for f in REQUIRED}}
(out / "manifest.json").write_text(json.dumps(manifest, indent=2))
print("Verified:", all(sha256(out / f) == manifest["files"][f]["sha256"] for f in REQUIRED))

# %% Exp01: copy the submitted weights into the repo and confirm IMGSZ
import shutil
from pathlib import Path

REPO = Path("/kaggle/working/repo")
exp01 = Path("/kaggle/working/experiments/exp01_yolov8n_768")
shutil.copy(exp01 / "best.pt", REPO / "weights" / "best.pt")
manifest = json.loads((exp01 / "manifest.json").read_text())
print("Repo weights match Exp01 backup:",
      sha256(REPO / "weights" / "best.pt") == manifest["files"]["best.pt"]["sha256"])
print("IMGSZ in repo:", [l for l in (REPO / "inference.py").read_text().splitlines() if l.startswith("IMGSZ")])

# %% Exp01: official validation benchmark and scorer
import subprocess

KIT = Path("/kaggle/input/competitions/datasprint-2-road-damage-detection")
OUT = Path("/kaggle/working/eval")
OUT.mkdir(exist_ok=True)
r = subprocess.run([sys.executable, str(REPO / "benchmark_speed.py"), "--repo", str(REPO),
                    "--images", str(KIT / "val_images"),
                    "--out-csv", str(OUT / "exp01_val_pred.csv"),
                    "--out-json", str(OUT / "exp01_val_speed.json")],
                   capture_output=True, text=True, cwd=str(REPO))
print("benchmark return code:", r.returncode)
r = subprocess.run([sys.executable, str(KIT / "score.py"),
                    "--pred", str(OUT / "exp01_val_pred.csv"),
                    "--gt", str(KIT / "val_labels.csv"),
                    "--image-ids", str(KIT / "val_image_ids.csv"),
                    "--speed-json", str(OUT / "exp01_val_speed.json"),
                    "--json-out", str(OUT / "exp01_val_score.json")],
                   capture_output=True, text=True)
print(r.stdout)

# %% Exp01: test predictions, Kaggle file, and format check (this is the submitted file)
OUT = Path("/kaggle/working/submission")
OUT.mkdir(exist_ok=True)
subprocess.run([sys.executable, str(REPO / "benchmark_speed.py"), "--repo", str(REPO),
                "--images", str(KIT / "test_images"),
                "--out-csv", str(OUT / "test_pred.csv"),
                "--out-json", str(OUT / "test_speed.json")],
               capture_output=True, text=True, cwd=str(REPO))
subprocess.run([sys.executable, str(REPO / "make_kaggle_submission.py"),
                "--pred", str(OUT / "test_pred.csv"),
                "--image-ids", str(KIT / "test_image_ids.csv"),
                "--out", str(OUT / "submission.csv")],
               capture_output=True, text=True, cwd=str(REPO))
r = subprocess.run([sys.executable, str(REPO / "check_submission.py"),
                    "--pred", str(OUT / "submission.csv"),
                    "--image-ids", str(KIT / "test_image_ids.csv")],
                   capture_output=True, text=True, cwd=str(REPO))
print("check_submission return code:", r.returncode, r.stdout)

# %% Exp02: 768 px, seed 1 (seed-variance check)
from ultralytics import YOLO

model = YOLO("/kaggle/working/weights/yolov8n_coco.pt")
model.train(
    data=str(WORK / "data.yaml"), epochs=50, imgsz=768, batch=16,
    device=[0, 1], workers=2, seed=1,
    project=str(WORK / "runs"), name="exp02_768_seed1", exist_ok=True,
    fliplr=0.5, flipud=0.0, patience=15, plots=True,
)

# %% Exp02: back up and copy weights into the repo (IMGSZ must already be 768)
import hashlib, json
from datetime import datetime

run = WORK / "runs" / "exp02_768_seed1"
flat = Path("/kaggle/working/baseline_work/flat_exp02_768_seed1")
out = Path("/kaggle/working/experiments/exp02_yolov8n_768_seed1")
flat.mkdir(exist_ok=True)
out.mkdir(parents=True, exist_ok=True)
shutil.copy(run / "weights" / "best.pt", flat / "best.pt")
shutil.copy(run / "weights" / "last.pt", flat / "last.pt")
shutil.copy(run / "results.csv", flat / "results.csv")
shutil.copy(run / "args.yaml", flat / "args.yaml")
for p in run.glob("*.png"):
    shutil.copy(p, flat / p.name)
for p in flat.iterdir():
    shutil.copy(p, out / p.name)
config = {"model": "yolov8n", "weights_init": "yolov8n_coco.pt", "epochs": 50, "imgsz": 768,
          "batch": 16, "seed": 1, "fliplr": 0.5, "flipud": 0.0, "patience": 15, "devices": [0, 1]}
(out / "config.json").write_text(json.dumps(config, indent=2))
manifest = {"experiment": out.name, "created": datetime.now().isoformat(timespec="seconds"),
            "files": {f: {"bytes": (out / f).stat().st_size, "sha256": sha256(out / f)} for f in REQUIRED}}
(out / "manifest.json").write_text(json.dumps(manifest, indent=2))
print("Backup verified:", all(sha256(out / f) == manifest["files"][f]["sha256"] for f in REQUIRED))
