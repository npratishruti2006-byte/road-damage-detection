# Intelligent Road Damage Detection (Anveshan DataSprint 2.0)

A YOLOv8n object detector, fine-tuned from COCO-pretrained weights, that finds and classifies four road damage types in photos. The submitted model is **exp01**: 768 px input, 50 epochs, seed 0.

| Item | Value |
|---|---|
| Official validation score (accuracy score, 2,150 images) | **42.64 / 100** |
| Hidden test leaderboard score (submitted model) | **[FILL: exact leaderboard value]** |
| Submitted weights | `weights/best.pt` (exp01), sha256 **[FILL]** |
| Submission file | `submission/submission.csv` (4,200 rows) |

## 1. Task and classes

| Class ID | Code | Name |
|---|---|---|
| 0 | D00 | longitudinal crack |
| 1 | D10 | transverse crack |
| 2 | D20 | alligator crack |
| 3 | D40 | pothole |

## 2. Repository contents

```
.
├── README.md
├── requirements.txt
├── inference.py                 # competition starter file, with IMGSZ = 768
├── src/
│   ├── train.py                 # training script
│   └── backup.py                # copies run outputs and verifies checksums
├── results/
│   └── experiment_log.csv       # official scores for every experiment
└── submission/
    └── submission.csv           # the file uploaded to the competition
```

The competition images, label CSVs, and scoring scripts are not included. They belong to the organizers. Download them from the competition page.

## 3. Environment

Training and evaluation ran on Kaggle notebooks:

- Hardware: 2 × NVIDIA Tesla T4 (15 GB each), driver 580.178.04, CUDA 13.0
- Python 3.13.16
- torch 2.11.0+cu130, torchvision 0.26.0+cu130
- ultralytics 8.4.216, numpy 2.1.3, pandas 2.3.3, opencv 4.14.0

The competition's `requirements.txt` pins torch 2.14.1, numpy 2.4.6, and pillow 11.3.0. We did not install those pins. We used the torch build preinstalled on Kaggle and installed only `ultralytics==8.4.216`. Scoring and benchmarking used the unmodified competition scripts.

## 4. Data

- **Splits:** train 14,691 images, validation 2,150, test 4,200. All images are `.jpg`.
- **Labels:** CSV rows `image_id,class_id,x_min,y_min,x_max,y_max` in original image pixels.
- **Train:** 17,517 boxes on 8,523 images. Per class: D00 4,600, D10 3,132, D20 5,864, D40 3,921. Images with no damage (6,168) are kept as background.
- **Validation:** 2,561 boxes on 1,254 images. Per class: D00 695, D10 454, D20 824, D40 588. Background images: 896.
- **Conversion to YOLO format:** the competition's `train_baseline.py` provides `build_yolo_split`, which writes normalized YOLO labels and symlinks the images. We used it unchanged, then wrote `data.yaml` with the four class names.
- **Checks run before training:** file counts match the README; no label IDs without a matching image; no inverted boxes; no boxes outside image bounds in a 300-image sample per split. The out-of-bounds check was a sample, not the full set.

## 5. Submitted model (exp01)

- Initialization: `yolov8n.pt` (COCO), approved by the README
- Epochs: 50 (2.26 hours on 2 × T4)
- Training image size: 768. Inference image size: 768 (`IMGSZ = 768` in `inference.py`)
- Batch 16, seed 0, `fliplr=0.5`, `flipud=0.0`, patience 15, AMP on
- `best.pt` is Ultralytics' best epoch on the validation split (fitness weighted toward mAP50-95)

Verify the weights file before running anything:

```
sha256sum weights/best.pt          # Linux/macOS
certutil -hashfile weights/best.pt SHA256   # Windows
```

The output must match: **[FILL: sha256 from the exp01 manifest]**

## 6. Pretrained weights

- Source: `https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt`
- Check: loading it gives 80 COCO classes, starting with `person`, `bicycle`, `car`
- No road, pavement, or crack data was used in any starting point for the results reported here. Earlier experiments that started from weights trained on this competition's data are excluded.

## 7. Training

Run from the Python API (equivalent to `src/train.py`):

```python
from ultralytics import YOLO

model = YOLO("weights/yolov8n_coco.pt")
model.train(
    data="baseline_work/data.yaml", epochs=50, imgsz=768, batch=16,
    device=[0, 1], workers=2, seed=0,
    project="baseline_work/runs", name="exp01_768", exist_ok=True,
    fliplr=0.5, flipud=0.0, patience=15, plots=True,
)
```

Ultralytics defaults used for everything else: optimizer auto (resolved to MuSGD, lr0 0.01), mosaic 1.0, close_mosaic 10, warmup 3 epochs, weight decay 0.0005, randaugment, erasing 0.4, HSV augmentation.

Note: the defaults in `src/train.py` are for the 80-epoch run that did not finish. Pass `--epochs 50 --name exp01_768` to reproduce exp01.

## 8. Inference and submission

- `inference.py` is the competition starter with `IMGSZ = 768`. It loads `weights/best.pt` relative to its own location.
- Settings: confidence 0.001, NMS IoU 0.6, max 100 detections per image, boxes in original image pixels.

Commands:

```
python benchmark_speed.py --repo . --images <val_images> --out-csv val_pred.csv --out-json val_speed.json
python score.py --pred val_pred.csv --gt <val_labels.csv> --image-ids <val_image_ids.csv> --speed-json val_speed.json
python benchmark_speed.py --repo . --images <test_images> --out-csv test_pred.csv --out-json test_speed.json
python make_kaggle_submission.py --pred test_pred.csv --image-ids <test_image_ids.csv> --out submission.csv
python check_submission.py --pred submission.csv --image-ids <test_image_ids.csv>
```

Output for the submitted file: 4,200 images, 125,055 boxes, 4,051 images with at least one detection, and 149 images written as `none`, which the format allows. The checker passed. Median inference time was 16.9 ms per image on a T4, under the 25 ms threshold for full speed credit.

## 9. Evaluation results (official scorer, validation set)

| Run | Image size | Seed | mAP@50 | mAP@50:95 | Precision | Recall | Accuracy score |
|---|---|---|---|---|---|---|---|
| exp00 (baseline) | 640 | 0 | 0.4906 | 0.2098 | 0.5265 | 0.4892 | 41.41 |
| exp01 (submitted) | 768 | 0 | 0.5036 | 0.2167 | 0.5536 | 0.4943 | **42.64** |
| exp02 | 768 | 1 | 0.4907 | 0.2117 | 0.5228 | 0.4924 | 41.46 |

Per class for exp01 (AP@50 / AP@50:95): D00 0.4595 / 0.1895; D10 0.4350 / 0.1563; D20 0.6698 / 0.3353; D40 0.4502 / 0.1859.

The accuracy score is computed by `score.py` as (0.30·mAP50 + 0.25·mAP50-95 + 0.15·precision + 0.15·recall) / 0.85 × 100. It does not include the speed term or the robustness term. Speed was 1.0 for all runs. The robustness term needs the degraded test images and was not computed locally.

Precision and recall are read at one confidence threshold chosen by `score.py`: 0.198 for exp01 and 0.167 for exp02.

## 10. Experiment history

- **exp00:** baseline at 640 px. 41.41.
- **exp01:** 768 px. 42.64. Submitted.
- **exp02:** same settings as exp01, seed 1. 41.46. The exp01 gain did not reproduce. Seed-to-seed variation is about 1.2 points, so differences smaller than that should not be read as real.
- **exp03:** 80 epochs at 768 px, started but interrupted by a power and network failure before finishing. It has no official score and is not reported. Its training-log numbers are not comparable to the official scores.

## 11. Reproduction

Expected result: an official validation accuracy score near 42.6, within about ±1.2 points because of seed and hardware nondeterminism.

1. Install: `pip install ultralytics==8.4.216`, with torch 2.11 or a compatible CUDA build.
2. Place the competition files in `<KIT>`.
3. Download `yolov8n.pt` into `weights/yolov8n_coco.pt` and confirm 80 classes.
4. Build the YOLO dataset:
   ```python
   import sys; sys.path.insert(0, "<KIT>")
   from train_baseline import build_yolo_split, CLASS_NAMES
   # call build_yolo_split(KIT, DATA, "train") and (KIT, DATA, "val"), then write data.yaml
   ```
5. Train with the settings in section 7.
6. Copy `runs/exp01_768/weights/best.pt` to `weights/best.pt`. Set `IMGSZ = 768` in `inference.py`.
7. Run the validation commands and the official scorer.
8. Run the test commands to regenerate `submission.csv`.

## 12. Limitations

- **Model selection used the validation split.** `best.pt` was chosen on that split, so the validation score is optimistic. The official test score is the fair measure.
- **Test-set gap.** The test set includes deliberately degraded photos, which the validation set does not. Our test score (40) is below our validation score (42.6). We believe the degraded images are the main cause, but we did not measure this.
- **Seed variance.** About 1.2 points. Small changes between experiments are not reliable evidence.
- **Only two inference settings were used.** Confidence 0.001 and IoU 0.6. We did not tune them.
- **Speed.** Measured on Kaggle T4 GPUs at batch size 1. The organizers' hardware may differ.
- **Trained weights.** The submitted weights are not included in this repository. They were lost in a session failure after the submission was made. Reproduction requires retraining (section 11). The official score of 42.64 was measured on the model before the loss.

## 13. Links

- Kaggle notebook: **[FILL: notebook URL, or "available on request"]**
- Training log: `results/experiment_log.csv`

## 14. Data attribution and licence

The images come from the RDD2020 dataset, licensed CC BY-NC 3.0 (https://creativecommons.org/licenses/by-nc/3.0/). Use is limited to this competition and non-commercial work.

Arya, D., Maeda, H., Ghosh, S. K., Toshniwal, D., Omata, H., Kashiyama, T., Seto, T., Mraz, A., and Sekimoto, Y. (2021). "RDD2020: An Image Dataset for Smartphone-based Road Damage Detection and Classification." Mendeley Data, V1, doi: 10.17632/5ty2wb6gvg.1.

## 15. Author

**[FILL: name, team name, email]**
