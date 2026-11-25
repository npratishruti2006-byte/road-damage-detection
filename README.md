# Intelligent Road Damage Detection (Anveshan DataSprint 2.0, CV Track)

Object detector for four road damage classes: D00 longitudinal crack, D10 transverse crack, D20 alligator crack, D40 pothole. Built on YOLOv8n with COCO-pretrained starting weights.

| Item | Value |
|---|---|
| Team / author | [FILL: N Pratishruti Das, (N Pratishruti Das, Ommjeet Panda] |
| Submitted model | exp01: YOLOv8n, 768 px, 50 epochs, seed 0 |
| Official validation accuracy score (score.py, 2,150 images) | 42.64 / 100 |
| Hidden test leaderboard accuracy score | [FILL: 40.95320] |
| Pinned commit for verification | [FILL: commit hash] |
| Submitted weights | Not included. See "Known issues". |

## Known issues (read first)

1. **The submitted checkpoint is not in this repository.** `weights/best.pt` from exp01 was lost in a Kaggle session failure after the submission was made. `inference.py` loads that file, so the inference reproduction in section 9 cannot run as shipped until the weights are supplied.
2. **Retraining does not fit the 30-minute verification window.** Training takes about 2.3 hours on 2 × T4. Reproducing the model from scratch is documented in section 7 for transparency, but it is not part of the verification run.
3. **Environment deviation.** The competition's `requirements.txt` pins torch 2.14.1, numpy 2.4.6 and pillow 11.3.0. Our results were produced with the Kaggle image's preinstalled torch 2.11.0 (CUDA 13.0), numpy 2.1.3, pandas 2.3.3 and opencv 4.14.0. We installed only `ultralytics==8.4.216` on top of that image.
4. **Training script not re-run in final form.** `src/train.py` was rewritten after the experiments to build the dataset itself. Its arguments reproduce exp01, but the file has not been executed end to end.

## 2. Repository contents

```
.
├── README.md                      this file
├── WRITEUP.md                     pretrained weights, libraries, compute, data confirmations
├── requirements.txt               pinned Ultralytics version and environment notes
├── inference.py                   competition inference file, IMGSZ = 768
├── weights/
│   └── yolov8n_coco.pt            approved COCO starting weights (80 classes)
├── src/
│   ├── train.py                   dataset build and training (exp01 settings by default)
│   └── backup.py                  checkpoint backup with SHA-256 verification
├── notebook/
│   └── pipeline_cells.py          Kaggle notebook cells in the order they were run
├── results/
│   └── experiment_log.csv         official scores for every reported experiment
└── submission/
    └── submission.csv             file uploaded to the competition (see section 8)
```

## 3. Environment

Hardware used for all results: Kaggle notebook, 2 × NVIDIA Tesla T4 (15 GB each), driver 580.178.04, CUDA 13.0.

| Component | Version used |
|---|---|
| Python | 3.13.16 |
| torch | 2.11.0+cu130 |
| torchvision | 0.26.0+cu130 |
| ultralytics | 8.4.216 |
| numpy | 2.1.3 |
| pandas | 2.3.3 |
| opencv-python | 4.14.0 |

Install: create a Python 3.13 environment, install torch 2.11.0 built for CUDA 13.0, then run `pip install -r requirements.txt`.

## 4. Data

Use only the competition data, obtained from the competition kit. The kit provides `train_images/`, `train_labels.csv`, `val_images/`, `val_labels.csv`, `val_image_ids.csv`, `test_images/`, `test_image_ids.csv`, and the scripts `score.py`, `benchmark_speed.py`, `make_kaggle_submission.py`, `check_submission.py`, `train_baseline.py`, plus the starter `inference.py`.

- Train: 14,691 images, 17,517 boxes (D00 4,600; D10 3,132; D20 5,864; D40 3,921). 6,168 background images with no boxes are kept.
- Validation: 2,150 images, 2,561 boxes (D00 695; D10 454; D20 824; D40 588). 896 background images.
- Test: 4,200 images, no labels.

Checks run before training: file counts match the kit's README; every label ID has an image; no inverted boxes; no out-of-bounds boxes in a 300-image sample per split (the sample check was not a full pass).

## 5. Method

- Starting weights: `yolov8n_coco.pt` (Ultralytics YOLOv8n trained on COCO), the only pretrained weights used.
- Training settings for exp01: 50 epochs, image size 768, batch 16, seed 0, horizontal flip only, `flipud=0.0`, patience 15, plots on, 2 GPUs. All other Ultralytics defaults (optimizer auto, mosaic 1.0, close_mosaic 10, randaugment, erasing 0.4, HSV augmentation).
- Dataset conversion: the kit's own `build_yolo_split` function, unchanged.
- Checkpoint used for the submission: Ultralytics `best.pt` (best epoch by validation fitness).

## 6. Inference

`inference.py` is the competition starter with `IMGSZ = 768`. It loads `weights/best.pt` relative to its own location, runs at confidence 0.001, NMS IoU 0.6, and at most 100 detections per image, and returns boxes in original image pixels. It makes no internet or API calls.

## 7. Reproduce the training (about 2.3 hours on 2 × T4)

From the repository root, with the competition kit at `KIT`:

```bash
pip install -r requirements.txt
python src/train.py --kit "$KIT" --work baseline_work --weights weights/yolov8n_coco.pt --device 0,1
python src/backup.py   # optional: see the backup_experiment() function
```

Expected output: `train: 14691 images prepared`, `val: 2150 images prepared`, and a training run of 50 epochs at 768 px. The training log's final validation numbers should be close to mAP50 0.495 and mAP50-95 0.223 (Ultralytics metrics on the validation split). Exact values vary with hardware and seed.

To score a trained model on validation, after copying its `best.pt` to `weights/best.pt`:

```bash
python "$KIT/benchmark_speed.py" --repo . --images "$KIT/val_images" --out-csv val_pred.csv --out-json val_speed.json
python "$KIT/score.py" --pred val_pred.csv --gt "$KIT/val_labels.csv" --image-ids "$KIT/val_image_ids.csv" --speed-json val_speed.json
```

## 8. Reproduce the test submission (inference only, needs `weights/best.pt`)

```bash
python "$KIT/benchmark_speed.py" --repo . --images "$KIT/test_images" --out-csv test_pred.csv --out-json test_speed.json
python "$KIT/make_kaggle_submission.py" --pred test_pred.csv --image-ids "$KIT/test_image_ids.csv" --out submission.csv
python "$KIT/check_submission.py" --pred submission.csv --image-ids "$KIT/test_image_ids.csv"
```

Expected: 4,200 images, about 125,000 boxes, 149 images written as `none`, and the checker passes. The uploaded `submission/submission.csv` is the file produced this way from exp01. Do not regenerate it from a different model.

## 9. Results (official scorer, validation set)

| Run | Image size | Seed | mAP@50 | mAP@50:95 | Precision | Recall | Accuracy score |
|---|---|---|---|---|---|---|---|
| exp00 (baseline) | 640 | 0 | 0.4906 | 0.2098 | 0.5265 | 0.4892 | 41.41 |
| exp01 (submitted) | 768 | 0 | 0.5036 | 0.2167 | 0.5536 | 0.4943 | **42.64** |
| exp02 | 768 | 1 | 0.4907 | 0.2117 | 0.5228 | 0.4924 | 41.46 |

exp01 per class (AP@50 / AP@50:95): D00 0.4595 / 0.1895; D10 0.4350 / 0.1563; D20 0.6698 / 0.3353; D40 0.4502 / 0.1859.

Accuracy score = 100 × (0.30·mAP@50 + 0.25·mAP@50:95 + 0.15·precision + 0.15·recall) / 0.85. It excludes the speed and robustness terms. Speed score was 1.0 for all runs (validation median 13.5 ms per image for exp01; test median 16.9 ms per image on one T4).

Hidden test leaderboard accuracy score for the submission: [FILL].

## 10. Limitations

- **Validation-based checkpoint selection.** `best.pt` was chosen on the validation split, so the validation score is optimistic. The hidden test score is the fair measure.
- **Test-set gap.** Our test score is below our validation score. The test set includes degraded photos that the validation set does not. We believe this is the main cause, but did not measure it.
- **Seed variance.** exp01 and exp02 differ only by seed and score 42.64 and 41.46. Differences smaller than about 1.2 points should not be read as real.
- **Limited tuning.** Only two inference settings were used (confidence 0.001, IoU 0.6). We did not tune them.
- **Robustness not measured.** The robustness term needs the organisers' degraded photos. We did not evaluate it locally.
- **Speed.** Measured on Kaggle T4 GPUs at batch size 1. Organisers' hardware may differ.

## 11. Data licence and credit

The images come from RDD2020, licensed CC BY-NC 3.0 (https://creativecommons.org/licenses/by-nc/3.0/). Use is limited to this competition and non-commercial work.

Arya, D., Maeda, H., Ghosh, S. K., Toshniwal, D., Omata, H., Kashiyama, T., Seto, T., Mraz, A., and Sekimoto, Y. (2021). "RDD2020: An Image Dataset for Smartphone-based Road Damage Detection and Classification." Mendeley Data, V1, doi: 10.17632/5ty2wb6gvg.1.

The competition data, labels and scoring scripts are not redistributed in this repository.
