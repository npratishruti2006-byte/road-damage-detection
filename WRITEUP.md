# Write-up: Road Damage Detection (Anveshan DataSprint 2.0, CV Track)

## 1. Approach

A YOLOv8n object detector (Ultralytics 8.4.216) fine-tuned from COCO-pretrained weights for 50 epochs at 768 px input, batch 16, seed 0, with horizontal flip as the only custom augmentation. Inference uses confidence 0.001, NMS IoU 0.6, and at most 100 detections per image, with boxes in original image pixels. The submitted model is the exp01 run.

## 2. Pretrained weights (every weight used)

| Weight | Source | Use |
|---|---|---|
| `yolov8n.pt` (COCO, 80 classes) | https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt | Starting weights for all training. Included as `weights/yolov8n_coco.pt`. |

No other pretrained weights were used. No weights trained on road, pavement or crack data were used, including any public road-damage model.

## 3. Libraries

- ultralytics 8.4.216
- torch 2.11.0+cu130, torchvision 0.26.0+cu130
- numpy 2.1.3, pandas 2.3.3, opencv-python 4.14.0, pillow
- Python 3.13.16

The competition's own `requirements.txt` pins newer torch and numpy versions. We used the preinstalled versions of the Kaggle GPU image, as recorded here, and did not change them.

## 4. Compute resources

- Kaggle notebook environment, 2 × NVIDIA Tesla T4 (15 GB each), CUDA 13.0.
- Training of the submitted model: about 2.26 hours for 50 epochs at 768 px.
- Inference: median about 17 ms per image on one T4 (batch size 1, measured by the competition's `benchmark_speed.py`).

## 5. Data

Only the competition data was used: `train_images` (14,691), `val_images` (2,150) and `test_images` (4,200), with their labels. No outside images, labels or data of any kind were used. The data is derived from RDD2020 and is licensed CC BY-NC 3.0. Data conversion used the kit's own `train_baseline.py`.

## 6. Confirmations

- No outside data was used.
- The only pretrained weights used are the approved COCO weights listed in section 2.
- No weights trained on road-damage data were used.
- No hand labelling, hard-coded outputs, or leaderboard probing were used to produce predictions.
- `inference.py` makes no internet or API calls and loads weights from a path relative to its own file.

## 7. Results (official scorer, validation set, 2,150 images)

| Run | Image size | Seed | Accuracy score |
|---|---|---|---|
| exp00 | 640 | 0 | 41.41 |
| exp01 (submitted) | 768 | 0 | 42.64 |
| exp02 | 768 | 1 | 41.46 |

Hidden test leaderboard accuracy score for the submitted model: [FILL].

## 8. Known gaps (stated plainly)

- The exp01 checkpoint submitted to the leaderboard was lost in a session failure after submission. The repository does not contain it. Reproducing the model requires retraining, which takes about 2.3 hours on 2 × T4.
- `src/train.py` was rewritten after the experiments to build the dataset itself. Its arguments reproduce exp01, but the final file was not executed end to end.
- Validation was used for checkpoint selection, so validation scores are optimistic.
- Seed-to-seed variation is about 1.2 points on the validation score.
