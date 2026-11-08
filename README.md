# python_pets

A small Python web app that detects cats and dogs in a photo, from an uploaded or pasted image or straight from your camera.

The model is a Faster R-CNN detector (MobileNetV3-Large FPN backbone, COCO-pretrained), fine-tuned on the
Kaggle dataset [andrewmvd/dog-and-cat-detection](https://www.kaggle.com/datasets/andrewmvd/dog-and-cat-detection).
Because it is a detector rather than a classifier, it draws a box around each animal, handles several animals in
one image, and says "no cat or dog" when it finds neither.

## Project layout

```
src/python_pets/
  config.py      paths, class labels, .env settings
  model.py       builds and loads the detector
  download.py    finds the dataset or fetches it from DATASET_URL
  dataset.py     Pascal-VOC parsing and the PyTorch dataset
  train.py       training loop
  inference.py   runs the detector on one image
  app.py         FastAPI server
  static/        web UI
tests/           unit tests
datasets/        data files (not committed)
  raw/           original images
  processed/     derived data (unused for now)
  annotations/   Pascal-VOC .xml labels
weights/         trained models (not committed)
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (macOS/Linux: source .venv/bin/activate)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # NVIDIA GPU
pip install -e .
copy .env.example .env            # Windows  (macOS/Linux: cp .env.example .env)
```

## Train

```bash
pets-train              # trains and saves weights/catdog_detector.pt
pets-train --epochs 5   # shorter run
pets-train --data D:\datasets\catdog   # use a specific dataset folder
pets-train --resume     # continue a run you stopped with Ctrl+C
```

You can stop training at any time with **Ctrl+C**. The best model so far stays in `weights/catdog_detector.pt`.
A checkpoint is saved after every finished epoch, so `--resume` continues from the last completed epoch.

Training uses the `--data` folder if you pass one, otherwise `datasets/raw/` and `datasets/annotations/`.
If those are empty, it fetches the dataset into them from `DATASET_URL` in `.env`.

`DATASET_URL` can be a Kaggle dataset page, or a direct link to a `.zip` or `.tar` archive that contains
`images/` and `annotations/` folders with Pascal-VOC `.xml` labels.

## Run the app

```bash
pets-serve
```

Then open http://127.0.0.1:8010. To use a different port, set `PORT` in `.env`.

- **Upload / Paste**: click the drop zone, drag a file in, or press **Ctrl+V** anywhere on the page.
- **Camera**: take a single snapshot or turn on **Live detection**. You can switch between front and back cameras on phones.
- **Min confidence**: a slider that hides detections below the threshold you set.

If `weights/catdog_detector.pt` doesn't exist yet, the app falls back to the stock COCO model filtered to cats and dogs.
That model already works, but the fine-tuned one is more accurate.

> Browsers only allow camera access on `localhost` or over `https`.

## Tests

```bash
python -m unittest discover tests
```

## API

`POST /api/predict?threshold=0.5` with a multipart `file` field returns:

```json
{"verdict": "cat", "counts": {"cat": 1, "dog": 0},
 "detections": [{"label": "cat", "score": 0.98, "box": [x1, y1, x2, y2]}],
 "width": 640, "height": 480, "mode": "trained"}
```
