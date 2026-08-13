<div align="center">
<img src="assets/banner.svg" alt="ReefSight banner" width="800"/>

# ReefSight

**Fine-tuned Faster R-CNN that finds and names aquarium fish, species by species.**

[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

</div>

## What it does

Point ReefSight at a photo of an aquarium and it draws a box around every fish it can find, then
labels each one with a species name and a confidence score. It's an object detector, not just a
classifier: a single photo can contain several fish of several species, and ReefSight locates
and identifies each one independently.

<div align="center">
<img src="https://img.shields.io/badge/ClownFish-0.97-brightgreen" alt="example prediction"/>
</div>

Under the hood, it fine-tunes `fasterrcnn_mobilenet_v3_large_320_fpn` — a lightweight
torchvision detector pretrained on COCO — on 13 aquarium species. Out of the box, COCO's 91
categories don't include a single fish; ReefSight's fine-tuning teaches the model to recognize
these 13 without needing a GPU.

**Species recognized:** AngelFish, BlueTang, ButterflyFish, ClownFish, GoldFish, Gourami,
MoorishIdol, PlatyFish, RibbonedSweetlips, ThreeStripedDamselfish, YellowCichlid, YellowTang,
ZebraFish.

## Why this approach

| Step | What happens |
|---|---|
| 1. Baseline | Load `fasterrcnn_mobilenet_v3_large_320_fpn` pretrained on COCO, and check how it handles fish photos with zero fish-specific training |
| 2. Data pipeline | Wrap the YOLO-format annotations in a PyTorch `Dataset` / `DataLoader` pair |
| 3. Fine-tuning | Freeze the pretrained backbone, replace only the box-classification head, and train it on the 13 target species |
| 4. Inference | Run the fine-tuned detector on new photos of any resolution, including raw phone photos |

The pretrained backbone already knows how to localize "an object" in a photo from its COCO
training — fine-tuning only teaches it *what kind* of object it's looking at, which is why a few
minutes of CPU training is enough to go from "sees a bird" to "sees a ClownFish."

## Project structure

```
reefsight/
├── notebooks/
│   ├── reefsight_finetuning.ipynb   # full pipeline: data, baseline, training, inference
│   └── utils.py                      # dataset loading (YOLO format), evaluation, plotting
├── assets/
│   └── banner.svg
├── pyproject.toml
├── requirements.txt
└── data/                             # created locally by kagglehub, not tracked in git
```

## Getting started

### Prerequisites

- Python 3.11+
- A free [Kaggle](https://www.kaggle.com/) account (the dataset downloads via `kagglehub`, which
  will prompt for a Kaggle API token on first run — see
  [Kaggle's API docs](https://www.kaggle.com/docs/api) for how to generate one)
- No GPU required — the notebook is built to fine-tune on CPU in a few minutes

### Installation

With [uv](https://docs.astral.sh/uv/) (recommended):

```bash
uv sync
```

Or with pip:

```bash
python -m venv .venv
source .venv/bin/activate  # .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Running the notebook

```bash
jupyter lab notebooks/reefsight_finetuning.ipynb
```

Run the cells top to bottom. The first code cell downloads the
[`fish-dataset`](https://www.kaggle.com/datasets/mahmoodyousaf/fish-dataset) (~350 MB, cached
locally after the first run). Training 1,000 photos for 3 epochs takes roughly 5-10 minutes on a
recent laptop CPU; both numbers are exposed as constants (`MAX_TRAIN_IMAGES`, `NUM_EPOCHS`) if you
want to scale up, ideally on a machine with a GPU.

### Trying it on your own photos

Once the model is fine-tuned, drop a photo into the project and call:

```python
boxes, labels, scores = detect_and_label_fish("my_photo.jpg")
```

Any resolution works — the detector resizes internally and maps the predicted boxes back to your
original image. Keep in mind the model only recognizes the 13 trained species: on anything else,
it will still draw a box, just with the closest-looking label among the 13.

## Results & limitations

- The pretrained COCO baseline detects *something* in most fish photos (the Region Proposal
  Network generalizes well) but reliably mislabels it, since COCO has no fish category.
- After fine-tuning on ~1,000 photos for 3 epochs, the total training loss drops consistently
  across epochs, and the model correctly names common species like ClownFish and GoldFish on
  held-out photos.
- Known limitations: no data augmentation, a capped training set for speed, a naturally
  imbalanced dataset (some species have ~10x more photos than others), and no "unknown species"
  output — the model always picks its closest match among the 13.
- `utils.compute_map` computes mAP (mean Average Precision), the standard detection metric, for a
  more rigorous evaluation than eyeballing predictions.

## Roadmap / ideas for going further

- [ ] Train on the full ~6,800-photo dataset with more epochs on a GPU, and compare mAP
- [ ] Try `fasterrcnn_resnet50_fpn` (heavier, often more accurate) as a comparison point
- [ ] Add box-aware data augmentation (`RandomHorizontalFlip`, `RandomPhotometricDistort`,
      `RandomZoomOut` from `torchvision.transforms.v2`)
- [ ] Compare against a YOLO baseline trained directly on the dataset's native YOLO annotations
      with `ultralytics`
- [ ] Wrap inference in a small Streamlit app with live camera input

## Tech stack

- [PyTorch](https://pytorch.org/) / [torchvision](https://pytorch.org/vision/stable/index.html) —
  model and training loop
- [torchmetrics](https://lightning.ai/docs/torchmetrics/stable/) — mAP evaluation
- [kagglehub](https://github.com/Kaggle/kagglehub) — dataset download and caching
- [pandas](https://pandas.pydata.org/) / [matplotlib](https://matplotlib.org/) — data wrangling
  and visualization

## Credits

- Dataset: [`fish-dataset`](https://www.kaggle.com/datasets/mahmoodyousaf/fish-dataset) by
  Mahmood Yousaf on Kaggle.
- Base model: `fasterrcnn_mobilenet_v3_large_320_fpn`, part of
  [torchvision's detection models](https://pytorch.org/vision/stable/models.html#object-detection),
  pretrained on [COCO](https://cocodataset.org/).

## License

Released under the [MIT License](LICENSE). The `fish-dataset` used for training has its own
license on Kaggle — check the dataset page before any use beyond this project.
