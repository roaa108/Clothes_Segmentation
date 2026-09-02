# Clothes Segmentation

Segments worn clothes out of photos of people. Every pixel gets one of four classes:

- `0 background` — background, skin, hair, accessories
- `1 upper` — shirts, tops, jackets
- `2 lower` — pants and skirts
- `3 dress` — one-piece dresses

U-Net with an ImageNet-pretrained ResNet34 encoder, trained on the ATR human parsing
dataset. Built as a preprocessing step for a virtual try-on pipeline, not as a full 18-class
human parser.

## Results

Test split, 200 images, best epoch 31:

| Class | IoU | Dice |
|---|---:|---:|
| background | 0.9864 | 0.9931 |
| upper | 0.8692 | 0.9300 |
| lower | 0.9000 | 0.9473 |
| dress | 0.1826 | 0.3089 |
| **garment mean** | **0.6506** | **0.7287** |

`upper` and `lower` are strong. **`dress` fails at 0.18** — the model over-predicts it, and
61% of pixels it labels `dress` are actually `upper`. Full analysis, figures and failure
cases in **[report/report.md](report/report.md)**.

---

## Quickest way to reproduce — Colab

1. Open **[notebooks/colab_train_clothes_segmentation.ipynb](notebooks/colab_train_clothes_segmentation.ipynb)** in Google Colab
2. **Runtime → Change runtime type → GPU**
3. Run all cells top to bottom

It clones this repo, downloads ATR, trains, evaluates, and displays the figures and the
results table. Roughly **25 min** for the dataset download plus **~1 hour** training on an
A100.

---

## Running it locally

### 1. Install

```bash
pip install -r requirements.txt
```

On Windows, if the `stringzilla` wheel fails to build:

```bash
pip install --only-binary=stringzilla -r requirements.txt
```

### 2. Get the dataset

```bash
python download_data.py
```

Writes `data/atr/{train,validation,test}/{images,masks}` as paired PNGs — **~5.7 GB**, about
**25 minutes**. The masks keep the original 18 ATR label ids; `dataset.py` folds them into
the four classes via `label_map`.

### 3. Verify the labels before spending an hour training

```bash
python dataset.py
```

Prints split sizes and the share of each class in one batch. `upper` should be the largest
garment class and `dress` the smallest.

### 4. Train

```bash
python -m experiments.train_unet
```

Settings are at the top of the script: 256×256 input, batch 16, 40 epochs, AdamW at a
constant 1e-4. Under an hour on an A100.

Checkpoints go to `checkpoints/unet/` — `latest.pt` each epoch, `best.pt` on the best
validation garment IoU — with `history.json` alongside.

### 5. Evaluate

```bash
python evaluate.py
```

Reads `checkpoints/unet/best.pt` and writes:

| Output | Contents |
|---|---|
| `results/metrics/final_results.json` | IoU and Dice per class, garment mean, pixel accuracy, confusion matrix |
| `results/metrics/results_table.md` | the same numbers as a markdown table |
| `results/metrics/training_curves.png` | loss and garment IoU per epoch |
| `results/examples/` | mask, overlay and clothing-only cutout for 8 test photos |

### 6. Segment your own photo

```bash
python inference.py path/to/photo.jpg
```

Writes `<name>_mask.png`, `<name>_overlay.png` and `<name>_clothing_only.png` to
`results/inference/`, all at the original input resolution.



## Notes on reproducibility


- The best checkpoint is selected on **validation garment IoU**, not loss.
- `results/` is gitignored because it is regenerated. The figures the report shows are copied
  into `report/figures/` and committed from there.
