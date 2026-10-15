# Satellite Tile Enhancement

This project takes 640×640 satellite tiles from the Google Maps Static API and makes them look better — either using classic image processing tricks or a pretrained deep learning super-resolution model (FSRCNN). There's also an evaluation script that runs all three methods and spits out PSNR/SSIM scores so you can actually compare them.

## What you need

- Python 3.9 or above
- No GPU needed — everything runs on CPU

## Install

```bash
pip install -r requirements.txt
```

## How to run it

### Classical enhancement (denoise + contrast + sharpen)

This runs a 3-step pipeline: first reduces noise, then boosts local contrast without destroying colors, then sharpens edges.

```bash
python enhance.py --input data/ --output out/ --method classical
```

### Super-Resolution — FSRCNN ×2

Upscales each tile 2x using the FSRCNN model. The model file is downloaded automatically the first time you run it.

```bash
python enhance.py --input data/ --output out/ --method sr
```

### Bicubic baseline

Just a plain 2x resize using bicubic interpolation. Useful as a reference to see how much the other methods actually improve things.

```bash
python enhance.py --input data/ --output out/ --method bicubic
```

### Run the evaluation

Compares all three methods on your data folder and saves a markdown report with per-image PSNR and SSIM scores.

```bash
python evaluate.py --input data/ --report results.md
```

### Visual comparison grid (optional)

Makes a side-by-side image showing original vs. all enhanced versions for one tile. Change line 7 in `generate-grid.py` to pick which image to use.

```bash
python generate-grid.py
```

### Failure case crops (optional)

Saves 200×200 crops of the worst classical enhancement failures and links them in `WRITEUP.md`.

```bash
python generate_crops.py
```

## Output files

| File | What it is |
|---|---|
| `out/*_classical.jpg` | Tiles after classical enhancement |
| `out/*_sr.jpg` | Tiles after SR upscaling (1280×1280) |
| `out/*_bicubic.jpg` | Tiles after plain bicubic resize |
| `results.md` | PSNR/SSIM scores for all tiles across all methods |
| `visual_grid.jpg` | Before/after visual comparison for one tile |
| `crops/` | Zoomed-in crops showing where classical enhancement fails |
| `WRITEUP.md` | My explanation of why I made each decision |

## Note

No Google Maps API key needed. The scripts run on whatever images are already in the `/data` folder.
