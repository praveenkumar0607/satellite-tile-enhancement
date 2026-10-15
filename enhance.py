import cv2
import numpy as np
import argparse
import os
import urllib.request
from pathlib import Path

def download_model(model_dir="models", model_name="FSRCNN_x2.pb"):
    # grab the pretrained FSRCNN weights from github if we don't have them yet
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, model_name)
    if not os.path.exists(model_path):
        print(f"Downloading {model_name}...")
        url = "https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x2.pb"
        urllib.request.urlretrieve(url, model_path)
        print("Download complete.")
    return model_path

def enhance_classical(img):
    # step 1 — denoise
    # using fastNlMeans because it handles JPEG block noise well without wiping out edges
    denoised = cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21)

    # step 2 — boost contrast without messing up colors
    # plain histogram equalization can make tiles look weird/oversaturated,
    # so convert to LAB and only touch the L (lightness) channel with CLAHE
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    contrast = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # step 3 — sharpen using unsharp masking
    # subtract a blurred version from the original to pull out edges
    gaussian = cv2.GaussianBlur(contrast, (0, 0), 2.0)
    sharpened = cv2.addWeighted(contrast, 1.5, gaussian, -0.5, 0)

    return sharpened

def enhance_sr(img, model_path):
    # run the FSRCNN super-resolution model (2x upscale)
    # works entirely on CPU, no GPU needed
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(model_path)
    sr.setModel("fsrcnn", 2)
    result = sr.upsample(img)
    return result

def main():
    parser = argparse.ArgumentParser(description="Enhance Satellite Tiles")
    parser.add_argument("--input", required=True, help="Input directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--method", choices=['classical', 'sr', 'bicubic'], required=True)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # only download the model if we're actually using SR
    model_path = download_model() if args.method == 'sr' else None

    for file_name in os.listdir(args.input):
        if not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img_path = os.path.join(args.input, file_name)
        img = cv2.imread(img_path)
        if img is None:
            # skip anything that cv2 can't open (corrupt file, etc.)
            continue

        if args.method == 'classical':
            res = enhance_classical(img)
        elif args.method == 'sr':
            res = enhance_sr(img, model_path)
        elif args.method == 'bicubic':
            # simple 2x upscale — used as the baseline to compare against
            h, w = img.shape[:2]
            res = cv2.resize(img, (w*2, h*2), interpolation=cv2.INTER_CUBIC)

        out_path = os.path.join(args.output, f"{Path(file_name).stem}_{args.method}.jpg")
        cv2.imwrite(out_path, res)
        print(f"Processed {file_name} -> {out_path}")

if __name__ == "__main__":
    main()