import cv2
import os
import argparse
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tabulate import tabulate
from enhance import enhance_classical, enhance_sr, download_model

def degrade_image(img, scale_factor=2, jpeg_quality=75):
    # we don't have a true low-res ground truth, so we fake it:
    # shrink the image by 2x and then re-encode as JPEG to add compression artifacts
    # this is meant to mimic what the Maps API gives us
    h, w = img.shape[:2]
    down = cv2.resize(img, (w // scale_factor, h // scale_factor), interpolation=cv2.INTER_CUBIC)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    _, encimg = cv2.imencode('.jpg', down, encode_param)
    degraded = cv2.imdecode(encimg, 1)
    return degraded

def evaluate(input_dir, report_path):
    results = []
    model_path = download_model()

    for file_name in os.listdir(input_dir):
        if not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img_path = os.path.join(input_dir, file_name)
        gt_img = cv2.imread(img_path)
        if gt_img is None:
            continue

        h, w = gt_img.shape[:2]

        # artificially degrade to 320x320 with JPEG noise
        degraded = degrade_image(gt_img)

        # now reconstruct back to 640x640 using each method
        recon_bicubic = cv2.resize(degraded, (w, h), interpolation=cv2.INTER_CUBIC)
        recon_classical = enhance_classical(recon_bicubic)

        # SR upscales directly from the small image, then we resize to match gt
        recon_sr = enhance_sr(degraded, model_path)
        recon_sr = cv2.resize(recon_sr, (w, h))

        # score each reconstruction against the original 640x640 tile
        psnr_bicubic = psnr(gt_img, recon_bicubic)
        ssim_bicubic = ssim(gt_img, recon_bicubic, channel_axis=-1)

        psnr_classical = psnr(gt_img, recon_classical)
        ssim_classical = ssim(gt_img, recon_classical, channel_axis=-1)

        psnr_sr = psnr(gt_img, recon_sr)
        ssim_sr = ssim(gt_img, recon_sr, channel_axis=-1)

        results.append([
            file_name,
            f"{psnr_bicubic:.2f} / {ssim_bicubic:.3f}",
            f"{psnr_classical:.2f} / {ssim_classical:.3f}",
            f"{psnr_sr:.2f} / {ssim_sr:.3f}"
        ])

    # dump everything into a markdown table
    headers = ["Filename", "Bicubic (PSNR/SSIM)", "Classical (PSNR/SSIM)", "SR (PSNR/SSIM)"]
    md_table = tabulate(results, headers=headers, tablefmt="github")

    with open(report_path, "w") as f:
        f.write("# Evaluation Report\n\n")
        f.write("Metrics format: PSNR (higher is better) / SSIM (closer to 1.0 is better).\n\n")
        f.write(md_table)

    print(f"Evaluation complete. Report saved to {report_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    evaluate(args.input, args.report)

if __name__ == "__main__":
    main()