# Satellite Tile Enhancement — Writeup

### 1. Why I picked each method

**Classical pipeline:**

1. **Denoise first (Non-Local Means)** — I chose this over a simple blur because it's smart about edges. It looks at similar patches across the whole image before deciding how much to smooth a pixel, so you get cleaner results without losing roof lines or road boundaries.

2. **CLAHE on the L channel (LAB color space)** — Normal histogram equalization on a full color image tends to make things look weirdly saturated or "glowing." By converting to LAB and only adjusting the lightness channel, I can boost local contrast without the colors going haywire.

3. **Unsharp masking to sharpen** — Subtracts a blurred copy of the image from itself to amplify edges. Good at making rooflines and pavement markings pop. The downside is that if there's already JPEG noise in the image, this step can amplify that too — which is exactly what the failure cases show.

**SR pipeline:**

I used FSRCNN through OpenCV's `dnn_superres` module. I picked FSRCNN specifically because it's lightweight enough to run on CPU in reasonable time and still consistently beats bicubic at preserving sharp edges in buildings and roads. It's not the newest model, but it's a solid choice when you don't have a GPU.

---

### 2. How I set up the evaluation

The tricky part here is that the 640×640 tiles we have *are* the highest quality images — there's no true "super hi-res" ground truth to compare against. So I had to fake it.

What I do: take the original 640×640 tile, shrink it to 320×320 and add JPEG compression artifacts (this simulates what a low-quality API response might look like), then let each method try to reconstruct the 640×640 original. Finally, measure PSNR and SSIM against the original to see how well each method recovered the details.

**Honest limitation:** I'm assuming that synthetic downsampling + JPEG compression is a fair stand-in for the real-world degradation that comes from sensors, atmospheric haze, and Google's actual compression pipeline. It isn't a perfect match. It's also a bit unfair to the classical method since sharpening intentionally shifts pixel values away from the "true" values to make things *look* sharper — which hurts PSNR even when the output looks better to human eyes.

---

### 3. Where classical enhancement fails

*Each image below is a 200×200 crop from the region with the most variation in that tile. Layout: Original | Classical | SR.*

**Failure 1 — Sharpening amplifies JPEG noise** (`building_57c41d_mapbox`)

Classical dropped **16 dB** below bicubic on this one. The tile already had compression block artifacts and unsharp masking treated them as texture, making them noticeably worse.

![Failure Case 1 crop](crops/building_57c41d_mapbox_crop.jpg)

**Failure 2 — Halo ringing around roof edges** (`building_fc48ee_mapbox`)

Classical dropped **16 dB** below bicubic. Where there's a bright roof next to dark asphalt, the CLAHE + sharpening combo creates visible bright halos around the edges.

![Failure Case 2 crop](crops/building_fc48ee_mapbox_crop.jpg)

**Failure 3 — Color banding in flat areas** (`building_b31756_mapbox`)

Classical dropped **~14 dB** below bicubic. Low-detail tiles with large uniform regions are the worst case for this pipeline — the sharpening step picks up JPEG grid artifacts and turns them into visible color banding.

![Failure Case 3 crop](crops/building_b31756_mapbox_crop.jpg)

---

### 4. What happens if you need to do 50,000 tiles a day

The SR model is the bottleneck. Running FSRCNN on CPU, one image at a time, in a loop, just won't cut it at that scale.

What I'd change:
1. Add a message queue (Celery + Redis, or AWS SQS) so tiles can be processed in parallel across multiple workers instead of one-by-one.
2. Move SR inference to a GPU and use ONNX Runtime or TensorRT to batch-process multiple images at once. This alone would drop inference time by an order of magnitude.

---

### 5. Google Maps ToS — what you can and can't do

Google's platform terms are pretty strict. You can't store their imagery indefinitely, you can't build a standalone database of their map tiles, and you can't mass-scrape. The caching exception they allow is for temporary performance reasons only (think short-lived CDN cache, not a permanent database).

In practice this means: if we were doing this for real, we'd have to process tiles on-the-fly and discard them after use, rather than building a persistent enhanced tile store. For anything bigger you'd need enterprise licensing.