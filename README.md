# Image Preprocessor (mainV2.py)

A Python script that intelligently resizes images based on **perceptual sharpness** — not fixed pixel sizes.

> **Also available as an installable Python package:** [ImgPre](https://github.com/maheshprintarts/ImgPre)

## Supported Image Types

| Format | Extension |
|---|---|
| JPEG | `.jpg`, `.jpeg` |
| PNG | `.png` |
| BMP | `.bmp` |
| TIFF | `.tiff` |
| CMYK (Photoshop) | `.jpg` |
| Grayscale | `.jpg`, `.png` |
| RGBA transparent | `.png` |

All color spaces (CMYK, Grayscale, RGBA, Palette) are automatically converted to RGB before processing.

## How to Use

1. Install dependencies:
```bash
pip install Pillow opencv-python numpy
```

2. Drop your images into the `input_images/` folder.

3. Run:
```bash
python mainV2.py
```

4. Find your results in `processed_images/`.

## How It Works

```
1. Open Image (any format/color space)
       ↓
2. Convert to RGB (CMYK, RGBA, Grayscale all normalized)
       ↓
3. Pre-Scale if >20MP (prevents memory crashes on huge files)
       ↓
4. Perceptual Optimization
       • Downscales 2% per step
       • Adaptive target: 1.5× own sharpness baseline
       • Stops when quality improvements plateau
       • Never shrinks below 500px on shorter side
       ↓
5. Screen Fit (if >2000px → proportionally fit within 1920×1080)
       ↓
6. Save at 300 DPI (JPEG: quality=100, progressive)
```

## Settings

| Setting | Value | Description |
|---|---|---|
| Sharpness Target | `1.5×` own baseline | Adaptive per-image |
| Step Size | `2%` per iteration | Slow, high-quality scaling |
| Pre-scale Trigger | `>20 Megapixels` | Safe memory handling |
| Screen Boundary | `>2000px` → fit `1920×1080` | Screen-safe output |
| Min Size Guard | `500px` short side | No postage-stamp outputs |
| Output DPI | `300` | Print-ready |
| JPEG Quality | `100` | Maximum quality |
