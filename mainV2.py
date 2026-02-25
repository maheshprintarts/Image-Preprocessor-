from PIL import Image, ImageFilter
import os
import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Sharpness / Quality Metrics
# ---------------------------------------------------------------------------

def get_sharpness_score(pil_img):
    """
    Calculates the Laplacian Variance of a PIL image.
    Higher score = sharper / more detail visible.
    """
    open_cv_image = np.array(pil_img.convert('RGB'))
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


# ---------------------------------------------------------------------------
# Progressive Resize (multi-step, high quality)
# ---------------------------------------------------------------------------

def progressive_resize(img, target_size, step=0.9):
    """
    Resizes an image in multiple steps to preserve quality.
    target_size: (width, height)
    step: scaling factor per step (0.9 = 10% reduction each step)
    """
    current_img = img
    target_w, target_h = target_size

    while current_img.width > target_w * 1.1 or current_img.height > target_h * 1.1:
        new_w = max(int(current_img.width * step), target_w)
        new_h = max(int(current_img.height * step), target_h)
        current_img = current_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        print(f"    -> Progressively scaled to: {new_w}x{new_h}")

    if current_img.size != target_size:
        current_img = current_img.resize(target_size, Image.Resampling.LANCZOS)
        print(f"    -> Final exact scale to: {target_w}x{target_h}")

    return current_img


# ---------------------------------------------------------------------------
# Peak-Sharpness Search  (the core perceptual quality engine)
# ---------------------------------------------------------------------------

def find_peak_sharpness_scale(img, step=0.95, min_size=500, max_steps=60):
    """
    Finds the scale at which the image reaches its PEAK perceptual sharpness.

    Instead of targeting a fixed score (which forces tiny blurry images to
    shrink forever), this algorithm:
      1. Iteratively downscales the image by `step` each iteration.
      2. Records the sharpness score at every scale.
      3. Stops when the score starts decreasing (peak passed) OR when the
         image reaches the minimum size floor.
      4. Returns the image at the best scale found.

    Parameters:
        img       : PIL Image (RGB)
        step      : Proportional scale reduction per iteration (default 0.95 = 5% per step)
        min_size  : Minimum pixels in the shortest dimension (default 500px)
        max_steps : Safety cap on number of iterations

    Returns:
        best_img  : PIL Image at its peak sharpness scale
    """
    current_img = img.copy()
    best_img = img.copy()
    best_score = get_sharpness_score(current_img)
    prev_score = best_score

    print(f"  [Peak Search] Starting sharpness: {best_score:.2f} at {current_img.width}x{current_img.height}")

    no_improve_streak = 0  # Count consecutive steps without improvement

    for i in range(max_steps):
        # Respect the minimum size floor
        short_side = min(current_img.width, current_img.height)
        if short_side <= min_size:
            print(f"  [Peak Search] Hit minimum size floor ({min_size}px). Stopping.")
            break

        new_w = max(int(current_img.width * step), 1)
        new_h = max(int(current_img.height * step), 1)
        current_img = current_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        score = get_sharpness_score(current_img)

        print(f"  -> Scale {i+1}: {new_w}x{new_h} | Sharpness: {score:.2f}")

        if score > best_score:
            best_score = score
            best_img = current_img.copy()
            no_improve_streak = 0
        else:
            no_improve_streak += 1

        # If score has been declining for 3 consecutive steps, we've passed the peak
        if no_improve_streak >= 3:
            print(f"  [Peak Search] Score declining for {no_improve_streak} steps. Peak found!")
            break

        prev_score = score

    print(f"  [Peak Search] Best sharpness: {best_score:.2f} at {best_img.width}x{best_img.height}")
    return best_img, best_score


# ---------------------------------------------------------------------------
# Unsharp Masking — for genuinely soft images
# ---------------------------------------------------------------------------

def enhance_sharpness_if_needed(img, score_threshold=150.0):
    """
    If the peak sharpness score is below `score_threshold`, apply Unsharp
    Masking to improve the perceived clarity of the image before saving.

    This is a standard photographic post-processing technique that enhances
    edges without adding fake detail.
    """
    score = get_sharpness_score(img)
    if score < score_threshold:
        print(f"  [Enhance] Score {score:.2f} is below threshold {score_threshold}. Applying Unsharp Mask...")
        # radius=2, percent=150, threshold=3 is a mild but effective setting
        img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
        new_score = get_sharpness_score(img)
        print(f"  [Enhance] Sharpness improved: {score:.2f} -> {new_score:.2f}")
    return img


# ---------------------------------------------------------------------------
# Main Image Processing Function
# ---------------------------------------------------------------------------

def process_and_overlay(input_path, output_path):
    try:
        # --- Step 1: Open image (with Decompression Bomb protection) ---
        img_obj = None
        try:
            img_obj = Image.open(input_path)
        except Image.DecompressionBombError:
            print(f"  [Warning] Decompression Bomb detected: {input_path}")
            print("  >>> Scaling down to half size and retrying...")
            original_limit = Image.MAX_IMAGE_PIXELS
            Image.MAX_IMAGE_PIXELS = None
            try:
                with Image.open(input_path) as big_img:
                    w, h = big_img.size
                    img_obj = progressive_resize(big_img.copy(), (w // 2, h // 2))
                print(f"    Resized to {img_obj.size}. Proceeding...")
            finally:
                Image.MAX_IMAGE_PIXELS = original_limit

        if not img_obj:
            print(f"  [Error] Could not open image: {input_path}")
            return

        with img_obj as img:
            print(f"--- Processing: {input_path} ---")

            # --- Step 2: Pre-scale images over 20 Megapixels ---
            total_pixels = img.width * img.height
            is_prescaled = False

            if total_pixels > 20_000_000:
                mp = total_pixels / 1_000_000
                print(f"  Image is {mp:.2f}MP (over 20MP). Pre-scaling to ~20MP first...")

                scale_factor = ((20_000_000 / total_pixels) ** 0.5) * 0.95
                new_w = max(1, int(img.width * scale_factor))
                new_h = max(1, int(img.height * scale_factor))

                print(f"  -> Pre-scaling from {img.width}x{img.height} to {new_w}x{new_h}")
                img = progressive_resize(img, (new_w, new_h))
                is_prescaled = True
                print(f"  -> Sharpness after pre-scale: {get_sharpness_score(img.convert('RGB')):.2f}")

            # --- Step 3: Peak-sharpness search ---
            # Convert to RGB for the analysis (OpenCV needs it)
            rgb_img = img.convert('RGB')

            # For pre-scaled images, use smaller min_size since they are already large
            min_dimension = 500 if not is_prescaled else 800

            best_rgb, best_score = find_peak_sharpness_scale(
                rgb_img,
                step=0.95,         # 5% reduction per step (smoother than 2%)
                min_size=min_dimension,
                max_steps=60
            )

            # --- Step 4: Apply unsharp masking for very soft images ---
            best_rgb = enhance_sharpness_if_needed(best_rgb, score_threshold=150.0)

            # --- Step 5: Apply the found optimal size back to original-mode image ---
            optimal_size = best_rgb.size
            if optimal_size != img.size:
                print(f"  Applying optimal size {optimal_size} to image...")
                img = progressive_resize(img, optimal_size)

            # --- Step 6: Screen boundary cap (>2000px → fit inside 1920×1080) ---
            max_w, max_h = 1920, 1080
            if img.width > 2000 or img.height > 2000:
                print(f"  Image {img.width}x{img.height} exceeds 2000px. Fitting to {max_w}x{max_h}...")
                scale = min(max_w / img.width, max_h / img.height)
                fit_w = max(1, int(img.width * scale))
                fit_h = max(1, int(img.height * scale))
                print(f"  -> Screen boundary scale: {fit_w}x{fit_h}")
                img = img.resize((fit_w, fit_h), Image.Resampling.LANCZOS)

            # --- Step 7: Save final output ---
            if output_path.lower().endswith(('.jpg', '.jpeg')):
                final_img = img.convert('RGB')
                final_img.save(
                    output_path,
                    dpi=(300, 300),
                    quality=100,
                    subsampling=0,
                    optimize=True,
                    progressive=True
                )
            else:
                final_img = img.convert('RGBA') if img.mode != 'RGBA' else img
                final_img.save(output_path, dpi=(300, 300))

            final_score = get_sharpness_score(final_img.convert('RGB'))
            print(f"  Saved: {output_path} | Size: {final_img.size[0]}x{final_img.size[1]} | DPI: 300 | Sharpness: {final_score:.2f}\n")

    except Exception as e:
        print(f"  [Error] Processing {input_path}: {e}")
        import traceback
        traceback.print_exc()


# ---------------------------------------------------------------------------
# Batch Processor
# ---------------------------------------------------------------------------

def process_batch(input_dir, output_dir):
    """
    Processes all images in input_dir and saves them to output_dir.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)]
    total_files = len(files)

    if total_files == 0:
        print(f"No images found in '{input_dir}'. Please add images and run again.")
        return

    print(f"Found {total_files} image(s) in '{input_dir}'\n")

    for i, filename in enumerate(files):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        print(f"[{i+1}/{total_files}] Processing {filename}...")
        process_and_overlay(input_path, output_path)


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    input_folder = "input_images"
    output_folder = "processed_images"

    if not os.path.exists(input_folder):
        os.makedirs(input_folder)
        print(f"Created '{input_folder}'. Please put your images here and run again.")
    else:
        process_batch(input_folder, output_folder)
        print("\nBatch processing complete!")
