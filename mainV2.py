from PIL import Image
import os
import cv2
import numpy as np


def get_sharpness_score(pil_img):
    """
    Calculates the Laplacian Variance of a PIL image.
    Higher score = sharper/denser pixels.
    """
    open_cv_image = np.array(pil_img)
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance


def get_edge_density(pil_img):
    """
    Calculates Edge Density (Canny Edge Pixels / Total Pixels).
    Low density (< 0.02-0.03) implies the image is soft, blurry, or lacks detail.
    """
    open_cv_image = np.array(pil_img.convert('RGB'))
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_pixels = np.count_nonzero(edges)
    total_pixels = gray.shape[0] * gray.shape[1]
    return edge_pixels / total_pixels


def progressive_resize(img, target_size, step=0.9):
    """
    Resizes an image in multiple steps to ensure higher quality when downscaling.
    target_size: (width, height)
    step: scaling factor per step (e.g. 0.9 = 10% reduction per step)
    """
    current_img = img
    target_w, target_h = target_size

    while current_img.width > target_w * 1.1 or current_img.height > target_h * 1.1:
        new_w = max(int(current_img.width * step), target_w)
        new_h = max(int(current_img.height * step), target_h)
        current_img = current_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        print(f"    -> Progressively scaled to: {new_w}x{new_h}")

    # Final step to exact target size
    if current_img.size != target_size:
        current_img = current_img.resize(target_size, Image.Resampling.LANCZOS)
        print(f"    -> Final exact scale to: {target_w}x{target_h}")

    return current_img


def optimize_image_size(img, step=0.98, is_prescaled=False, min_short_side=300):
    """
    Dynamically scales image down until perceptual quality peak is reached.

    Improvements over the old version:
    1. Relative target: adapts to each image's own starting sharpness
    2. Diminishing returns detection: stops when score improvements become tiny
    3. Minimum size guard: never crushes image below min_short_side pixels
    4. Pre-scale aware: adjusts target for images that were pre-shrunk from >20MP
    """
    current_img = img
    score = get_sharpness_score(current_img)

    # --- Determine adaptive target score ---
    # We scale to 1.5x the image's OWN starting sharpness, or a minimum floor.
    # This means: every image needs at least 50% sharpness improvement over itself.
    min_absolute_target = 400.0
    relative_target = score * 1.5
    target_score = max(relative_target, min_absolute_target)

    # For pre-scaled images, their sharpness artificially jumps after resize.
    # Treat their post-scale score as baseline and require a 5% further improvement.
    if is_prescaled and score > min_absolute_target:
        print(f"  [Notice] Image was pre-scaled. Using relative target from current density.")
        target_score = score * 1.05

    print(f"Initial Sharpness: {score:.2f} | Adaptive Target: {target_score:.2f}")

    prev_score = score
    no_improvement_count = 0
    max_no_improvement_steps = 5  # Stop if score barely moves for 5 steps in a row

    while score < target_score:
        # --- Minimum size guard ---
        short_side = min(current_img.width, current_img.height)
        if short_side <= min_short_side:
            print(f"  [Guard] Reached minimum size ({short_side}px short side). Stopping.")
            break

        new_width = max(1, int(current_img.width * step))
        new_height = max(1, int(current_img.height * step))
        current_img = current_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        score = get_sharpness_score(current_img)
        print(f"  -> Downscaled to {new_width}x{new_height} | Score: {score:.2f}")

        # --- Diminishing returns detection ---
        # If the score improvement is less than 1.5% of previous score, not worth continuing
        improvement = score - prev_score
        if improvement < prev_score * 0.015 and score > prev_score:
            no_improvement_count += 1
            if no_improvement_count >= max_no_improvement_steps:
                print(f"  [Guard] Diminishing returns detected ({no_improvement_count} flat steps). Stopping.")
                break
        else:
            no_improvement_count = 0

        prev_score = score

    print(f"Final Sharpness: {score:.2f} at {current_img.size[0]}x{current_img.size[1]}")
    return current_img


def process_and_overlay(input_path, output_path):
    try:
        # Open image with Decompression Bomb handling
        img_obj = None
        try:
            img_obj = Image.open(input_path)
        except Image.DecompressionBombError:
            print(f"Decompression Bomb activated for: {input_path}")
            print(">>> Scaling down to half size and retrying...")
            original_limit = Image.MAX_IMAGE_PIXELS
            Image.MAX_IMAGE_PIXELS = None
            try:
                with Image.open(input_path) as big_img:
                    w, h = big_img.size
                    img_obj = progressive_resize(big_img, (w // 2, h // 2))
                print(f"    Resized to {img_obj.size}. Proceeding...")
            finally:
                Image.MAX_IMAGE_PIXELS = original_limit

        if img_obj:
            with img_obj as img:
                print(f"--- Processing: {input_path} ---")

                # Check pixel count. If > 20MP, progressively pre-scale first.
                total_pixels = img.width * img.height
                if total_pixels > 20_000_000:
                    mp = total_pixels / 1_000_000
                    print(f"Image is {mp:.2f}MP (over 20MP limit). Scaling down first...")
                    scale_factor = (20_000_000 / total_pixels) ** 0.5
                    safe_scale_factor = scale_factor * 0.95
                    new_w = max(1, int(img.width * safe_scale_factor))
                    new_h = max(1, int(img.height * safe_scale_factor))
                    print(f"  -> Pre-scaling from {img.width}x{img.height} to {new_w}x{new_h}")
                    img = progressive_resize(img, (new_w, new_h))
                    post_scale_score = get_sharpness_score(img.convert('RGB'))
                    print(f"  -> Sharpness after pre-scale: {post_scale_score:.2f}")
                    is_prescaled = True
                else:
                    is_prescaled = False

                # Convert to RGBA for consistent handling
                if img.mode != 'RGBA':
                    img = img.convert('RGBA')

                # --- Perceptual Quality Optimization ---
                rgb_check = img.convert('RGB')
                optimized_rgb = optimize_image_size(
                    rgb_check,
                    step=0.98,
                    is_prescaled=is_prescaled,
                    min_short_side=300
                )

                # Apply optimized dimensions back to original RGBA image
                final_size = optimized_rgb.size
                if final_size != img.size:
                    print(f"Applying optimized dimensions: {final_size[0]}x{final_size[1]}")
                    img = progressive_resize(img, final_size)

                # --- Screen Boundary Check ---
                # If either dimension exceeds 2000px, proportionally fit within 1920x1080
                max_w, max_h = 1920, 1080
                if img.width > 2000 or img.height > 2000:
                    print(f"Image {img.width}x{img.height} exceeds 2000px. Fitting into {max_w}x{max_h}...")
                    scale = min(max_w / img.width, max_h / img.height)
                    fit_w = max(1, int(img.width * scale))
                    fit_h = max(1, int(img.height * scale))
                    print(f"  -> Screen fit scale: {fit_w}x{fit_h}")
                    img = img.resize((fit_w, fit_h), Image.Resampling.LANCZOS)

                # --- Save Output ---
                if output_path.lower().endswith(('.jpg', '.jpeg')):
                    final_img = img.convert('RGB')
                    final_img.save(output_path, dpi=(300, 300), quality=100,
                                   subsampling=0, optimize=True, progressive=True)
                else:
                    final_img = img  # Keep RGBA for PNG
                    final_img.save(output_path, dpi=(300, 300))

                print(f"Saved: {output_path} ({final_img.size[0]}x{final_img.size[1]} @ 300 DPI)\n")

    except Exception as e:
        print(f"Error processing {input_path}: {e}")


def process_batch(input_dir, output_dir):
    """
    Processes all images in input_dir and saves them to output_dir.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)]

    total_files = len(files)
    print(f"Found {total_files} images in '{input_dir}'\n")

    for i, filename in enumerate(files):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        print(f"[{i+1}/{total_files}] Processing {filename}...")
        process_and_overlay(input_path, output_path)


if __name__ == "__main__":
    input_folder = "input_images"
    output_folder = "processed_images"

    if not os.path.exists(input_folder):
        os.makedirs(input_folder)
        print(f"Created '{input_folder}'. Please put your images here and run the script again.")
    else:
        process_batch(input_folder, output_folder)
        print("\nBatch processing complete!")
