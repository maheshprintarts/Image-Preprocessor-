from PIL import Image
import os
import cv2
import numpy as np

def create_master_layer(width=1920, height=1200, output_mode='RGBA'):
    """
    Creates a new, transparent (alpha channel) base layer.
    Default size: 1920x1200 (WUXGA)
    """
    # 'RGBA' mode includes Alpha channel (Transparency)
    # The color (0, 0, 0, 0) means fully transparent black
    master_layer = Image.new(output_mode, (width, height), (0, 0, 0, 0))
    print(f"Created Master Layer: {width}x{height}, Mode: {output_mode}")
    return master_layer


def get_sharpness_score(pil_img):
    """
    Calculates the Laplacian Variance of a PIL image.
    Higher score = sharper/denser pixels.
    """
    # Convert PIL RGB to OpenCV BGR (or Grayscale directly)
    open_cv_image = np.array(pil_img) 
    # PIL is RGB, OpenCV expects BGR for color, but we just need Grayscale
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance

def get_edge_density(pil_img):
    """
    Calculates Edge Density (Canny Edge Pixels / Total Pixels).
    Low density (< 0.02-0.03) implies the image is soft, blurry, or lacks detail.
    """
    # Convert properly to CV2 grayscale
    open_cv_image = np.array(pil_img.convert('RGB')) 
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2GRAY)
    
    # Canny Edge Detection
    edges = cv2.Canny(gray, 100, 200)
    
    edge_pixels = np.count_nonzero(edges)
    total_pixels = gray.shape[0] * gray.shape[1]
    
    return edge_pixels / total_pixels

def progressive_resize(img, target_size, step=0.9):
    """
    Resizes an image in multiple steps to ensure higher quality when downscaling.
    target_size: (width, height)
    step: scaling factor per step (e.g. 0.9 = 10% reduction)
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
    """
    Calculates the Laplacian variance to measure sharpness/edge density.
    Higher score = sharper image.
    """
    open_cv_image = np.array(pil_img.convert('L'))
    laplacian_var = cv2.Laplacian(open_cv_image, cv2.CV_64F).var()
    return laplacian_var

def optimize_image_size(img, target_score=600.0, step=0.95, is_prescaled=False):
    """
    Dynamically scales image globally until it matches the "10% reference ratio" crispness of ~600.
    """
    current_img = img
    width, height = current_img.size
    
    score = get_sharpness_score(current_img)
    print(f"Initial Sharpness: {score:.2f} (Target: {target_score})")
    
    # If it was pre-scaled from >20MB, its sharpness artificially skyrocketed.
    # We must treat its current high score as the baseline and target a slightly 
    # higher sharpness to force it to run at least one or two perceptual downscaling iterations.
    if is_prescaled and score > target_score:
        print(f"  [Notice] Image was pre-scaled. Adjusting target from {target_score} to match new density.")
        # We set the target to its current score + a small margin so it performs
        # the same relative perceptual scaling as a normal image.
        target_score = score * 1.05
        print(f"  [Notice] New Adaptive Target Score: {target_score:.2f}")
    
    while score < target_score:
        new_width = max(1, int(current_img.width * step))
        new_height = max(1, int(current_img.height * step))
            
        current_img = current_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        score = get_sharpness_score(current_img)
        print(f"  -> Downscaled to {new_width}x{new_height} | New Score: {score:.2f}")

    print(f"Final Sharpness: {score:.2f} at {current_img.size[0]}x{current_img.size[1]}")
    return current_img

def process_and_overlay(input_path, output_path):
    try:
        # Open User Image logic with Decompression Bomb Handling
        img_obj = None
        try:
            # Try to open normally
            img_obj = Image.open(input_path)
        except Image.DecompressionBombError:
            print(f"Decompression Bomb activated for: {input_path}")
            print(">>> Scaling down to half size and retrying current process...")
            
            original_limit = Image.MAX_IMAGE_PIXELS
            Image.MAX_IMAGE_PIXELS = None
            try:
                with Image.open(input_path) as big_img:
                    w, h = big_img.size
                    img_obj = progressive_resize(big_img, (w // 2, h // 2))
                print(f"    Resized to {img_obj.size}. Proceeding...")
            finally:
                Image.MAX_IMAGE_PIXELS = original_limit

        # Proceed with processing the image object
        if img_obj:
            with img_obj as img:
                print(f"--- Processing: {input_path} ---")
                
                # Check pixel count. If > 20MP, scale it down first.
                total_pixels = img.width * img.height
                if total_pixels > 20_000_000:
                    mp = total_pixels / 1_000_000
                    print(f"Image is {mp:.2f}MP (over 20MP limit). Scaling down first...")
                    
                    # We estimate how much to scale based on pixel ratio
                    # Area scales quadratically with dimensions, so we use the square root
                    scale_factor = (20_000_000 / total_pixels) ** 0.5
                    
                    # To be safe, we reduce it a bit more (e.g., 95% of the calculated ratio)
                    safe_scale_factor = scale_factor * 0.95 
                    
                    new_w = max(1, int(img.width * safe_scale_factor))
                    new_h = max(1, int(img.height * safe_scale_factor))
                    
                    print(f"  -> Pre-scaling image from {img.width}x{img.height} to {new_w}x{new_h} to get under 20MP")
                    # Use progressive_resize instead of a single massive resize step
                    img = progressive_resize(img, (new_w, new_h))
                    
                    # Because we downscaled massively, the pixel density (sharpness) 
                    # will jump significantly (e.g., from 500 to 1500).
                    # We need to calculate the *new* sharpness score now.
                    post_scale_score = get_sharpness_score(img)
                    print(f"  -> Sharpness after pre-scale: {post_scale_score:.2f}")
                    
                    # We pass a flag to tell the optimization it's a pre-scaled image
                    is_prescaled = True
                else:
                    is_prescaled = False
                
                # Convert to RGBA for consistent handling
                if img.mode != 'RGBA':
                    img = img.convert('RGBA')
                
                # Dynamic Perceptual Target Scaling 
                rgb_check = img.convert('RGB')
                optimized_rgb = optimize_image_size(rgb_check, target_score=550.0, step=0.95, is_prescaled=is_prescaled)
                
                # Apply optimization resize to RGBA original if changed
                final_size = optimized_rgb.size
                if final_size != img.size:
                    print(f"Applying Target Scale dimensions: {final_size}")
                    img = progressive_resize(img, final_size)
                
                # Save Final Output guaranteeing exactly 300 DPI
                if output_path.lower().endswith(('.jpg', '.jpeg')):
                    final_img = img.convert('RGB')
                    final_img.save(output_path, dpi=(300, 300), quality=100, subsampling=0, optimize=True, progressive=True)
                else:
                    final_img = img  # Keep RGBA for PNG
                    final_img.save(output_path, dpi=(300, 300))
                
                print(f"Saved optimized image to: {output_path} (Size: {final_img.size[0]}x{final_img.size[1]} @ 300 DPI)\n")
                
    except Exception as e:
        print(f"Error processing {input_path}: {e}")

import os

def process_batch(input_dir, output_dir):
    """
    Processes all images in input_dir and saves them to output_dir.
    """
    # 1. Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # 2. Get list of valid images
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)]
    
    total_files = len(files)
    print(f"Found {total_files} images in '{input_dir}'\n")

    # 3. Process each file
    for i, filename in enumerate(files):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        print(f"[{i+1}/{total_files}] Processing {filename}...")
        process_and_overlay(input_path, output_path)

if __name__ == "__main__":
    # Define directories
    input_folder = "input_images"
    output_folder = "processed_images"
    
    # Create input folder if it doesn't exist (for user convenience)
    if not os.path.exists(input_folder):
        os.makedirs(input_folder)
        print(f"Created '{input_folder}'. Please put your images here and run the script again.")
    else:
        # Run batch processing
        process_batch(input_folder, output_folder)
        print("\nBatch processing complete!")
