from PIL import Image
import os
import cv2
import numpy as np

# Apply monkey-patch to scikit-image color module for compatibility with imquality
try:
    import skimage.color
    _orig_rgb2gray = skimage.color.rgb2gray
    def _patched_rgb2gray(rgb):
        if rgb.ndim == 2:
            return rgb
        return _orig_rgb2gray(rgb)
    skimage.color.rgb2gray = _patched_rgb2gray

    import libsvm.svmutil
    if not hasattr(libsvm.svmutil, 'PRECOMPUTED'):
        libsvm.svmutil.PRECOMPUTED = 4
    import libsvm.svm
    if not hasattr(libsvm.svm, 'PRECOMPUTED'):
        libsvm.svm.PRECOMPUTED = 4

    import imquality.brisque as brisque
    BRISQUE_AVAILABLE = True
except ImportError:
    print("Warning: Could not import imquality.brisque. Is 'image-quality' installed?")
    BRISQUE_AVAILABLE = False

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
    open_cv_image = np.array(pil_img.convert('L'))
    laplacian_var = cv2.Laplacian(open_cv_image, cv2.CV_64F).var()
    return laplacian_var


def get_brisque_score(pil_img):
    """
    Calculates the BRISQUE score of a PIL image.
    Lower score = better perceptual quality.
    Note: Evaluates a 512x512 center crop to avoid massive CPU overhead.
    """
    if not BRISQUE_AVAILABLE:
        # Fallback to laplacian if brisque isn't available
        open_cv_image = np.array(pil_img) 
        gray = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2GRAY)
        return -cv2.Laplacian(gray, cv2.CV_64F).var() # Negative so lower is better

    w, h = pil_img.size
    crop_size = min(512, w, h)
    left = (w - crop_size) // 2
    top = (h - crop_size) // 2
    cropped_img = pil_img.crop((left, top, left + crop_size, top + crop_size))
    
    open_cv_image = np.array(cropped_img)
    return brisque.score(open_cv_image)

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

def optimize_image_size(img, target_sharpness=600.0, step=0.95):
    """
    Dynamically scales image globally to hit a target Laplacian sharpness,
    while using the BRISQUE score to prevent perceptual degradation,
    and setting hard boundaries to avoid stamp-sized shrinkages.
    """
    current_img = img
    
    best_brisque = get_brisque_score(current_img)
    best_img = current_img
    
    sharpness = get_sharpness_score(current_img)
    print(f"Initial Sharpness: {sharpness:.2f} | Initial BRISQUE Score: {best_brisque:.2f}")
    
    margin = 1.5 
    
    # Check if the initial sharpness is already heavily dense.
    if sharpness >= target_sharpness:
        print(f"  -> Image is already highly dense/sharp. Evaluating BRISQUE optimization only if extremely large.")
        
    while sharpness < target_sharpness:
        new_width = max(1, int(current_img.width * step))
        new_height = max(1, int(current_img.height * step))
        
        # Prevent absolute stamp size (don't shrink below 1500 for largest dimension)
        if max(new_width, new_height) < 1500:
            print(f"  -> Stopping: Reached minimum safe resolution limits.")
            break
            
        # Prevent shrinking below 30% (protect original size relationship)
        if new_width < img.width * 0.3 or new_height < img.height * 0.3:
            print("  -> Stopping: Reached 30% limit of original dimensions.")
            break
            
        current_img = current_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        brisque_score = get_brisque_score(current_img)
        sharpness = get_sharpness_score(current_img)
        
        print(f"  -> {new_width}x{new_height} | Sharpness: {sharpness:.2f} | BRISQUE: {brisque_score:.2f}")
        
        if brisque_score < best_brisque:
            # Perceptual quality improved, save as the best version.
            best_brisque = brisque_score
            best_img = current_img
        elif brisque_score > best_brisque + margin:
            # Perceptual quality degraded past our safety margin. Halt shrink.
            print(f"  -> Stopping: Perceptual quality degraded past sweet spot margin.")
            break
        else:
            # BRISQUE stayed roughly identical, but sharpness increased (getting denser). Update best.
            best_img = current_img 

    print(f"Final Sweet Spot Size: {best_img.size[0]}x{best_img.size[1]} | Sharpness: {get_sharpness_score(best_img):.2f} | BRISQUE: {get_brisque_score(best_img):.2f}")
    return best_img

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
                
                # Convert to RGBA for consistent handling
                if img.mode != 'RGBA':
                    img = img.convert('RGBA')
                
                # Dynamic Perceptual Target Scaling 
                rgb_check = img.convert('RGB')
                optimized_rgb = optimize_image_size(rgb_check, step=0.95)
                
                # Apply optimization resize to RGBA original if changed
                final_size = optimized_rgb.size
                if final_size != img.size:
                    print(f"Applying Target Scale dimensions: {final_size}")
                    img = progressive_resize(img, final_size)
                
                # Save Final Output guaranteeing exactly 96 DPI
                if output_path.lower().endswith(('.jpg', '.jpeg')):
                    final_img = img.convert('RGB')
                    final_img.save(output_path, dpi=(300, 300), quality=300, subsampling=0, optimize=True, progressive=True)
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
