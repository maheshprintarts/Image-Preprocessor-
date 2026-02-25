import cv2
import numpy as np
import os
from PIL import Image

def get_blurry_area_density(pil_img, block_size=21):
    open_cv_image = np.array(pil_img.convert('L'))
    img_float = open_cv_image.astype(np.float32)
    
    mu = cv2.blur(img_float, (block_size, block_size))
    mu_sq = cv2.blur(img_float**2, (block_size, block_size))
    variance = np.maximum(mu_sq - mu**2, 0)
    std_dev = np.sqrt(variance)
    
    # Get the average density of the bottom 25% (the blurriest parts)
    bottom_25_threshold = np.percentile(std_dev, 25)
    blurry_pixels = std_dev[std_dev <= bottom_25_threshold]
    
    return np.mean(blurry_pixels) if len(blurry_pixels) > 0 else 0

def profile_blurry_density(path):
    print(f"\n--- Profiling Blurry Area Density for {os.path.basename(path)} ---")
    img = Image.open(path)
    
    original_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = None
    
    try:
        scales = [1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        for scale in scales:
            new_w, new_h = max(1, int(img.width * scale)), max(1, int(img.height * scale))
            test_img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            density = get_blurry_area_density(test_img)
            print(f"Scale {scale*100:3.0f}% | Size: {new_w:4d}x{new_h:4d} | Blurry Area Density: {density:.2f}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        Image.MAX_IMAGE_PIXELS = original_limit

if __name__ == '__main__':
    profile_blurry_density('input_images/0-0_4x_refined.png')
    profile_blurry_density('input_images/30769253-2.jpg')
    profile_blurry_density('input_images/30911774-1.jpg')
    profile_blurry_density('input_images/30960192-1.jpg')
