import cv2
import numpy as np
from PIL import Image
import os

def get_sharpness(img):
    open_cv_image = np.array(img.convert('L'))
    return cv2.Laplacian(open_cv_image, cv2.CV_64F).var()

def profile_image(path):
    try:
        img = Image.open(path)
        print(f"\n--- Profiling {os.path.basename(path)} ---")
        
        # We handle Decompression Bombs here just in case
        original_limit = Image.MAX_IMAGE_PIXELS
        Image.MAX_IMAGE_PIXELS = None
        
        scales = np.arange(1.0, 0.09, -0.05)
        last_s = None
        for scale in scales:
            new_w, new_h = max(1, int(img.width * scale)), max(1, int(img.height * scale))
            test_img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            s = get_sharpness(test_img)
            
            diff = 0
            if last_s is not None:
                diff = s - last_s
                
            print(f"Scale {scale*100:3.0f}% | Size: {new_w:4d}x{new_h:4d} | Sharpness: {s:7.2f} | Change: +{diff:7.2f}")
            last_s = s
    except Exception as e:
        print(f"Error profiling {path}: {e}")
    finally:
        Image.MAX_IMAGE_PIXELS = original_limit

if __name__ == '__main__':
    profile_image('input_images/0-0_4x_refined.png')
    profile_image('input_images/30769253-2.jpg')
    profile_image('input_images/30911774-1.jpg')
    profile_image('input_images/30960192-1.jpg')
