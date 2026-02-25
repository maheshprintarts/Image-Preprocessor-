import cv2
import numpy as np
import os
from PIL import Image

def get_native_sharpness_metric(pil_img):
    img_gray = np.array(pil_img.convert('L'))
    # A sharp image will change drastically when blurred.
    # A blurry (upscaled) image won't change much when blurred.
    blurred = cv2.GaussianBlur(img_gray, (3, 3), 0)
    diff = cv2.absdiff(img_gray, blurred)
    return np.mean(diff)

def profile_native_resolution(path):
    print(f"\n--- Profiling Native Resolution for {os.path.basename(path)} ---")
    img = Image.open(path)
    
    original_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = None
    
    try:
        scales = np.arange(1.0, 0.09, -0.05)
        last_metric = None
        for scale in scales:
            new_w, new_h = max(1, int(img.width * scale)), max(1, int(img.height * scale))
            test_img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            metric = get_native_sharpness_metric(test_img)
            
            change = 0
            if last_metric is not None:
                change = metric - last_metric
                
            print(f"Scale {scale*100:3.0f}% | Size: {new_w:4d}x{new_h:4d} | Blur Diff MAE: {metric:5.2f} | Change: +{change:5.2f}")
            last_metric = metric
    except Exception as e:
        print(f"Error: {e}")
    finally:
        Image.MAX_IMAGE_PIXELS = original_limit

if __name__ == '__main__':
    profile_native_resolution('input_images/0-0_4x_refined.png')
    profile_native_resolution('input_images/30769253-2.jpg')
    profile_native_resolution('input_images/30911774-1.jpg')
    profile_native_resolution('input_images/30960192-1.jpg')
