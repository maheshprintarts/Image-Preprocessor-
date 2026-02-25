import cv2
import numpy as np
from PIL import Image
import os

def get_sharpness_score(pil_img):
    open_cv_image = np.array(pil_img.convert('RGB')) 
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance

def get_edge_density(pil_img):
    open_cv_image = np.array(pil_img.convert('RGB')) 
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_pixels = np.count_nonzero(edges)
    total_pixels = gray.shape[0] * gray.shape[1]
    return edge_pixels / total_pixels

def analyze_scale(image_path, scales):
    img = Image.open(image_path).convert('RGB')
    orig_w, orig_h = img.size
    
    print(f"--- Analyzing {os.path.basename(image_path)} ({orig_w}x{orig_h}) ---")
    
    for scale in scales:
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        sharpness = get_sharpness_score(resized)
        density = get_edge_density(resized)
        
        print(f"Scale: {scale * 100:3.0f}% | Size: {new_w:4d}x{new_h:4d} | Sharpness (Laplacian): {sharpness:7.2f} | Edge Density: {density:.4f}")

if __name__ == "__main__":
    scales_to_test = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.48, 0.4, 0.3, 0.2]
    # Test on the primary image and another image
    analyze_scale("input_images/0-0_4x_refined.png", scales_to_test)
    print("\n")
    analyze_scale("input_images/30974762-1.jpg", scales_to_test)
