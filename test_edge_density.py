import cv2
import numpy as np
import sys
import os

def analyze_edge_density(image_path):
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Could not load {image_path}")
            return

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Canny Edge Detection
        # Low, High thresholds
        edges = cv2.Canny(gray, 100, 200)
        
        # Count edge pixels
        edge_pixels = np.count_nonzero(edges)
        total_pixels = img.shape[0] * img.shape[1]
        
        density = edge_pixels / total_pixels
        
        # Laplacian (Sharpness) again for comparison
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        print(f"--- {os.path.basename(image_path)} ---")
        print(f"Dimensions: {img.shape[1]}x{img.shape[0]}")
        print(f"Edge Density: {density:.4f}")
        print(f"Laplacian Var: {laplacian_var:.2f}")
        print("-" * 20)
        
        return density, laplacian_var

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    folder = "input_images"
    if os.path.exists(folder):
        files = [f for f in os.listdir(folder) if f.lower().endswith('.jpg')]
        for f in files:
            analyze_edge_density(os.path.join(folder, f))
