import cv2
import sys
import os

def calculate_sharpness(image_path):
    try:
        # Read image in grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Error: Could not read {image_path}")
            return None
        
        # Calculate Laplacian Variance
        # Higher variance = Sharper edges (more detail)
        # Lower variance = Blurry or flat
        variance = cv2.Laplacian(img, cv2.CV_64F).var()
        return variance

    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) > 1:
        files = sys.argv[1:]
    else:
        files = ["1.jpg", "3.jpg"] # Default test files if they exist

    print(f"{'File':<20} | {'Sharpness Score':<15}")
    print("-" * 40)
    
    for f in files:
        if os.path.exists(f):
            score = calculate_sharpness(f)
            print(f"{f:<20} | {score:.2f}")
        else:
             print(f"{f:<20} | Not Found")
