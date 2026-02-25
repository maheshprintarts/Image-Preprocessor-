import cv2
import numpy as np
import os

def create_density_map(image_path, output_path, block_size=31):
    """
    Creates a heatmap showing pixel density (local variance/detail).
    Dark/Blue areas = Low Density (smooth, blurry, or flat color).
    Bright/Red areas = High Density (sharp edges, lots of detail).
    """
    # 1. Read the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Could not read {image_path}")
        return

    # 2. Calculate local variance (E[X^2] - (E[X])^2)
    # This measures how much the pixels change in a local neighborhood
    img_float = img.astype(np.float32)
    
    # E[X] (Local mean)
    mu = cv2.blur(img_float, (block_size, block_size))
    # E[X^2] (Local mean of squares)
    mu_sq = cv2.blur(img_float**2, (block_size, block_size))
    
    # Variance
    variance = mu_sq - mu**2
    variance = np.maximum(variance, 0) # Prevent negative values due to float precision
    
    # Standard deviation represents the "density" of details
    std_dev = np.sqrt(variance)
    
    # 3. Create a mask to highlight ONLY the VERY LOW density areas
    # Let's say anything below the 10th percentile of variance is "very low"
    # Or we can just build a heatmap of the whole thing.
    
    # Normalize standard deviation to 0-255 for visualization
    std_dev_norm = cv2.normalize(std_dev, None, 0, 255, cv2.NORM_MINMAX)
    std_dev_8u = np.uint8(std_dev_norm)
    
    # Apply a Jet colormap (Blue = Low Density, Red/Yellow = High Density)
    heatmap = cv2.applyColorMap(std_dev_8u, cv2.COLORMAP_JET)
    
    # 4. Highlight the extremely low density areas explicitly with solid black or white
    # Find the threshold for the bottom 15% of detail density
    threshold_val = np.percentile(std_dev, 15)
    low_density_mask = std_dev < threshold_val
    
    # Highlight the low density areas with a bright color (e.g., pure red or pure white)
    # Let's tint the absolute lowest density areas pure MAGENTA (255, 0, 255) to make them obvious
    heatmap[low_density_mask] = [255, 0, 255] # BGR format in OpenCV
    
    cv2.imwrite(output_path, heatmap)
    print(f"Saved density map to: {output_path}")

def process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_extensions)]
    
    print(f"Found {len(files)} images. Generating pixel density maps...\n")
    
    for filename in files:
        input_path = os.path.join(input_folder, filename)
        name, ext = os.path.splitext(filename)
        output_path = os.path.join(output_folder, f"{name}_density.jpg")
        
        print(f"Analyzing: {filename}...")
        create_density_map(input_path, output_path, block_size=31)

if __name__ == "__main__":
    input_dir = "input_images"
    output_dir = "density_analysis"
    
    if os.path.exists(input_dir):
        process_folder(input_dir, output_dir)
        print("\nAnalysis complete! Check the 'density_analysis' folder.")
    else:
        print(f"Folder '{input_dir}' not found.")
