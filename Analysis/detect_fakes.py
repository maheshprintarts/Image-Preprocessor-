import cv2
import numpy as np
import os
from PIL import Image, ImageChops

def calculate_fake_pixel_percentage(image, quality=90, threshold=30):
    """
    Performs ELA on an image object and returns the percentage of pixels 
    that differ by more than `threshold`.
    """
    # 1. Save a temporary compressed copy
    temp_filename = "temp_ela_calc.jpg"
    image.save(temp_filename, 'JPEG', quality=quality)
    
    # 2. Open the compressed copy
    compressed = Image.open(temp_filename)
    
    # 3. Calculate difference
    orig_np = np.array(image).astype(np.float32)
    comp_np = np.array(compressed).astype(np.float32)
    diff = np.abs(orig_np - comp_np)
    
    # 4. Count pixels where the difference across any color channel exceeds the threshold
    # A high difference means it degraded significantly, which indicates a "fake" or "pasted" pixel
    if len(diff.shape) == 3: # RGB
        max_diff_per_pixel = np.max(diff, axis=2)
    else:
        max_diff_per_pixel = diff
        
    fake_pixels = np.sum(max_diff_per_pixel > threshold)
    total_pixels = image.width * image.height
    
    fake_percentage = (fake_pixels / total_pixels) * 100
    
    # Clean up
    if os.path.exists(temp_filename):
        os.remove(temp_filename)
        
    return fake_percentage

def optimize_fake_pixels(image_path, output_path, max_fake_percentage=2.0, step=0.95):
    """
    Iteratively scales an image down until the percentage of "fake" pixels 
    is below the max_fake_percentage limit.
    """
    current_img = Image.open(image_path).convert('RGB')
    
    # Calculate initial fake percentage
    percentage = calculate_fake_pixel_percentage(current_img)
    print(f"[{os.path.basename(image_path)}] Initial Fake Pixels: {percentage:.2f}% (Target: < {max_fake_percentage}%)")
    
    # Iteratively reduce size until fake pixels are removed/blended
    while percentage > max_fake_percentage and current_img.width > 100:
        new_w = int(current_img.width * step)
        new_h = int(current_img.height * step)
        
        current_img = current_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        percentage = calculate_fake_pixel_percentage(current_img)
        print(f"  -> Reduced to {new_w}x{new_h} | Fake Pixels: {percentage:.2f}%")
        
    current_img.save(output_path, quality=95)
    print(f"Saved cleaned image to: {output_path} (Final Fake Pixels: {percentage:.2f}%)\n")

def process_folder_for_fakes(input_folder, output_folder):
    """
    Scans a folder and generates ELA maps for all images.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_extensions)]
    
    print(f"Found {len(files)} images to analyze and clean...\n")
    
    for filename in files:
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, f"cleaned_{filename}")
        
        try:
            # Temporarily disable max pixels limit for large images
            original_limit = Image.MAX_IMAGE_PIXELS
            Image.MAX_IMAGE_PIXELS = None
            
            optimize_fake_pixels(input_path, output_path, max_fake_percentage=5.0) # Aim for < 5% fake markers
            
            Image.MAX_IMAGE_PIXELS = original_limit
        except Exception as e:
            print(f"Error analyzing {filename}: {e}")

if __name__ == "__main__":
    input_dir = "input_images"
    output_dir = "fake_pixel_analysis"
    
    if os.path.exists(input_dir):
        process_folder_for_fakes(input_dir, output_dir)
        print("\nAnalysis complete! Check the 'fake_pixel_analysis' folder.")
    else:
        print(f"Input folder '{input_dir}' not found.")
