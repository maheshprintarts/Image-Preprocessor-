from PIL import Image
import os
import sys

def get_image_dpi_info(image_path):
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            dpi = img.info.get('dpi')
            
            print(f"--- Processing: {image_path} ---")
            print(f"Dimensions: {width}x{height} pixels")
            
            if dpi:
                dpi_x, dpi_y = dpi
                print(f"DPI: {dpi_x} x {dpi_y}")
                
                # Calculate Physical Size in Inches
                width_inch = width / dpi_x
                height_inch = height / dpi_y
                print(f"Physical Size: {width_inch:.2f}\" x {height_inch:.2f}\"")
                
                # Calculate Diagonal Size (Simple approximation)
                diagonal = (width_inch**2 + height_inch**2)**0.5
                print(f"Diagonal: {diagonal:.2f}\"")
            else:
                print("DPI: Not found (Defaulting to 72 for calculation)")
                width_inch = width / 72
                height_inch = height / 72
                print(f"Physical Size (assuming 72 DPI): {width_inch:.2f}\" x {height_inch:.2f}\"")

            print("-" * 30)
            return dpi

    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) > 1:
        files = sys.argv[1:]
    else:
        # Default test files
        files = ["1.jpg", "2.jpg"]
        # Add files from input_images folder if they exist
        if os.path.exists("input_images"):
            input_files = [os.path.join("input_images", f) for f in os.listdir("input_images") if f.lower().endswith('.jpg')]
            files.extend(input_files[:3]) # Just check first 3

    for f in files:
         if os.path.exists(f):
            get_image_dpi_info(f)
