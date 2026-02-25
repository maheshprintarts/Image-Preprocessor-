from PIL import Image

def resize_and_set_dpi(input_path, output_path, target_dpi=300):
    try:
        with Image.open(input_path) as img:
            # 1. READ IMAGE INFO
            width_px, height_px = img.size
            current_dpi = img.info.get('dpi', (72, 72))[0] # Default to 72 if missing
            print(current_dpi)
            
            # 2. CALCULATE THE REDUCTION FACTOR 
            # Example: 72 / 300 = 0.24
            factor = current_dpi / target_dpi
            
            # 3. CALCULATE NEW PIXEL DIMENSIONS
            new_width = int(width_px * factor)
            new_height = int(height_px * factor)
            
            print(f"Original: {width_px}x{height_px} pixels @ {current_dpi} DPI")
            print(f"Reduction Factor: {factor:.4f}")
            print(f"New Dimensions: {new_width}x{new_height} pixels")

            # 4. RESIZE THE IMAGE (Reduce the pixels)
            # We use resampling (LANCZOS) for the best quality when shrinking
            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # 5. SAVE WITH NEW DPI
            resized_img.save(output_path, dpi=(target_dpi, target_dpi))
            print(f"Success! Saved to {output_path} @ {target_dpi} DPI")

    except Exception as e:
        print(f"Error: {e}")



# Output: approx 154x115 pixels @ 300 DPI
#resize_and_set_dpi("face2.png", "image_resized_300.png")
