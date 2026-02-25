from PIL import Image

def set_image_dpi_preserve_pixels(input_path, output_path, target_dpi=300):
    try:
        with Image.open(input_path) as img:
            # 1. READ IMAGE
            width, height = img.size
            original_dpi = img.info.get('dpi', (72, 72))[0]
            
            print(f"--- Step 1: Read Image ---")
            print(f"Original: {width}x{height} pixels @ {original_dpi} DPI")
            print(f"Physical Size: {width/original_dpi:.2f} x {height/original_dpi:.2f} inches")

            # 2. CALCULATE FACTOR (The 'reduction' logic)
            # We don't need to resize pixels. We just change the DPI tag.
            # Changing 72 -> 300 automatically reduces the physical inches.
            
            # 3. SAVE OUTPUT
            # We save the exact same pixel data, but write new DPI info.
            img.save(output_path, dpi=(target_dpi, target_dpi))
            
            print(f"\n--- Step 2: Save Output ---")
            print(f"Saved to: {output_path}")
            
            # 4. VERIFY (Check the new file)
            with Image.open(output_path) as check_img:
                new_dpi = check_img.info.get('dpi', (72, 72))[0]
                print(f"Verification: New DPI is {new_dpi}")
                print(f"Final Pixels: {check_img.size[0]}x{check_img.size[1]} (Unchanged)")
                print(f"New Physical Size: {check_img.size[0]/new_dpi:.2f} inches (Reduced)")

    except Exception as e:
        print(f"Error: {e}")

# Run the function
# Replace 'input.png' with your actual file name
set_image_dpi_preserve_pixels("jacpicture_8071.jpg", "image_1_300dpi.png")
