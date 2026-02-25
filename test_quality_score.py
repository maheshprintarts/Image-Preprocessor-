import os
import sys
from PIL import Image
import numpy as np

# Force using imquality for better PIL support
try:
    import imquality.brisque as brisque
    MODE = 'imquality'
except ImportError:
    print("Error: Could not import imquality.brisque. Is 'image-quality' installed?")
    MODE = None

def test_synthetic():
    print("--- Testing Synthetic 3-Channel Image ---")
    if MODE != 'imquality':
        print("Skipping synthetic test (library not loaded)")
        return

    try:
        # Create a random RGB image (H=100, W=100, C=3)
        fake_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        # Ensure it's passed as expected
        score = brisque.score(fake_img)
        print(f"Synthetic Score: {score}")
    except Exception as e:
        print(f"Synthetic Test Failed: {e}")
    print("-" * 30)

def test_quality(folder):
    if MODE is None:
        return

    if not os.path.exists(folder):
        print(f"Folder {folder} not found.")
        return

    files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"--- Benchmarking BRISQUE Scores for '{folder}' (Using {MODE}) ---")
    print("(Score: 0 = Best Quality, 100+ = Worst Quality)\n")
    
    for filename in files:
        path = os.path.join(folder, filename)
        try:
            # imquality expects RGB image
            img = Image.open(path).convert('RGB')
            img_np = np.array(img)
            
            # Debug shape
            # print(f"DEBUG: Processing {filename} | Mode: {img.mode} | Shape: {img_np.shape}")

            score = brisque.score(img_np)
                
            print(f"Image: {filename}")
            print(f"  -> Score: {score:.2f}")
            print("-" * 30)

        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    folder = "input_images"
    if len(sys.argv) > 1:
        folder = sys.argv[1]
    
    if MODE:
        test_synthetic()
        test_quality(folder)
