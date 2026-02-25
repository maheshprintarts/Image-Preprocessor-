import cv2
import numpy as np
from PIL import Image

def get_fake_pixel_percentage(pil_img, block_size=21, threshold=10.0):
    # Convert PIL Image to OpenCV Grayscale
    open_cv_image = np.array(pil_img.convert('L'))
    img_float = open_cv_image.astype(np.float32)
    
    # Calculate Variance
    mu = cv2.blur(img_float, (block_size, block_size))
    mu_sq = cv2.blur(img_float**2, (block_size, block_size))
    variance = np.maximum(mu_sq - mu**2, 0)
    std_dev = np.sqrt(variance)
    
    # Count pixels below absolute threshold
    total_pixels = std_dev.size
    fake_pixels = np.sum(std_dev < threshold)
    
    percentage = (fake_pixels / total_pixels) * 100.0
    return percentage

if __name__ == "__main__":
    path = "input_images/30911774-1.jpg"
    img = Image.open(path)
    print(f"Original size: {img.size}")
    
    for scale in [1.0, 0.8, 0.6, 0.48, 0.3, 0.2, 0.1]:
        new_w, new_h = int(img.width * scale), int(img.height * scale)
        test_img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        pct = get_fake_pixel_percentage(test_img)
        print(f"Scale {scale*100:0.0f}% ({new_w}x{new_h}) -> Fake Pixels: {pct:.2f}%")
