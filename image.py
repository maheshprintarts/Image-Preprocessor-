import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
from PIL import Image



# 1. Load the image and get pixels (Your existing code)
image_raw = tf.io.read_file("test.jpg")
image_tensor = tf.io.decode_image(image_raw)
shape = tf.shape(image_tensor)
h = int(shape[0]) # 504
w = int(shape[1]) # 388

# 2. Define your target DPI
target_dpi = 300

# 3. Calculate Physical Size (Inches)
width_inches = round(w / target_dpi, 2)
height_inches = round(h / target_dpi, 2)

# 4. Print Results
print(f"--- For Target DPI: {target_dpi} ---")
print(f"Pixel Dimensions: {w} x {h}")
print(f"Physical Size:    {width_inches} x {height_inches} inches")

#width_inches1 = float(width_inches:.2f)
#print(width_inches1)
image_input = Image.open("test.jpg")
# 5. RESIZE THE IMAGE 
# We use resampling (LANCZOS) for the best quality when shrinking
resized_img = image_input.resize((width_inches), (height_inches), Image.Resampling.LANCZOS)

resized_img.save("image_resized.png", dpi=(target_dpi, target_dpi))


