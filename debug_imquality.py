import numpy as np
try:
    import imquality.brisque as brisque
    print("imquality imported")
    
    # Test 1: Random RGB Noise
    print("Test 1: Random RGB (100, 100, 3)")
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    print(f"Shape: {img.shape}")
    s = brisque.score(img)
    print(f"Score: {s}")

    # Test 2: PIL Image
    print("\nTest 2: PIL Image RGB")
    from PIL import Image
    pil_img = Image.fromarray(img, 'RGB')
    s2 = brisque.score(pil_img)
    print(f"Score: {s2}")

except Exception as e:
    print(f"FAIL: {e}")
