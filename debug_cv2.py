try:
    import cv2
    print(f"OpenCV Version: {cv2.__version__}")
    try:
        from cv2 import quality
        print("cv2.quality is available!")
        # Test BRISQUE
        import numpy as np
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        score = quality.QualityBRISQUE_compute(img, "brisque_model_live.yml", "brisque_range_live.yml") # Need model files?
        print("Compute called successfully (might fail due to missing model files)")
    except ImportError:
        print("cv2.quality NOT found.")
    except Exception as e:
        print(f"Error using cv2.quality: {e}")
except Exception as e:
    print(f"OpenCV import failed: {e}")
