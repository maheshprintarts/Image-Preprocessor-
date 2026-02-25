import os
from PIL import Image

def compare_io(input_dir, output_dir):
    print(f"--- Comparing '{input_dir}' vs '{output_dir}' ---")
    
    files = sorted([f for f in os.listdir(output_dir) if f.lower().endswith(('.jpg', '.png'))])
    
    for f in files:
        in_path = os.path.join(input_dir, f)
        out_path = os.path.join(output_dir, f)
        
        if os.path.exists(in_path):
            with Image.open(in_path) as img_in:
                w_in, h_in = img_in.size
            
            with Image.open(out_path) as img_out:
                w_out, h_out = img_out.size
            
            # Calculate reduction
            area_in = w_in * h_in
            area_out = w_out * h_out
            ratio = area_out / area_in if area_in > 0 else 0
            
            print(f"File: {f}")
            print(f"  Input:  {w_in}x{h_in}")
            print(f"  Output: {w_out}x{h_out}")
            print(f"  Size Ratio: {ratio:.2f} ({ratio*100:.1f}%)")
            if w_out == w_in and h_out == h_in:
                 print("  Result: SAME SIZE")
            elif w_out < w_in:
                 print("  Result: REDUCED")
            print("-" * 20)

if __name__ == "__main__":
    compare_io("input_images", "processed_images")
