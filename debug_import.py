try:
    import imquality.brisque
    print("imquality imported setup success")
except Exception as e:
    print(f"imquality failed: {e}")

try:
    from pybrisque import BRISQUE
    print("pybrisque imported setup success")
except Exception as e:
    print(f"pybrisque failed: {e}")
