import os
from PIL import Image


with Image.open("imgs/nature.png") as im:
    src_size = os.path.getsize("imgs/nature.png")
    print(f"Source: imgs/nature.png ({src_size/1000000:.2f}MB)")
    im.convert("RGB").save(f"imgs/nature.tiff", compression="raw")
    conv_size = os.path.getsize("imgs/nature.tiff")
    print(f"Conv: imgs/nature.tiff ({conv_size/1000000:.2f}MB)")
