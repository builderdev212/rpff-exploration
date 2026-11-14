import os
from PIL import Image

quality = [0, 25, 50, 75, 100]


with Image.open("imgs/nature.png") as im:
    src_size = os.path.getsize("imgs/nature.png")
    print(f"Source: imgs/nature.png ({int(src_size/1000000)}MB)")
    for q in quality:
        im.convert("RGB").save(f"imgs/nature_{q}.jpg", quality=q)
        print(f"Conv: imgs/nature_{q}.jpg ({os.path.getsize(f"imgs/nature_{q}.jpg")/src_size:.1%})")
