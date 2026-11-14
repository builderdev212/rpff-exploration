import os
import numpy as np
from PIL import Image
import RPFFImagePlugin

quality = [0, 25, 50, 75, 100]


def psnr(original: Image.Image, decoded: Image.Image) -> float:
    a = np.asarray(original.convert("RGB"), dtype=np.float64)
    b = np.asarray(decoded.convert("RGB"), dtype=np.float64)
    mse = np.mean((a - b) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255**2 / mse)


def convert(im: Image.Image, filename: str, src_size: int, mode: str | None = None, **save_args) -> None:
    im = im.convert(mode) if mode else im
    im.save(filename, **save_args)
    conv_size = os.path.getsize(filename)
    with Image.open(filename) as dec:
        score = psnr(im, dec)
    print(f"Conv: {filename} ({conv_size/1000000:.2f}MB) - {conv_size/src_size:.4f}% - PSNR ({score})")


with Image.open("imgs/nature.tiff") as im:
    src_size = os.path.getsize("imgs/nature.tiff")
    print(f"Source: imgs/nature.png ({src_size/1000000:.2f}MB)")
    convert(im, "imgs/nature.png", src_size)
    for q in quality:
        convert(im, f"imgs/nature_{q}.jpg", src_size, quality=q)
    convert(im, "imgs/nature.rpff", src_size, mode="YCbCr")

with Image.open("imgs/nature.rpff") as im:
    im.load()
    print(f"Size: {im.size}\nMode: {im.mode}")
    im.convert("RGB").save("imgs/reconstructed_nature.png")
    src_size = os.path.getsize("imgs/reconstructed_nature.png")
    print(
        f"Reconstructed Source: imgs/reconstructed_nature.png ({src_size/1000000:.2f}MB)"
    )
