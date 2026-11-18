import os
from pathlib import Path
import numpy as np
from PIL import Image
import RPFFImagePlugin

imgs = [
    "kodim01",
    "kodim02",
    "kodim03",
    "kodim04",
    "kodim05",
    "kodim06",
    "kodim07",
    "kodim08",
    "kodim09",
    "kodim10",
    "kodim11",
    "kodim12",
    "kodim13",
    "kodim14",
    "kodim15",
    "kodim16",
    "kodim17",
    "kodim18",
    "kodim19",
    "kodim20",
    "kodim21",
    "kodim22",
    "kodim23",
    "kodim24",
]


def psnr(original: Image.Image, decoded: Image.Image) -> float:
    a = np.asarray(original.convert("RGB"), dtype=np.float64)
    b = np.asarray(decoded.convert("RGB"), dtype=np.float64)
    mse = np.mean((a - b) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255**2 / mse)


def convert(
    im: Image.Image, filename: str, src_size: int, mode: str | None = None
) -> tuple[float, float]:
    im = im.convert(mode) if mode else im
    im.save(filename)
    conv_size = os.path.getsize(filename)
    with Image.open(filename) as dec:
        score = psnr(im, dec)
    print(
        f"Conv: {filename} ({conv_size/1000000:.3f}MB) - {conv_size/src_size*100:.3f}% - PSNR ({score})"
    )

    return (conv_size/src_size*100, score)



Path("imgs/.raw/").mkdir(parents=True, exist_ok=True)
Path("imgs/.conv/").mkdir(parents=True, exist_ok=True)
Path("imgs/.reconstructed/").mkdir(parents=True, exist_ok=True)

total_png_compression = 0
total_jpg_compression = 0
total_rpff_compression = 0

total_png_quality = 0
total_jpg_quality = 0
total_rpff_quality = 0

for img in imgs:
    print(f"Image: {img}")
    # Convert to raw TIFF file
    with Image.open(f"imgs/{img}.png") as im:
        im.convert("RGB").save(f"imgs/.raw/{img}.tiff", compression="raw")
        raw_size = os.path.getsize(f"imgs/.raw/{img}.tiff")
        print(f"RAW ({raw_size/1000000:.3f}MB)")

    # Convert to PNG/JPEG/RPFF
    with Image.open(f"imgs/.raw/{img}.tiff") as im:
        # PNG
        png_compression, png_quality = convert(im, f"imgs/.conv/{img}.png", raw_size)
        total_png_compression += png_compression
        total_png_quality += png_quality
        # JPEG
        jpg_compression, jpg_quality = convert(im, f"imgs/.conv/{img}.jpg", raw_size)
        total_jpg_compression += jpg_compression
        total_jpg_quality += jpg_quality
        # RPFF
        rpff_compression, rpff_quality = convert(im, f"imgs/.conv/{img}.rpff", raw_size, mode="YCbCr")
        total_rpff_compression += rpff_compression
        total_rpff_quality += rpff_quality

    # Convert RPFF back to PNG
    with Image.open(f"imgs/.conv/{img}.rpff") as im:
        im.convert("RGB").save(f"imgs/.reconstructed/{img}.png")

print("Averages:")
print(f"PNG:  {total_png_compression/len(imgs):.3f}% - PSNR ({total_png_quality/len(imgs):.1f})")
print(f"JPG:  {total_jpg_compression/len(imgs):.3f}% - PSNR ({total_jpg_quality/len(imgs):.1f})")
print(f"RPFF: {total_rpff_compression/len(imgs):.3f}% - PSNR ({total_rpff_quality/len(imgs):.1f})")