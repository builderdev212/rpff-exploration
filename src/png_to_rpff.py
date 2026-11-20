import os
from pathlib import Path
import csv
import numpy as np
from PIL import Image
import RPFFImagePlugin
from skimage.metrics import (
    mean_squared_error,
    peak_signal_noise_ratio,
    structural_similarity,
)

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

jpg_qualitys = [0, 25, 50, 75, 100]

tf_cases = [
    ("DCT", 1),
    ("DCT", 2),
    ("DCT", 3),
    ("DCT", 4),
    ("DST", 1),
    ("DST", 2),
    ("DST", 3),
    ("DST", 4),
]

results_csv = []
avg_results_csv = []


def psnr(original: Image.Image, decoded: Image.Image) -> float:
    a = np.asarray(original.convert("RGB"))
    b = np.asarray(decoded.convert("RGB"))
    return float(peak_signal_noise_ratio(a, b, data_range=255))


def mse(original: Image.Image, decoded: Image.Image) -> float:
    a = np.asarray(original.convert("RGB"))
    b = np.asarray(decoded.convert("RGB"))
    return float(mean_squared_error(a, b))


def ssim(original: Image.Image, decoded: Image.Image) -> float:
    a = np.asarray(original.convert("RGB"))
    b = np.asarray(decoded.convert("RGB"))
    return float(
        structural_similarity(
            a,
            b,
            data_range=255,
            channel_axis=-1,
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False,
        )
    )


def convert(
    im: Image.Image, filename: str, src_size: int, mode: str | None = None, **save_args
) -> tuple[float, np.ndarray]:
    im = im.convert("RGB")
    im = im.convert(mode) if mode else im
    im.save(filename, **save_args)
    conv_size = os.path.getsize(filename)
    with Image.open(filename) as dec:
        mse_score = mse(im, dec)
        psnr_score = psnr(im, dec)
        ssim_score = ssim(im, dec)
    compression = conv_size / src_size * 100

    file_format = Path(filename).suffix.lstrip(".").upper()
    is_rpff = file_format == "RPFF"

    results_csv.append(
        {
            "filename": filename,
            "format": file_format,
            "jpeg_quality": save_args.get("quality", ""),
            "rpff_transform": RPFFImagePlugin.TF_TYPE[0] if is_rpff else "",
            "rpff_transform_type": RPFFImagePlugin.TF_TYPE[1] if is_rpff else "",
            "compression": compression,
            "mse": mse_score,
            "psnr": psnr_score,
            "ssim": ssim_score,
        }
    )

    print(
        f"Conv: {filename} - {compression:.3f}% - MSE ({mse_score}) - PSNR ({psnr_score}) - SSIM ({ssim_score})"
    )

    return (compression, np.array([mse_score, psnr_score, ssim_score]))


Path("imgs/.raw/").mkdir(parents=True, exist_ok=True)
Path("imgs/.conv/").mkdir(parents=True, exist_ok=True)
Path("imgs/.reconstructed/").mkdir(parents=True, exist_ok=True)

total_png_compression = 0
total_jpg_compression = np.array(
    [0 for _ in range(len(jpg_qualitys))], dtype=np.float64
)
total_rpff_compression = np.array([0 for _ in range(len(tf_cases))], dtype=np.float64)

total_png_quality = np.zeros(3, dtype=np.float64)
total_jpg_quality = np.zeros((len(jpg_qualitys), 3), dtype=np.float64)
total_rpff_quality = np.zeros((len(tf_cases), 3), dtype=np.float64)

for img in imgs:
    print(f"Image: {img}")
    # Convert to raw TIFF file
    with Image.open(f"imgs/{img}.png") as im:
        im.convert("RGB").save(f"imgs/.raw/{img}.tiff", compression="raw")
        raw_size = os.path.getsize(f"imgs/.raw/{img}.tiff")
        print(f"RAW ({raw_size/1000000:.3f}MB)")

    # Convert to PNG/JPEG/RPFF
    with Image.open(f"imgs/.raw/{img}.tiff") as im:
        # JPEG
        for n, qual in enumerate(jpg_qualitys):
            jpg_compression, jpg_quality = convert(
                im, f"imgs/.conv/{img}_{qual}.jpg", raw_size, quality=qual
            )
            total_jpg_compression[n] += jpg_compression
            total_jpg_quality[n] += jpg_quality
        # RPFF
        for n, tf_case in enumerate(tf_cases):
            RPFFImagePlugin.TF_TYPE = tf_case
            rpff_compression, rpff_quality = convert(
                im, f"imgs/.conv/{img}_{n}.rpff", raw_size, mode="YCbCr"
            )
            total_rpff_compression[n] += rpff_compression
            total_rpff_quality[n] += rpff_quality

    # Convert RPFF back to PNG
    for n, tf_case in enumerate(tf_cases):
        RPFFImagePlugin.TF_TYPE = tf_case
        with Image.open(f"imgs/.conv/{img}_{n}.rpff") as im:
            im.convert("RGB").save(f"imgs/.reconstructed/{img}_{n}.png")

csv_path = Path("imgs/compression_results.csv")

with csv_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=list(results_csv[0]))
    writer.writeheader()
    writer.writerows(results_csv)

print(f"Saved {len(results_csv)} results to {csv_path}")

print("Averages:")
for n, qual in enumerate(jpg_qualitys):
    avg_results_csv.append(
        {
            "format": "JPG",
            "jpeg_quality": qual,
            "rpff_transform": "",
            "rpff_transform_type": "",
            "compression": total_jpg_compression[n] / len(imgs),
            "mse": total_jpg_quality[n][0] / len(imgs),
            "psnr": total_jpg_quality[n][1] / len(imgs),
            "ssim": total_jpg_quality[n][2] / len(imgs),
        }
    )

    print(
        f"JPG ({qual}%):  {total_jpg_compression[n]/len(imgs):.3f}% - MSE ({total_jpg_quality[n][0]/len(imgs):.1f}) - PSNR ({total_jpg_quality[n][1]/len(imgs):.1f}) - SSIM ({total_jpg_quality[n][2]/len(imgs):.3f})"
    )

for n, tf_case in enumerate(tf_cases):
    avg_results_csv.append(
        {
            "format": "RPFF",
            "jpeg_quality": "",
            "rpff_transform": tf_case[0],
            "rpff_transform_type": tf_case[1],
            "compression": total_rpff_compression[n] / len(imgs),
            "mse": total_rpff_quality[n][0] / len(imgs),
            "psnr": total_rpff_quality[n][1] / len(imgs),
            "ssim": total_rpff_quality[n][2] / len(imgs),
        }
    )

    print(
        f"RPFF {tf_cases[n]}: {total_rpff_compression[n]/len(imgs):.3f}% - MSE ({total_rpff_quality[n][0]/len(imgs):.1f}) - PSNR ({total_rpff_quality[n][1]/len(imgs):.1f}) - SSIM ({total_rpff_quality[n][2]/len(imgs):.3f})"
    )


csv_path = Path("imgs/average_compression_results.csv")

with csv_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=list(avg_results_csv[0]))
    writer.writeheader()
    writer.writerows(avg_results_csv)

print(f"Saved {len(avg_results_csv)} average results to {csv_path}")
