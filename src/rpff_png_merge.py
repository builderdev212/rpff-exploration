from pathlib import Path
from PIL import Image

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

Path("imgs/.combined/").mkdir(parents=True, exist_ok=True)

for im in imgs:
    with Image.open(f"imgs/{im}.png") as a:
        a = a.convert("RGB")
        combined = Image.new(
            "RGB",
            (a.width * 3, a.height * 3),
            color="white",
        )
        combined.paste(a, (0, 0))

        for n in range(8):
            with Image.open(f"imgs/.reconstructed/{im}_{n}.png") as b:
                combined.paste(b, (a.width * ((n + 1) % 3), a.height * ((n + 1) // 3)))

        combined.save(f"imgs/.combined/{im}.png")
