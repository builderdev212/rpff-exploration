import pandas as pd
import matplotlib.pyplot as plt
from adjustText import adjust_text

vals = pd.read_csv("imgs/average_compression_results.csv")
vals.columns = vals.columns.str.strip()
vals["format"] = vals["format"].str.strip()
vals["rpff_transform"] = vals["rpff_transform"].str.strip()

# Create Graph
fig, ax = plt.subplots()

# JPEG
jpg = vals[vals["format"] == "JPG"].sort_values("compression")
ax.plot(
    jpg["compression"],
    jpg["ssim"],
    "o-",
    label="JPEG",
    color="tab:blue",
)

# RPFF points and labels
rpff = vals[vals["format"] == "RPFF"]
colors = {"DCT": "tab:orange", "DST": "tab:green"}
texts = []

for transform, group in rpff.groupby("rpff_transform"):
    ax.scatter(
        group["compression"],
        group["ssim"],
        label=f"RPFF {transform}",
        color=colors[transform],
    )

    for _, row in group.iterrows():
        texts.append(
            ax.text(
                row["compression"],
                row["ssim"],
                f"({transform}, {int(row['rpff_transform_type'])})",
                fontsize=9,
            )
        )

ax.set_xlabel("Compressed size (% of raw size)")
ax.set_ylabel("SSIM")
ax.set_title("Compression vs. Image Quality")
ax.grid(True)
ax.legend()
fig.tight_layout()

# Separate labels and draw connecting lines
adjust_text(
    texts,
    ax=ax,
    x=rpff["compression"].to_numpy(),
    y=rpff["ssim"].to_numpy(),
    min_arrow_len=0,
    arrowprops=dict(
        arrowstyle="-",
        color="gray",
        lw=0.5,
        shrinkA=0,
        shrinkB=0,
    ),
)

plt.show()
