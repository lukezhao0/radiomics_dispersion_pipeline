import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import math

# ------------------------------------------------------------
# Circle areas represent data fractions
#   Full dataset = 100%
#   Outer train  = 80% of full
#   Inner train  = 80% of outer train = 64% of full
# ------------------------------------------------------------

R_full = 1.0
R_outer_train = R_full * math.sqrt(0.80)
R_inner_train = R_outer_train * math.sqrt(0.80)

# Carefully chosen off-center positions that still guarantee containment
C_full = (0.0, 0.0)
C_outer_train = (0.06, 0.04)   # small offset, safely inside full circle
C_inner_train = (0.11, 0.07)   # offset inside outer training circle

def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

# Safety checks
assert dist(C_outer_train, C_full) + R_outer_train <= R_full + 1e-12, \
    "Outer training circle is not fully inside full dataset circle."

assert dist(C_inner_train, C_outer_train) + R_inner_train <= R_outer_train + 1e-12, \
    "Inner training circle is not fully inside outer training circle."

# Plot
fig, ax = plt.subplots(figsize=(7, 7))

circles = [
    Circle(C_full, R_full, facecolor="#dddddd", edgecolor="black", linewidth=2, alpha=0.9),
    Circle(C_outer_train, R_outer_train, facecolor="#8ecae6", edgecolor="black", linewidth=2, alpha=0.85),
    Circle(C_inner_train, R_inner_train, facecolor="#219ebc", edgecolor="black", linewidth=2, alpha=0.9),
]

for c in circles:
    ax.add_patch(c)

# Labels
ax.text(C_full[0], C_full[1] - 1.08, "Full dataset (100%)", ha="center", va="top", fontsize=12)
ax.text(C_outer_train[0], C_outer_train[1] + R_outer_train + 0.05,
        "Outer train (80%)", ha="center", va="bottom", fontsize=12)
ax.text(C_inner_train[0], C_inner_train[1],
        "Inner resampling\n(80% of outer train)", ha="center", va="center",
        fontsize=11, color="white")

# Formatting
ax.set_aspect("equal")
ax.set_xlim(-1.15, 1.15)
ax.set_ylim(-1.15, 1.15)
ax.axis("off")
ax.set_title("Nested 80-20 ML Data Splits", fontsize=14, pad=16)

# Save single figure
plt.savefig("nested_80_20_splits.png", dpi=300, bbox_inches="tight")
plt.show()