"""Group-level NOSER figures.

The four clock positions are the group. Each map is rotated so the inclusion
lies on the positive x-axis, then the four maps are averaged. Metrics are
averaged across frequency first, then across the four positions.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import rotate

ROOT = Path(__file__).resolve().parents[1]
NOSER_DIR = ROOT / "results" / "noser lambda= 0.2"
OUT_DIR = NOSER_DIR / "ANOVA_NOSER"
CLOCKS = ("3oclock", "6oclock", "9oclock", "12oclock")
# Mathematical angle of each clock mark. ndimage applies the opposite sense
# on these arrays, so the angle itself is the rotation that brings the mark to +x.
CLOCK_ANGLE = {"3oclock": 0.0, "6oclock": -90.0, "9oclock": 180.0, "12oclock": 90.0}
MATERIALS = ("conductor", "insulator")
TARGET_KHZ = 50.0


def display_image(kind: str, grid: np.ndarray) -> np.ndarray:
    """Stored insulator grids are already flipped positive. Show the raw sign."""
    return grid if kind == "conductor" else -grid


def rotate_image(image: np.ndarray, degrees: float) -> np.ndarray:
    """Rotate about the tank center and keep pixels outside the tank as NaN."""
    if abs(degrees) < 1e-9:
        return image.copy()
    finite = np.isfinite(image)
    filled = np.where(finite, image, 0.0)
    rotated = rotate(filled, degrees, reshape=False, order=1, cval=0.0)
    mask = rotate(finite.astype(float), degrees, reshape=False, order=1, cval=0.0)
    out = np.full(image.shape, np.nan, dtype=float)
    keep = mask > 0.5
    out[keep] = rotated[keep] / np.maximum(mask[keep], 1e-6)
    return out


def mass_angle(image: np.ndarray) -> float:
    """Angle of the positive-mass centroid, degrees, y up and x to the right."""
    mass = np.where(np.isfinite(image), np.clip(image, 0.0, None), 0.0)
    total = float(mass.sum())
    if total <= 0.0:
        return float("nan")
    rows, cols = np.indices(mass.shape)
    center = (mass.shape[0] - 1) / 2.0
    cy = float((rows * mass).sum() / total) - center
    cx = float((cols * mass).sum() / total) - center
    return float(np.degrees(np.arctan2(cy, cx)))


def load() -> dict:
    loaded = {}
    for clock in CLOCKS:
        data = np.load(NOSER_DIR / f"{clock}.npz", allow_pickle=True)
        loaded[clock] = data
    return loaded


def aligned_stack(loaded: dict, kind: str, depth_i: int, freq_i: int) -> np.ndarray:
    frames = []
    for clock in CLOCKS:
        image = display_image(kind, np.asarray(loaded[clock][f"{kind}_grid"][depth_i, freq_i], dtype=float))
        frames.append(rotate_image(image, CLOCK_ANGLE[clock]))
    return np.stack(frames, axis=0)


def save_figure(fig: plt.Figure, path: Path) -> None:
    try:
        fig.savefig(path, dpi=150)
    except OSError:
        path = path.with_name(path.stem + "_new" + path.suffix)
        fig.savefig(path, dpi=150)
    plt.close(fig)
    print("saved", path)


def draw_images(loaded: dict, freq_i: int, walls: np.ndarray) -> None:
    fig, axes = plt.subplots(2, len(walls), figsize=(2.3 * len(walls), 5.2), squeeze=False)
    for row, kind in enumerate(MATERIALS):
        stacks = [aligned_stack(loaded, kind, depth_i, freq_i) for depth_i in range(len(walls))]
        means = [np.nanmean(stack, axis=0) for stack in stacks]
        peak = max(float(np.nanmax(np.abs(image))) for image in means) or 1.0
        for col, (wall, image) in enumerate(zip(walls, means)):
            axis = axes[row, col]
            axis.imshow(image, origin="lower", cmap="RdBu_r", vmin=-peak, vmax=peak)
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(f"wall {wall:.0f} cm", fontsize=9)
            if col == 0:
                axis.set_ylabel(kind, fontsize=9)
            before = [
                mass_angle(display_image(kind, np.asarray(loaded[clock][f"{kind}_grid"][col, freq_i], dtype=float)))
                for clock in CLOCKS
            ]
            after = [mass_angle(frame) for frame in stacks[col]]
            print(f"{kind} wall {wall:.0f} cm  angles before {np.round(before, 1)}  after {np.round(after, 1)}")
    fig.suptitle("NOSER λ=0.2 group mean  |  4 clocks aligned to 3 o'clock  |  50 kHz", fontsize=11)
    fig.tight_layout()
    save_figure(fig, OUT_DIR / "group_mean_50khz.png")


def draw_metrics(loaded: dict, walls: np.ndarray) -> None:
    metrics = {
        "shape relative error": "re_shape",
        "COM error (cm)": "com_error_cm",
        "resolution (cm)": "resolution_cm",
    }
    colors = {"conductor": "C0", "insulator": "C1"}
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.0), squeeze=False)
    for axis, (label, key) in zip(axes[0], metrics.items()):
        for kind in MATERIALS:
            per_clock = []
            for clock in CLOCKS:
                values = np.asarray(loaded[clock][f"{kind}_{key}"], dtype=float)
                per_clock.append(values.mean(axis=1))
            per_clock = np.stack(per_clock, axis=0)
            mean = per_clock.mean(axis=0)
            sem = per_clock.std(axis=0, ddof=1) / np.sqrt(per_clock.shape[0])
            for row in per_clock:
                axis.plot(walls, row, color=colors[kind], alpha=0.25, linewidth=1.0)
            axis.errorbar(
                walls,
                mean,
                yerr=sem,
                fmt="o-",
                color=colors[kind],
                capsize=3,
                label=kind,
            )
        axis.set_xlabel("Distance from wall (cm)")
        axis.set_ylabel(label)
        axis.set_xticks(walls)
        axis.grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("NOSER λ=0.2 group mean ± SEM  |  4 clocks, frequency-averaged", fontsize=11)
    fig.tight_layout()
    save_figure(fig, OUT_DIR / "group_metrics.png")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    loaded = load()
    frequencies = np.asarray(loaded["3oclock"]["frequencies_khz"], dtype=float)
    walls = np.asarray(loaded["3oclock"]["wall_cm"], dtype=float)
    freq_i = int(np.where(np.isclose(frequencies, TARGET_KHZ))[0][0])
    draw_images(loaded, freq_i, walls)
    draw_metrics(loaded, walls)


if __name__ == "__main__":
    main()
