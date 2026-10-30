"""NOSER (Gauss-Newton, LM) reconstructions at lambda = 0.2.

Same phantom set as the EM results: four clock positions, conductor and
insulator, wall distances 2/4/6/8/10 cm, every recorded frequency.
Separate reference per position. Figures and metrics match that run.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from general_utils.recon_utils.eit_create import create_eit_object
from general_utils.recon_utils.eit_solver import reconstruct_eit_image

from em_gprn_wang2012 import (
    CLOCKS,
    DATA_ROOT,
    GRID_N,
    TANK_RADIUS_CM,
    blob_metrics,
    load_by_frequency,
    position_files,
    relative_error,
    to_grid,
    true_map,
    truth_xy,
)

CONFIG = Path(__file__).parent / "assets" / "phantom_noser_lambda0.2.yaml"
OUT_DIR = ROOT / "results" / "noser lambda= 0.2"


def inclusion_signed(kind: str, sigma: np.ndarray) -> np.ndarray:
    """Conductors increase conductivity. Insulators decrease it, so flip them."""
    return sigma if kind == "conductor" else -sigma


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    solver, _ = create_eit_object(str(CONFIG))
    mesh = {
        "node": np.asarray(solver.mesh["node"], dtype=float) * TANK_RADIUS_CM,
        "element": np.asarray(solver.mesh["element"]),
    }
    centroids = mesh["node"][mesh["element"]].mean(axis=1)
    cache = {}
    folders = {
        "3oclock": DATA_ROOT / "3'o clock",
        "6oclock": DATA_ROOT / "6'o clock",
        "9oclock": DATA_ROOT / "9'o clock",
        "12oclock": DATA_ROOT / "12'o clock",
    }

    for clock, angle in CLOCKS.items():
        files = position_files(folders[clock])
        depths = sorted(set(files["conductor"]) & set(files["insulator"]))
        refs = {
            "conductor": load_by_frequency(files["conductor_ref"]),
            "insulator": load_by_frequency(files["insulator_ref"]),
        }
        loaded = {
            "conductor": {depth: load_by_frequency(path) for depth, path in files["conductor"].items()},
            "insulator": {depth: load_by_frequency(path) for depth, path in files["insulator"].items()},
        }
        frequencies = sorted(loaded["conductor"][depths[0]])
        freq_index = frequencies.index(50.0) if 50.0 in frequencies else len(frequencies) // 2
        stored = {
            "frequencies_khz": np.asarray(frequencies),
            "wall_cm": np.asarray(depths),
            "r_center_cm": np.asarray([TANK_RADIUS_CM - depth for depth in depths]),
        }
        grids = {}
        metrics = {}
        for kind in ("conductor", "insulator"):
            sigma_rows, grid_rows = [], []
            com_rows, res_rows = [], []
            re_raw_rows, re_shape_rows, amp_rows = [], [], []
            for depth in depths:
                r_center = TANK_RADIUS_CM - depth
                target = truth_xy(angle, r_center)
                truth = true_map(centroids, angle, r_center)
                sigma_row, grid_row = [], []
                com_row, res_row = [], []
                re_raw_row, re_shape_row, amp_row = [], [], []
                for freq in frequencies:
                    image = reconstruct_eit_image(
                        loaded[kind][depth][freq], solver, refs[kind][freq]
                    )
                    sigma = np.real(np.asarray(image.data, dtype=float)).ravel()
                    signed = inclusion_signed(kind, sigma)
                    grid = to_grid(mesh, signed, cache)
                    blob = blob_metrics(signed, centroids, target)
                    raw_re, shape_re = relative_error(signed, truth)
                    sigma_row.append(sigma)
                    grid_row.append(grid)
                    com_row.append(blob["com_error_cm"])
                    res_row.append(blob["resolution_cm"])
                    re_raw_row.append(raw_re)
                    re_shape_row.append(shape_re)
                    amp_row.append(blob["max_amplitude"])
                sigma_rows.append(sigma_row)
                grid_rows.append(grid_row)
                com_rows.append(com_row)
                res_rows.append(res_row)
                re_raw_rows.append(re_raw_row)
                re_shape_rows.append(re_shape_row)
                amp_rows.append(amp_row)
                print(
                    f"{clock} {kind} wall={depth} cm "
                    f"RE={re_shape_row[freq_index]:.3f} COM={com_row[freq_index]:.2f} cm "
                    f"res={res_row[freq_index]:.2f} cm"
                )
            grids[kind] = np.asarray(grid_rows)
            metrics[kind] = {
                "com": np.asarray(com_rows),
                "resolution": np.asarray(res_rows),
                "re_raw": np.asarray(re_raw_rows),
                "re_shape": np.asarray(re_shape_rows),
                "amplitude": np.asarray(amp_rows),
            }
            stored[kind] = np.asarray(sigma_rows, dtype=object)
            stored[f"{kind}_grid"] = grids[kind]
            stored[f"{kind}_com_error_cm"] = metrics[kind]["com"]
            stored[f"{kind}_resolution_cm"] = metrics[kind]["resolution"]
            stored[f"{kind}_re_raw"] = metrics[kind]["re_raw"]
            stored[f"{kind}_re_shape"] = metrics[kind]["re_shape"]
            stored[f"{kind}_max_amplitude"] = metrics[kind]["amplitude"]

        r_center = stored["r_center_cm"]
        fig = plt.figure(figsize=(2.4 * len(depths) + 3.4, 5.8))
        grid_spec = fig.add_gridspec(2, len(depths) + 1, width_ratios=[1] * len(depths) + [1.25])
        for row, kind in enumerate(("conductor", "insulator")):
            for col, depth in enumerate(depths):
                axis = fig.add_subplot(grid_spec[row, col])
                image = grids[kind][col, freq_index]
                if kind == "insulator":
                    image = -image
                peak = np.nanmax(np.abs(image)) or 1.0
                axis.imshow(image, origin="lower", cmap="RdBu_r", vmin=-peak, vmax=peak)
                axis.set_xticks([])
                axis.set_yticks([])
                if row == 0:
                    axis.set_title(f"wall {depth} cm\nr={TANK_RADIUS_CM - depth:.0f} cm", fontsize=8)
                if col == 0:
                    axis.set_ylabel(kind, fontsize=9)
            axis = fig.add_subplot(grid_spec[row, -1])
            for freq_i in range(len(frequencies)):
                axis.plot(r_center, metrics[kind]["re_shape"][:, freq_i], color="0.75", linewidth=0.8)
            axis.plot(r_center, metrics[kind]["re_shape"].mean(axis=1), "o-", color="C0")
            axis.set_xlabel("r from center (cm)")
            axis.set_ylabel("shape RE")
            axis.set_title(f"{kind} RE")
            axis.grid(True, alpha=0.3)
            axis.invert_xaxis()
        fig.suptitle(f"{clock}  |  NOSER  |  λ=0.2  |  50 kHz", fontsize=11)
        fig.tight_layout()
        figure_path = OUT_DIR / f"{clock}.png"
        try:
            fig.savefig(figure_path, dpi=140)
        except OSError:
            figure_path = OUT_DIR / f"{clock}_new.png"
            fig.savefig(figure_path, dpi=140)
        plt.close(fig)

        analysis = plt.figure(figsize=(10.5, 6.4))
        analysis_axes = analysis.subplots(2, 2)
        series = (
            ("re_shape", "shape RE"),
            ("com", "center-of-mass error (cm)"),
            ("resolution", "resolution (cm)"),
            ("amplitude", "maximum amplitude"),
        )
        for axis, (key, label) in zip(analysis_axes.ravel(), series):
            for kind, color in (("conductor", "C0"), ("insulator", "C1")):
                for freq_i in range(len(frequencies)):
                    axis.plot(r_center, metrics[kind][key][:, freq_i], color=color, alpha=0.25, linewidth=0.7)
                axis.plot(r_center, metrics[kind][key].mean(axis=1), "o-", color=color, label=kind)
            axis.set_xlabel("r from center (cm)")
            axis.set_ylabel(label)
            axis.grid(True, alpha=0.3)
            axis.invert_xaxis()
            axis.legend(fontsize=8)
        analysis.suptitle(f"{clock}  |  NOSER λ=0.2", fontsize=11)
        analysis.tight_layout()
        analysis_path = OUT_DIR / f"{clock}_analysis.png"
        try:
            analysis.savefig(analysis_path, dpi=140)
        except OSError:
            analysis_path = OUT_DIR / f"{clock}_analysis_new.png"
            analysis.savefig(analysis_path, dpi=140)
        plt.close(analysis)
        np.savez(OUT_DIR / f"{clock}.npz", **stored)
        print("saved", figure_path)


if __name__ == "__main__":
    main()
