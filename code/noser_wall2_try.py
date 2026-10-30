"""Try a one-electrode shift and a few lambda values on the wall-2 cm NOSER cases."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from general_utils.recon_utils.eit_create import create_eit_object
from general_utils.recon_utils.eit_solver import reconstruct_eit_image

from em_gprn_wang2012 import (
    CLOCKS,
    DATA_ROOT,
    TANK_RADIUS_CM,
    blob_metrics,
    load_by_frequency,
    position_files,
    to_grid,
    truth_xy,
)
from noser_lambda02 import inclusion_signed

BASE = Path(__file__).parent / "assets" / "phantom_noser_lambda0.2.yaml"
OUT = ROOT / "results" / "noser lambda= 0.2"
FOLDERS = {
    "3oclock": DATA_ROOT / "3'o clock",
    "6oclock": DATA_ROOT / "6'o clock",
    "9oclock": DATA_ROOT / "9'o clock",
    "12oclock": DATA_ROOT / "12'o clock",
}


def solver_for(rotate: int, lamb: float):
    config = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    config["eit"]["electrodes"]["rotate_el_clockwise"] = int(rotate)
    config["eit"]["reconstruction"]["lambda"] = float(lamb)
    path = Path(__file__).parent / "assets" / "_noser_wall_try.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    solver, _ = create_eit_object(str(path))
    mesh = {
        "node": np.asarray(solver.mesh["node"], dtype=float) * TANK_RADIUS_CM,
        "element": np.asarray(solver.mesh["element"]),
    }
    return solver, mesh, mesh["node"][mesh["element"]].mean(axis=1)


def main() -> None:
    cache = {}
    loaded = {}
    for clock in CLOCKS:
        files = position_files(FOLDERS[clock])
        loaded[clock] = {}
        for kind in ("conductor", "insulator"):
            loaded[clock][kind] = {
                "meas": load_by_frequency(files[kind][2])[50.0],
                "ref": load_by_frequency(files[f"{kind}_ref"])[50.0],
            }
    print("rotate lambda  clock kind  COM  res", flush=True)
    best = None
    panels = {}
    for rotate in (14, 13, 15):
        for lamb in (0.2, 0.05, 0.5):
            solver, mesh, centroids = solver_for(rotate, lamb)
            coms = []
            for clock, angle in CLOCKS.items():
                target = truth_xy(angle, TANK_RADIUS_CM - 2.0)
                for kind in ("conductor", "insulator"):
                    image = reconstruct_eit_image(
                        loaded[clock][kind]["meas"], solver, loaded[clock][kind]["ref"]
                    )
                    sigma = np.real(np.asarray(image.data, dtype=float)).ravel()
                    signed = inclusion_signed(kind, sigma)
                    blob = blob_metrics(signed, centroids, target)
                    com = blob["com_error_cm"]
                    coms.append(com)
                    print(
                        f"{rotate:7d} {lamb:6.2f}  {clock:8s} {kind:10s} "
                        f"{com:5.2f}  {blob['resolution_cm']:5.2f}",
                        flush=True,
                    )
                    panels[(rotate, lamb, clock, kind)] = to_grid(mesh, sigma, cache)
            mean_com = float(np.nanmean(coms))
            print(f"  mean COM {mean_com:.2f}", flush=True)
            if best is None or mean_com < best[0]:
                best = (mean_com, rotate, lamb)

    print(f"best mean COM {best[0]:.2f} at rotate={best[1]} lambda={best[2]}", flush=True)
    if not panels:
        return
    chosen_rotate, chosen_lambda = best[1], best[2]
    fig, axes = plt.subplots(4, 2, figsize=(6.4, 12.0))
    for row, clock in enumerate(CLOCKS):
        target = truth_xy(CLOCKS[clock], TANK_RADIUS_CM - 2.0)
        for col, kind in enumerate(("conductor", "insulator")):
            axis = axes[row, col]
            image = panels[(chosen_rotate, chosen_lambda, clock, kind)]
            peak = np.nanmax(np.abs(image)) or 1.0
            axis.imshow(image, origin="lower", cmap="RdBu_r", vmin=-peak, vmax=peak, extent=(-10, 10, -10, 10))
            axis.plot(target[0], target[1], "k+", markersize=8, markeredgewidth=1.2)
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_title(f"{clock} {kind}", fontsize=9)
    fig.suptitle(
        f"NOSER wall 2 cm, 50 kHz, rotate {chosen_rotate}, λ={chosen_lambda:g}",
        fontsize=11,
    )
    fig.tight_layout()
    path = OUT / f"wall2_rotate{chosen_rotate}_lambda{chosen_lambda:g}.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print("saved", path, flush=True)


if __name__ == "__main__":
    main()
