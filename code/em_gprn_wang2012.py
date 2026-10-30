"""
EM-GPRN reconstructions for the phantom-data tank (Wang-style two-stage solver).

Tank: diameter 20 cm, 16 electrodes, adjacent drive.
delta_U = (-1)^q (U_mea - U_ref), q=1 conductor, q=2 insulator.
gamma = 0.35, omega^2 = 0.01, L = I.
2 gradient-projection steps, then 5 reduced-Newton steps.
Non-negative delta_sigma.

pyEIT's forward model is the point-electrode adjacent model. Its complete
electrode model is not implemented, so the Jacobian comes from that model.
The mesh is graded: about 2.5 mm under the electrodes, about 3.6 mm on the
outer ring, and about 1 cm at the center. Each Jacobian column is multiplied
by its triangle area and then divided by the largest absolute entry, so the
small boundary triangles do not set the scale for the large interior ones.
Signs are kept. A min-max map onto [0, 1] would erase the sign of the sensitivity.

Filename centimetres are distance from the wall, as in the earlier depth
study: r_center = 10 - label. So 10 cm in the file name is the center.
"""
from __future__ import annotations

import pickle
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from general_utils.recon_utils.pyeit import mesh as mesh_creator
from general_utils.recon_utils.pyeit.eit.fem import Forward
from general_utils.recon_utils.pyeit.eit.interp2d import meshgrid, weight_sigmod
from general_utils.recon_utils.pyeit.eit.utils import eit_scan_lines
from general_utils.recon_utils.pyeit.mesh.shape import circle, fix_points_fd
from general_utils.recon_utils.pyeit.mesh.utils import check_order

DATA_ROOT = ROOT / "data" / "phantom"
OUT_DIR = ROOT / "workspace" / "em_trials"

TANK_RADIUS_CM = 10.0
N_ELECTRODES = 16
GAMMA = 0.35
OMEGA2 = 0.01
N_GP = 2
N_NEWTON = 5
GRID_N = 64
INCLUSION_RADIUS_CM = 1.5
# Same electrode frame used for the earlier 3 o'clock / 6 o'clock calibration.
ROTATE_CLOCKWISE = 14
REVERSE_ELECTRODES = True

CLOCKS = {
    "3oclock": 0.0,
    "6oclock": -90.0,
    "9oclock": 180.0,
    "12oclock": 90.0,
}


def load_by_frequency(path: Path) -> dict[float, np.ndarray]:
    """Mean of every frame from every repeat, one vector per recorded frequency."""
    with open(path, "rb") as handle:
        raw = pickle.load(handle)
    frames: dict[float, list] = {}
    for block in raw:
        freq = float(block["freq"]) / 1000.0
        amp = np.asarray(block["amp"]["data"], dtype=float)
        if amp.ndim == 1:
            amp = amp.reshape(1, -1)
        frames.setdefault(freq, []).append(amp)
    return {freq: np.nanmean(np.vstack(chunks), axis=0) for freq, chunks in frames.items()}


def target_edge_cm(points, electrodes):
    """2.0–2.2 mm on the wall and electrodes, about 1.2 cm at the center."""
    points = np.atleast_2d(np.asarray(points, dtype=float))
    wall = TANK_RADIUS_CM - np.linalg.norm(points, axis=1)
    delta = points[:, None, :] - electrodes[None, :, :]
    nearest = np.sqrt(np.min(np.sum(delta * delta, axis=2), axis=1))
    radial = np.where(
        wall <= 1.2,
        0.22,
        0.22 + (1.20 - 0.22) * np.clip((wall - 1.2) / 8.8, 0.0, 1.0) ** 1.4,
    )
    blend = np.clip(nearest / 0.55, 0.0, 1.0)
    return 0.20 + (radial - 0.20) * blend


def graded_nodes(electrodes):
    """Hexagonal lattice, thinned so the rim and the 16 electrodes stay fine."""
    spacing = 0.12
    ys = np.arange(-TANK_RADIUS_CM, TANK_RADIUS_CM + spacing, spacing * np.sqrt(3.0) / 2.0)
    xs = np.arange(-TANK_RADIUS_CM, TANK_RADIUS_CM + spacing, spacing)
    xx, yy = np.meshgrid(xs, ys)
    xx = xx.copy()
    xx[1::2, :] += spacing / 2.0
    lattice = np.column_stack((xx.ravel(), yy.ravel()))
    lattice = lattice[np.linalg.norm(lattice, axis=1) <= TANK_RADIUS_CM - 0.04]

    n_boundary = int(np.ceil(2.0 * np.pi * TANK_RADIUS_CM / 0.22))
    theta = np.linspace(0.0, 2.0 * np.pi, n_boundary, endpoint=False)
    boundary = np.column_stack((TANK_RADIUS_CM * np.cos(theta), TANK_RADIUS_CM * np.sin(theta)))
    boundary_gap = np.linalg.norm(boundary[:, None, :] - electrodes[None, :, :], axis=2).min(axis=1)
    boundary = boundary[boundary_gap > 0.16]

    candidates = np.vstack((boundary, lattice))
    order = np.argsort(target_edge_cm(candidates, electrodes), kind="mergesort")
    candidates = candidates[order]
    allowance = 0.72 * target_edge_cm(candidates, electrodes)

    kept = np.empty((electrodes.shape[0] + candidates.shape[0], 2), dtype=float)
    kept[: electrodes.shape[0]] = electrodes
    n_kept = electrodes.shape[0]
    for point, limit in zip(candidates, allowance):
        distance = np.linalg.norm(kept[:n_kept] - point, axis=1).min()
        if distance >= limit:
            kept[n_kept] = point
            n_kept += 1
    return kept[:n_kept]


def triangle_area(nodes, triangles):
    a = nodes[triangles[:, 0]]
    b = nodes[triangles[:, 1]]
    c = nodes[triangles[:, 2]]
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])
    return 0.5 * np.abs(cross)


def build_workspace():
    """Graded 20 cm mesh, adjacent Jacobian, area-weighted then normalized."""
    electrodes = fix_points_fd(circle, n_el=N_ELECTRODES) * TANK_RADIUS_CM
    nodes = graded_nodes(electrodes)
    triangles = check_order(nodes, Delaunay(nodes).simplices.copy())
    centroids = nodes[triangles].mean(axis=1)
    triangles = triangles[np.linalg.norm(centroids, axis=1) <= TANK_RADIUS_CM + 1e-8]
    centroids = nodes[triangles].mean(axis=1)
    areas = triangle_area(nodes, triangles)
    perm = np.ones(triangles.shape[0])
    mesh = {"element": triangles, "node": nodes, "perm": perm}
    el_pos = np.arange(N_ELECTRODES)
    el_pos = np.roll(el_pos, -ROTATE_CLOCKWISE)
    if REVERSE_ELECTRODES:
        el_pos = el_pos[::-1]

    lengths = []
    for offset in ((0, 1), (1, 2), (2, 0)):
        lengths.append(np.linalg.norm(nodes[triangles[:, offset[0]]] - nodes[triangles[:, offset[1]]], axis=1))
    edge = np.column_stack(lengths)
    nearest = np.linalg.norm(centroids[:, None, :] - electrodes[None, :, :], axis=2).min(axis=1)
    radius = np.linalg.norm(centroids, axis=1)
    print(
        f"elements={triangles.shape[0]} nodes={nodes.shape[0]} "
        f"electrode_edge_mm={np.median(edge[nearest <= 0.45]) * 10:.2f} "
        f"ring_edge_mm={np.median(edge[radius >= 8.0]) * 10:.2f} "
        f"center_edge_mm={np.median(edge[radius <= 3.0]) * 10:.2f}"
    )

    forward = Forward(mesh, el_pos)
    solved = forward.solve_eit(
        eit_scan_lines(N_ELECTRODES, 1),
        step=1,
        perm=perm,
        parser="fmmu",
    )
    jacobian = np.asarray(solved.jac, dtype=float) * areas[np.newaxis, :]
    scale = np.max(np.abs(jacobian))
    if scale == 0:
        raise RuntimeError("Jacobian is all zeros")
    jacobian = jacobian / scale
    centroids = nodes[triangles].mean(axis=1)
    return mesh, jacobian, centroids


def build_uniform_workspace():
    """First EM-GPRN mesh: uniform triangles, Jacobian scaled by its largest entry."""
    mesh, el_pos = mesh_creator.create(N_ELECTRODES, h0=0.1)
    mesh["node"] = np.asarray(mesh["node"], dtype=float) * TANK_RADIUS_CM
    el_pos = np.roll(np.asarray(el_pos), -ROTATE_CLOCKWISE)
    if REVERSE_ELECTRODES:
        el_pos = el_pos[::-1]
    perm = np.ones(mesh["element"].shape[0])
    mesh["perm"] = perm
    forward = Forward(mesh, el_pos)
    solved = forward.solve_eit(
        eit_scan_lines(N_ELECTRODES, 1),
        step=1,
        perm=perm,
        parser="fmmu",
    )
    jacobian = np.asarray(solved.jac, dtype=float)
    jacobian = jacobian / np.max(np.abs(jacobian))
    centroids = mesh["node"][mesh["element"]].mean(axis=1)
    print(f"uniform elements={jacobian.shape[1]} measurements={jacobian.shape[0]}")
    return mesh, jacobian, centroids


def delta_u(u_mea: np.ndarray, u_ref: np.ndarray, kind: str) -> np.ndarray:
    """(-1)^q (U_mea - U_ref). q=1 conductor, q=2 insulator."""
    q = 1 if kind == "conductor" else 2
    return ((-1) ** q) * (u_mea - u_ref)


def sensitivity_weights(jacobian: np.ndarray, areas: np.ndarray) -> np.ndarray:
    """Diagonal of L. Penalty is inversely proportional to sensitivity per unit area.

    The stored Jacobian has already been multiplied by triangle area, so the
    column norm is divided by that area. Large central triangles are then not
    treated as more sensitive than the boundary. W is scaled so its median is 1.
    """
    sensitivity = np.linalg.norm(jacobian, axis=0) / np.maximum(areas, 1e-12)
    floor = max(float(np.median(sensitivity)) * 1e-6, 1e-12)
    sensitivity = np.maximum(sensitivity, floor)
    return np.median(sensitivity) / sensitivity


def penalize_outer_layer(weights: np.ndarray, centroids: np.ndarray, radius_cm: float = 9.0, target: float = 3.0) -> np.ndarray:
    """Invert the outer-layer weights so the wall is heavily penalized.

    Interior weights are left as they are, so a central reconstruction that
    uses the original weights is unchanged. Outer W is replaced by median(W)^2 / W
    and then lifted until the outer median equals target.
    """
    updated = np.array(weights, copy=True)
    outer = np.linalg.norm(centroids, axis=1) >= radius_cm
    reference = float(np.median(updated))
    updated[outer] = (reference ** 2) / np.maximum(updated[outer], 1e-6)
    outer_median = float(np.median(updated[outer]))
    if outer_median > 0:
        updated[outer] *= target / outer_median
    return updated


def apply_hessian(jacobian: np.ndarray, state: np.ndarray, omega2: float, penalty: np.ndarray) -> np.ndarray:
    """H x with H = J^T J / omega^2 + diag(penalty). penalty_i = gamma * W_i^2."""
    return jacobian.T @ (jacobian @ state) / omega2 + penalty * state


def solve_reduced(jacobian_free: np.ndarray, rhs: np.ndarray, omega2: float, penalty_free: np.ndarray) -> np.ndarray:
    """Solve (J_f^T J_f / omega^2 + diag(penalty)) s = rhs.

    The penalty is diagonal, so the Woodbury identity still reduces the
    factorization to the measurement-sized matrix.
    """
    ainv = 1.0 / penalty_free
    gram = (jacobian_free * ainv[np.newaxis, :]) @ jacobian_free.T
    system = gram + omega2 * np.eye(gram.shape[0])
    v = ainv * rhs
    correction = np.linalg.solve(system, jacobian_free @ v)
    return v - ainv * (jacobian_free.T @ correction)


def em_gprn(
    jacobian: np.ndarray,
    delta_voltage: np.ndarray,
    gamma: float = GAMMA,
    omega2: float = OMEGA2,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """2 gradient-projection steps, then 5 reduced-Newton steps. x >= 0.

    Objective:
        (1 / (2 omega^2)) ||J x - d||^2 + (gamma / 2) ||L x||^2
    with L = diag(W). W defaults to 1, which recovers L = I.
    """
    if weights is None:
        weights = np.ones(jacobian.shape[1])
    penalty = gamma * np.square(weights)
    data_grad = jacobian.T @ delta_voltage / omega2
    x = np.zeros(jacobian.shape[1])

    def gradient(state: np.ndarray) -> np.ndarray:
        return apply_hessian(jacobian, state, omega2, penalty) - data_grad

    for _ in range(N_GP):
        grad = gradient(x)
        curvature = float(grad @ apply_hessian(jacobian, grad, omega2, penalty))
        if curvature <= 0:
            break
        step = float(grad @ grad) / curvature
        x = np.maximum(0.0, x - step * grad)

    for _ in range(N_NEWTON):
        free = np.flatnonzero(x > 1e-12)
        if free.size == 0:
            break
        grad = gradient(x)
        step = solve_reduced(jacobian[:, free], -grad[free], omega2, penalty[free])
        x[free] = np.maximum(0.0, x[free] + step)
    return x


def true_map(centroids: np.ndarray, angle_deg: float, r_center_cm: float) -> np.ndarray:
    theta = np.deg2rad(angle_deg)
    center = np.array([r_center_cm * np.cos(theta), r_center_cm * np.sin(theta)])
    distance = np.linalg.norm(centroids - center, axis=1)
    truth = np.zeros(centroids.shape[0])
    truth[distance <= INCLUSION_RADIUS_CM] = 1.0
    return truth


def relative_error(calculated: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    denom = np.linalg.norm(truth)
    if denom == 0:
        return np.nan, np.nan
    raw = float(np.linalg.norm(calculated - truth) / denom)
    scale_denom = float(calculated @ calculated)
    scale = float(calculated @ truth) / scale_denom if scale_denom > 0 else 0.0
    shaped = float(np.linalg.norm(scale * calculated - truth) / denom)
    return raw, shaped


def blob_metrics(values: np.ndarray, centroids: np.ndarray, truth_xy: np.ndarray) -> dict:
    positive = np.maximum(values, 0.0)
    total = float(positive.sum())
    if total <= 0:
        return {
            "com_x": np.nan,
            "com_y": np.nan,
            "com_error_cm": np.nan,
            "max_amplitude": 0.0,
            "resolution_cm": np.nan,
        }
    com = (centroids * positive[:, None]).sum(axis=0) / total
    peak = float(positive.max())
    active = positive >= 0.5 * peak
    if not np.any(active):
        resolution = np.nan
    else:
        offset = centroids[active] - com
        weights = positive[active]
        resolution = float(np.sqrt(np.sum(weights * np.sum(offset**2, axis=1)) / np.sum(weights)))
    return {
        "com_x": float(com[0]),
        "com_y": float(com[1]),
        "com_error_cm": float(np.linalg.norm(com - truth_xy)),
        "max_amplitude": peak,
        "resolution_cm": resolution,
    }


def to_grid(mesh: dict, element_values: np.ndarray, grid_cache: dict) -> np.ndarray:
    if not grid_cache:
        nodes = mesh["node"]
        triangles = mesh["element"]
        xg, yg, outside = meshgrid(nodes, n=GRID_N)
        xy = nodes[triangles].mean(axis=1)
        xyi = np.column_stack((xg.ravel(), yg.ravel()))
        grid_cache["w"] = weight_sigmod(xy, xyi)
        grid_cache["mask"] = np.asarray(outside).reshape(xg.shape)
        grid_cache["shape"] = xg.shape
    image = grid_cache["w"].T @ element_values
    image = image.reshape(grid_cache["shape"])
    image = image.astype(float)
    image[grid_cache["mask"]] = np.nan
    return image


def position_files(folder: Path) -> dict:
    found = {"conductor_ref": None, "insulator_ref": None, "conductor": {}, "insulator": {}}
    for path in folder.glob("*.pkl"):
        name = path.name.lower()
        depth = re.search(r"(\d+)\s*cm", name)
        if "ref" in name and "cond" in name:
            found["conductor_ref"] = path
        elif "ref" in name and "ins" in name:
            found["insulator_ref"] = path
        elif depth and "cond" in name:
            found["conductor"][int(depth.group(1))] = path
        elif depth and "ins" in name:
            found["insulator"][int(depth.group(1))] = path
    return found


def truth_xy(angle_deg: float, r_center_cm: float) -> np.ndarray:
    theta = np.deg2rad(angle_deg)
    return np.array([r_center_cm * np.cos(theta), r_center_cm * np.sin(theta)])


def reconstruct_position(clock: str, angle: float, mesh, jacobian, centroids, grid_cache) -> None:
    folder = {
        "3oclock": DATA_ROOT / "3'o clock",
        "6oclock": DATA_ROOT / "6'o clock",
        "9oclock": DATA_ROOT / "9'o clock",
        "12oclock": DATA_ROOT / "12'o clock",
    }[clock]
    files = position_files(folder)
    if files["conductor_ref"] is None or files["insulator_ref"] is None:
        raise FileNotFoundError(f"Missing reference in {folder}")

    loaded = {
        "conductor": {depth: load_by_frequency(path) for depth, path in files["conductor"].items()},
        "insulator": {depth: load_by_frequency(path) for depth, path in files["insulator"].items()},
    }
    frequencies = sorted(next(iter(loaded["conductor"].values())))
    depths = sorted(set(files["conductor"]) & set(files["insulator"]))
    print(f"{clock}: {len(frequencies)} frequencies, depths {depths}")

    refs = {
        "conductor": load_by_frequency(files["conductor_ref"]),
        "insulator": load_by_frequency(files["insulator_ref"]),
    }
    images = {}
    grids = {}
    metrics = {
        "frequency_khz": frequencies,
        "wall_cm": depths,
        "r_center_cm": [TANK_RADIUS_CM - depth for depth in depths],
    }
    for kind in ("conductor", "insulator"):
        stack = []
        grid_stack = []
        re_raw = []
        re_shape = []
        com_err = []
        max_amp = []
        resolution = []
        for depth in depths:
            r_center = TANK_RADIUS_CM - depth
            truth = true_map(centroids, angle, r_center)
            target = truth_xy(angle, r_center)
            raw_row = []
            grid_row = []
            re_raw_row = []
            re_shape_row = []
            com_row = []
            amp_row = []
            res_row = []
            for freq in frequencies:
                measured = loaded[kind][depth][freq]
                voltage = delta_u(measured, refs[kind][freq], kind)
                sigma = em_gprn(jacobian, voltage)
                raw_row.append(sigma)
                grid_row.append(to_grid(mesh, sigma, grid_cache))
                raw_re, shape_re = relative_error(sigma, truth)
                blob = blob_metrics(sigma, centroids, target)
                re_raw_row.append(raw_re)
                re_shape_row.append(shape_re)
                com_row.append(blob["com_error_cm"])
                amp_row.append(blob["max_amplitude"])
                res_row.append(blob["resolution_cm"])
            stack.append(raw_row)
            grid_stack.append(grid_row)
            re_raw.append(re_raw_row)
            re_shape.append(re_shape_row)
            com_err.append(com_row)
            max_amp.append(amp_row)
            resolution.append(res_row)
            print(f"  {kind} wall={depth} cm done")
        images[kind] = np.asarray(stack)  # depth, freq, elem
        grids[kind] = np.asarray(grid_stack)
        metrics[f"{kind}_re_raw"] = np.asarray(re_raw)
        metrics[f"{kind}_re_shape"] = np.asarray(re_shape)
        metrics[f"{kind}_com_error_cm"] = np.asarray(com_err)
        metrics[f"{kind}_max_amplitude"] = np.asarray(max_amp)
        metrics[f"{kind}_resolution_cm"] = np.asarray(resolution)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_npz = OUT_DIR / f"{clock}.npz"
    np.savez_compressed(
        out_npz,
        frequencies_khz=np.asarray(frequencies),
        wall_cm=np.asarray(depths),
        r_center_cm=np.asarray(metrics["r_center_cm"]),
        conductor=images["conductor"],
        insulator=images["insulator"],
        conductor_grid=grids["conductor"],
        insulator_grid=grids["insulator"],
        conductor_re_raw=metrics["conductor_re_raw"],
        insulator_re_raw=metrics["insulator_re_raw"],
        conductor_re_shape=metrics["conductor_re_shape"],
        insulator_re_shape=metrics["insulator_re_shape"],
        conductor_com_error_cm=metrics["conductor_com_error_cm"],
        insulator_com_error_cm=metrics["insulator_com_error_cm"],
        conductor_max_amplitude=metrics["conductor_max_amplitude"],
        insulator_max_amplitude=metrics["insulator_max_amplitude"],
        conductor_resolution_cm=metrics["conductor_resolution_cm"],
        insulator_resolution_cm=metrics["insulator_resolution_cm"],
        inclusion_radius_cm=INCLUSION_RADIUS_CM,
        gamma=GAMMA,
        omega2=OMEGA2,
    )
    print("saved", out_npz)
    save_figure(clock, frequencies, depths, grids, metrics)


def save_figure(clock, frequencies, depths, grids, metrics) -> None:
    # Show 50 kHz when present, otherwise the middle recorded frequency.
    if 50.0 in frequencies:
        freq_index = frequencies.index(50.0)
        freq_label = "50 kHz"
    else:
        freq_index = len(frequencies) // 2
        freq_label = f"{frequencies[freq_index]:.0f} kHz"
    r_center = np.asarray(metrics["r_center_cm"])

    fig = plt.figure(figsize=(3.0 * len(depths) + 4.5, 8.2))
    grid_spec = fig.add_gridspec(2, len(depths) + 1, width_ratios=[1] * len(depths) + [1.35])
    for row, kind in enumerate(("conductor", "insulator")):
        for col, depth in enumerate(depths):
            axis = fig.add_subplot(grid_spec[row, col])
            image = grids[kind][col, freq_index]
            peak = np.nanmax(np.abs(image)) or 1.0
            axis.imshow(image, origin="lower", cmap="Blues", vmin=0, vmax=peak)
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(f"wall {depth} cm\nr={TANK_RADIUS_CM - depth:.0f} cm", fontsize=8)
            if col == 0:
                axis.set_ylabel(kind, fontsize=10)
        axis = fig.add_subplot(grid_spec[row, -1])
        for freq_i, freq in enumerate(frequencies):
            axis.plot(r_center, metrics[f"{kind}_re_shape"][:, freq_i], color="0.75", linewidth=0.8)
        axis.plot(r_center, metrics[f"{kind}_re_shape"].mean(axis=1), "o-", color="C0", label="mean")
        axis.set_xlabel("r from center (cm)")
        axis.set_ylabel("shape RE")
        axis.set_title(f"{kind} RE")
        axis.grid(True, alpha=0.3)
        axis.invert_xaxis()
    fig.suptitle(
        f"{clock}  |  EM-GPRN  |  maps at {freq_label}  |  γ={GAMMA}, ω²={OMEGA2}",
        fontsize=12,
    )
    fig.tight_layout()
    out = OUT_DIR / f"{clock}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("saved", out)


def raise_center_weight(weights: np.ndarray, centroids: np.ndarray, target: float, radius_cm: float = 3.0) -> np.ndarray:
    """Scale the central weights so their median equals target. Outer weights stay put."""
    updated = np.array(weights, copy=True)
    center = np.linalg.norm(centroids, axis=1) <= radius_cm
    current = float(np.median(updated[center]))
    updated[center] *= target / current
    return updated


def conductor_center_weight_sweep(mesh, jacobian, centroids) -> None:
    """Conductor center files only. Central median weight 1.80, 2.5, and 3.0."""
    base = sensitivity_weights(jacobian, triangle_area(mesh["node"], mesh["element"]))
    radius = np.linalg.norm(centroids, axis=1)
    base_center = float(np.median(base[radius <= 3.0]))
    targets = (round(base_center, 2), 2.5, 3.0)
    weight_sets = [raise_center_weight(base, centroids, target) for target in targets]
    for target, weights in zip(targets, weight_sets):
        print(
            f"conductor center median W={np.median(weights[radius <= 3.0]):.2f} "
            f"outer median W={np.median(weights[radius >= 8.0]):.2f}"
        )

    gammas = (1.0, 2.0, 5.0)
    omegas = (0.05, 0.1)
    sweep_dir = OUT_DIR / "center_sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    grid_cache = {}
    stored = {"gammas": np.asarray(gammas), "omegas": np.asarray(omegas), "center_weight_targets": np.asarray(targets)}

    for omega2 in omegas:
        fig, axes = plt.subplots(len(CLOCKS), len(gammas) * len(targets), figsize=(16.5, 9.2))
        for clock_i, (clock, angle) in enumerate(CLOCKS.items()):
            folder = {
                "3oclock": DATA_ROOT / "3'o clock",
                "6oclock": DATA_ROOT / "6'o clock",
                "9oclock": DATA_ROOT / "9'o clock",
                "12oclock": DATA_ROOT / "12'o clock",
            }[clock]
            files = position_files(folder)
            loaded = load_by_frequency(files["conductor"][10])
            reference = load_by_frequency(files["conductor_ref"])
            frequencies = sorted(loaded)
            freq_index = frequencies.index(50.0) if 50.0 in frequencies else len(frequencies) // 2
            target_xy = truth_xy(angle, 0.0)
            for gamma_i, gamma in enumerate(gammas):
                for weight_i, (target, weights) in enumerate(zip(targets, weight_sets)):
                    sigmas = []
                    com_errors = []
                    resolutions = []
                    for freq in frequencies:
                        voltage = delta_u(loaded[freq], reference[freq], "conductor")
                        sigma = em_gprn(jacobian, voltage, gamma=gamma, omega2=omega2, weights=weights)
                        sigmas.append(sigma)
                        blob = blob_metrics(sigma, centroids, target_xy)
                        com_errors.append(blob["com_error_cm"])
                        resolutions.append(blob["resolution_cm"])
                    column = gamma_i * len(targets) + weight_i
                    image = to_grid(mesh, sigmas[freq_index], grid_cache)
                    axis = axes[clock_i, column]
                    peak = np.nanmax(image) or 1.0
                    axis.imshow(image, origin="lower", cmap="Blues", vmin=0, vmax=peak)
                    axis.set_xticks([])
                    axis.set_yticks([])
                    if clock_i == 0:
                        axis.set_title(f"γ={gamma:g}\nW={target:g}", fontsize=9)
                    if column == 0:
                        axis.set_ylabel(clock, fontsize=9)
                    print(
                        f"ω²={omega2} {clock} γ={gamma:g} W={target:g} "
                        f"COM={com_errors[freq_index]:.2f} cm res={resolutions[freq_index]:.2f} cm"
                    )
                    key = f"w{omega2:g}_{clock}_g{gamma:g}_W{target:g}"
                    stored[key + "_com"] = np.asarray(com_errors)
                    stored[key + "_resolution"] = np.asarray(resolutions)
                    stored[key + "_grid50"] = image
        fig.suptitle(
            f"Conductors only, r = 0  |  ω² = {omega2:g}  |  central weight raised, outer weight unchanged  |  50 kHz",
            fontsize=12,
        )
        fig.tight_layout()
        out = sweep_dir / f"conductor_weight_omega{omega2:g}.png"
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print("saved", out)

    np.savez_compressed(sweep_dir / "conductor_center_weight.npz", **stored)
    print("saved", sweep_dir / "conductor_center_weight.npz")


def center_parameter_sweep(mesh, jacobian, centroids) -> None:
    """Center inclusion only (file label 10 cm, r = 0). gamma in {1, 2, 5}, omega^2 in {0.05, 0.1}."""
    weights = sensitivity_weights(jacobian, triangle_area(mesh["node"], mesh["element"]))
    radius = np.linalg.norm(centroids, axis=1)
    center_weight = float(np.median(weights[radius <= 3.0]))
    edge_weight = float(np.median(weights[radius >= 8.0]))
    print(f"L = diag(W), median W center={center_weight:.2f} outer={edge_weight:.2f}")

    gammas = (1.0, 2.0, 5.0)
    omegas = (0.05, 0.1)
    sweep_dir = OUT_DIR / "center_sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    grid_cache = {}
    stored = {
        "gammas": np.asarray(gammas),
        "omegas": np.asarray(omegas),
        "weights": weights,
    }

    for omega2 in omegas:
        fig, axes = plt.subplots(len(CLOCKS) * 2, len(gammas), figsize=(8.4, 16.5))
        for clock_i, (clock, angle) in enumerate(CLOCKS.items()):
            folder = {
                "3oclock": DATA_ROOT / "3'o clock",
                "6oclock": DATA_ROOT / "6'o clock",
                "9oclock": DATA_ROOT / "9'o clock",
                "12oclock": DATA_ROOT / "12'o clock",
            }[clock]
            files = position_files(folder)
            depth = 10
            loaded = {
                "conductor": load_by_frequency(files["conductor"][depth]),
                "insulator": load_by_frequency(files["insulator"][depth]),
            }
            refs = {
                "conductor": load_by_frequency(files["conductor_ref"]),
                "insulator": load_by_frequency(files["insulator_ref"]),
            }
            frequencies = sorted(loaded["conductor"])
            freq_index = frequencies.index(50.0) if 50.0 in frequencies else len(frequencies) // 2
            truth = true_map(centroids, angle, 0.0)
            target = truth_xy(angle, 0.0)
            for kind_i, kind in enumerate(("conductor", "insulator")):
                row = clock_i * 2 + kind_i
                for gamma_i, gamma in enumerate(gammas):
                    sigmas = []
                    com_errors = []
                    resolutions = []
                    for freq in frequencies:
                        voltage = delta_u(loaded[kind][freq], refs[kind][freq], kind)
                        sigma = em_gprn(jacobian, voltage, gamma=gamma, omega2=omega2, weights=weights)
                        sigmas.append(sigma)
                        blob = blob_metrics(sigma, centroids, target)
                        com_errors.append(blob["com_error_cm"])
                        resolutions.append(blob["resolution_cm"])
                    image = to_grid(mesh, sigmas[freq_index], grid_cache)
                    axis = axes[row, gamma_i]
                    peak = np.nanmax(image) or 1.0
                    axis.imshow(image, origin="lower", cmap="Blues", vmin=0, vmax=peak)
                    axis.set_xticks([])
                    axis.set_yticks([])
                    if row == 0:
                        axis.set_title(f"γ={gamma:g}", fontsize=11)
                    if gamma_i == 0:
                        axis.set_ylabel(f"{clock}\n{kind}", fontsize=8)
                    print(
                        f"ω²={omega2} {clock} {kind} γ={gamma:g} "
                        f"COM={com_errors[freq_index]:.2f} cm res={resolutions[freq_index]:.2f} cm"
                    )
                    key = f"w{omega2:g}_{clock}_{kind}_g{gamma:g}"
                    stored[key + "_sigma"] = np.asarray(sigmas)
                    stored[key + "_com"] = np.asarray(com_errors)
                    stored[key + "_resolution"] = np.asarray(resolutions)
                    stored[key + "_grid50"] = image
            stored[f"{clock}_frequencies"] = np.asarray(frequencies)
        fig.suptitle(
            f"Center only (r = 0)  |  ω² = {omega2:g}  |  L = diag(1 / sensitivity)  |  50 kHz",
            fontsize=12,
        )
        fig.tight_layout()
        out = sweep_dir / f"center_omega{omega2:g}.png"
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print("saved", out)

    np.savez_compressed(sweep_dir / "center_sweep.npz", **stored)
    print("saved", sweep_dir / "center_sweep.npz")


def insulator_wall_override(mesh, jacobian, centroids) -> None:
    """3 o'clock insulator, 2 cm from the wall, with the outer layer heavily penalized."""
    base = sensitivity_weights(jacobian, triangle_area(mesh["node"], mesh["element"]))
    edged = penalize_outer_layer(base, centroids)
    radius = np.linalg.norm(centroids, axis=1)
    outer = radius >= 9.0
    print(
        f"base outer median W={np.median(base[outer]):.2f} "
        f"insulator-edge outer median W={np.median(edged[outer]):.2f} "
        f"center median W stays {np.median(base[radius <= 3.0]):.2f}"
    )
    folder = DATA_ROOT / "3'o clock"
    files = position_files(folder)
    gammas = (1.0, 2.0, 5.0)
    omegas = (0.05, 0.1)
    grid_cache = {}
    out_dir = OUT_DIR / "center_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    target = truth_xy(CLOCKS["3oclock"], TANK_RADIUS_CM - 2)
    loaded = load_by_frequency(files["insulator"][2])
    reference = load_by_frequency(files["insulator_ref"])
    frequencies = sorted(loaded)
    freq_index = frequencies.index(50.0) if 50.0 in frequencies else len(frequencies) // 2
    weight_sets = (("standard", base), ("outer penalized", edged))
    fig, axes = plt.subplots(2, len(omegas) * len(gammas), figsize=(14.5, 5.4))
    for row, (label, weights) in enumerate(weight_sets):
        for omega_i, omega2 in enumerate(omegas):
            for gamma_i, gamma in enumerate(gammas):
                com_at_50 = np.nan
                res_at_50 = np.nan
                image = None
                for freq_i, freq in enumerate(frequencies):
                    voltage = delta_u(loaded[freq], reference[freq], "insulator")
                    sigma = em_gprn(jacobian, voltage, gamma=gamma, omega2=omega2, weights=weights)
                    if freq_i == freq_index:
                        blob = blob_metrics(sigma, centroids, target)
                        com_at_50 = blob["com_error_cm"]
                        res_at_50 = blob["resolution_cm"]
                        image = to_grid(mesh, sigma, grid_cache)
                column = omega_i * len(gammas) + gamma_i
                axis = axes[row, column]
                peak = np.nanmax(image) or 1.0
                axis.imshow(image, origin="lower", cmap="Blues", vmin=0, vmax=peak)
                axis.set_xticks([])
                axis.set_yticks([])
                if row == 0:
                    axis.set_title(f"ω²={omega2:g}\nγ={gamma:g}", fontsize=9)
                if column == 0:
                    axis.set_ylabel(label, fontsize=8)
                print(f"{label} ω²={omega2:g} γ={gamma:g} COM={com_at_50:.2f} cm res={res_at_50:.2f} cm")
    fig.suptitle("3 o'clock insulator, 2 cm from the wall  |  50 kHz  |  outer layer penalized on the bottom row", fontsize=11)
    fig.tight_layout()
    out = out_dir / "3oclock_wall2cm_insulator_edge.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("saved", out)

    # Same mesh and the original weights, so the center penalty is not the wall override.
    center_fig, center_axes = plt.subplots(1, 2, figsize=(6.2, 3.4))
    for axis, kind in zip(center_axes, ("conductor", "insulator")):
        measured = load_by_frequency(files[kind][10])
        ref = load_by_frequency(files[f"{kind}_ref"])
        frequencies = sorted(measured)
        freq_index = frequencies.index(50.0) if 50.0 in frequencies else len(frequencies) // 2
        voltage = delta_u(measured[frequencies[freq_index]], ref[frequencies[freq_index]], kind)
        sigma = em_gprn(jacobian, voltage, gamma=5.0, omega2=0.1, weights=base)
        image = to_grid(mesh, sigma, grid_cache)
        peak = np.nanmax(image) or 1.0
        axis.imshow(image, origin="lower", cmap="Blues", vmin=0, vmax=peak)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_title(kind)
        blob = blob_metrics(sigma, centroids, truth_xy(CLOCKS["3oclock"], 0.0))
        print(f"center {kind} ω²=0.1 γ=5 COM={blob['com_error_cm']:.2f} cm res={blob['resolution_cm']:.2f} cm")
    center_fig.suptitle("3 o'clock center, original weights, new wall mesh  |  ω²=0.1, γ=5", fontsize=10)
    center_fig.tight_layout()
    center_out = out_dir / "3oclock_center_after_wall_mesh.png"
    center_fig.savefig(center_out, dpi=140)
    plt.close(center_fig)
    print("saved", center_out)


def wall2_3oclock(mesh, jacobian, centroids) -> None:
    """3 o'clock, 2 cm from the wall, with the center-sweep penalty and gamma/omega values."""
    weights = sensitivity_weights(jacobian, triangle_area(mesh["node"], mesh["element"]))
    folder = DATA_ROOT / "3'o clock"
    files = position_files(folder)
    depth = 2
    r_center = TANK_RADIUS_CM - depth
    angle = CLOCKS["3oclock"]
    target = truth_xy(angle, r_center)
    gammas = (1.0, 2.0, 5.0)
    omegas = (0.05, 0.1)
    grid_cache = {}
    out_dir = OUT_DIR / "center_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, len(omegas) * len(gammas), figsize=(14.5, 5.2))
    stored = {"gammas": np.asarray(gammas), "omegas": np.asarray(omegas), "r_center_cm": r_center}

    for kind_i, kind in enumerate(("conductor", "insulator")):
        loaded = load_by_frequency(files[kind][depth])
        reference = load_by_frequency(files[f"{kind}_ref"])
        frequencies = sorted(loaded)
        freq_index = frequencies.index(50.0) if 50.0 in frequencies else len(frequencies) // 2
        for omega_i, omega2 in enumerate(omegas):
            for gamma_i, gamma in enumerate(gammas):
                sigmas = []
                com_errors = []
                resolutions = []
                for freq in frequencies:
                    voltage = delta_u(loaded[freq], reference[freq], kind)
                    sigma = em_gprn(jacobian, voltage, gamma=gamma, omega2=omega2, weights=weights)
                    sigmas.append(sigma)
                    blob = blob_metrics(sigma, centroids, target)
                    com_errors.append(blob["com_error_cm"])
                    resolutions.append(blob["resolution_cm"])
                column = omega_i * len(gammas) + gamma_i
                image = to_grid(mesh, sigmas[freq_index], grid_cache)
                axis = axes[kind_i, column]
                peak = np.nanmax(image) or 1.0
                axis.imshow(image, origin="lower", cmap="Blues", vmin=0, vmax=peak)
                axis.set_xticks([])
                axis.set_yticks([])
                if kind_i == 0:
                    axis.set_title(f"ω²={omega2:g}\nγ={gamma:g}", fontsize=9)
                if column == 0:
                    axis.set_ylabel(kind, fontsize=10)
                print(
                    f"{kind} ω²={omega2:g} γ={gamma:g} "
                    f"COM={com_errors[freq_index]:.2f} cm res={resolutions[freq_index]:.2f} cm"
                )
                key = f"{kind}_w{omega2:g}_g{gamma:g}"
                stored[key + "_grid50"] = image
                stored[key + "_com"] = np.asarray(com_errors)
                stored[key + "_resolution"] = np.asarray(resolutions)
                stored[key + "_sigma"] = np.asarray(sigmas)
        stored[f"{kind}_frequencies"] = np.asarray(frequencies)

    fig.suptitle("3 o'clock, 2 cm from the wall (r = 8 cm)  |  50 kHz  |  same L, γ, and ω² as the center sweep", fontsize=11)
    fig.tight_layout()
    out = out_dir / "3oclock_wall2cm.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    np.savez_compressed(out_dir / "3oclock_wall2cm.npz", **stored)
    print("saved", out)


FINAL_DIR = ROOT / "results" / "expectation maximum"
INSULATOR_GAMMA = 5.0
INSULATOR_OMEGA2 = 0.1


def save_final_results(uniform) -> None:
    """Both materials on the first EM-GPRN solver: uniform mesh, gamma=0.35, omega^2=0.01, L=I."""
    uniform_mesh, uniform_j, uniform_centroids = uniform
    uniform_cache = {}
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
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
            sigma_rows = []
            grid_rows = []
            com_rows = []
            res_rows = []
            re_raw_rows = []
            re_shape_rows = []
            amp_rows = []
            for depth in depths:
                r_center = TANK_RADIUS_CM - depth
                target = truth_xy(angle, r_center)
                truth = true_map(uniform_centroids, angle, r_center)
                sigma_row = []
                grid_row = []
                com_row = []
                res_row = []
                re_raw_row = []
                re_shape_row = []
                amp_row = []
                for freq in frequencies:
                    voltage = delta_u(loaded[kind][depth][freq], refs[kind][freq], kind)
                    sigma = em_gprn(uniform_j, voltage, gamma=GAMMA, omega2=OMEGA2)
                    image = to_grid(uniform_mesh, sigma, uniform_cache)
                    blob = blob_metrics(sigma, uniform_centroids, target)
                    raw_re, shape_re = relative_error(sigma, truth)
                    sigma_row.append(sigma)
                    grid_row.append(image)
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
                peak = np.nanmax(image) or 1.0
                axis.imshow(image, origin="lower", cmap="Blues", vmin=0, vmax=peak)
                axis.set_xticks([])
                axis.set_yticks([])
                if row == 0:
                    axis.set_title(f"wall {depth} cm\nr={TANK_RADIUS_CM - depth:.0f} cm", fontsize=8)
                if col == 0:
                    axis.set_ylabel(kind, fontsize=9)
            axis = fig.add_subplot(grid_spec[row, -1])
            for freq_i in range(len(frequencies)):
                axis.plot(r_center, metrics[kind]["re_shape"][:, freq_i], color="0.75", linewidth=0.8)
            axis.plot(r_center, metrics[kind]["re_shape"].mean(axis=1), "o-", color="C0", label="mean")
            axis.set_xlabel("r from center (cm)")
            axis.set_ylabel("shape RE")
            axis.set_title(f"{kind} RE")
            axis.grid(True, alpha=0.3)
            axis.invert_xaxis()
        fig.suptitle(f"{clock}  |  first EM-GPRN  |  γ=0.35, ω²=0.01, L=I  |  50 kHz", fontsize=11)
        fig.tight_layout()
        figure_path = FINAL_DIR / f"{clock}.png"
        try:
            fig.savefig(figure_path, dpi=140)
        except OSError:
            figure_path = FINAL_DIR / f"{clock}_new.png"
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
        analysis.suptitle(f"{clock}  |  depth analysis  |  faint lines are individual frequencies", fontsize=11)
        analysis.tight_layout()
        analysis_path = FINAL_DIR / f"{clock}_analysis.png"
        try:
            analysis.savefig(analysis_path, dpi=140)
        except OSError:
            analysis_path = FINAL_DIR / f"{clock}_analysis_new.png"
            analysis.savefig(analysis_path, dpi=140)
        plt.close(analysis)
        np.savez_compressed(FINAL_DIR / f"{clock}.npz", **stored)
        print("saved", figure_path)
        print("saved", analysis_path)


def main() -> None:
    print("Building 20 cm mesh and normalized Jacobian")
    mesh, jacobian, centroids = build_workspace()
    print(f"measurements={jacobian.shape[0]}")
    grid_cache = {}
    for clock, angle in CLOCKS.items():
        reconstruct_position(clock, angle, mesh, jacobian, centroids, grid_cache)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "center-sweep":
        print("Building 20 cm mesh and normalized Jacobian")
        mesh, jacobian, centroids = build_workspace()
        center_parameter_sweep(mesh, jacobian, centroids)
    elif len(sys.argv) > 1 and sys.argv[1] == "insulator-wall":
        print("Building 20 cm mesh and normalized Jacobian")
        mesh, jacobian, centroids = build_workspace()
        insulator_wall_override(mesh, jacobian, centroids)
    elif len(sys.argv) > 1 and sys.argv[1] == "wall2-3oclock":
        print("Building 20 cm mesh and normalized Jacobian")
        mesh, jacobian, centroids = build_workspace()
        wall2_3oclock(mesh, jacobian, centroids)
    elif len(sys.argv) > 1 and sys.argv[1] == "final":
        print("Building the first uniform mesh")
        uniform = build_uniform_workspace()
        save_final_results(uniform)
    else:
        main()
