"""Complete-electrode EM reconstructions for inclusions 2 cm from the wall.

The saved point-electrode images linearize once on a homogeneous tank and
inject current at a single node. An inclusion whose center is at r = 8 cm
sits in the electrode near-field, so that Jacobian splits insulators and
pulls conductors onto the boundary.

This script builds a graded mesh (about 2 mm on the wall), places each
electrode on a 1 cm arc with a contact impedance, and repeats the EM solve
three times. After each solve the conductivity is updated and the Jacobian
is rebuilt at that conductivity.
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_matrix
from scipy.spatial import Delaunay

from em_gprn_wang2012 import (
    CLOCKS,
    DATA_ROOT,
    GAMMA,
    GRID_N,
    INCLUSION_RADIUS_CM,
    N_ELECTRODES,
    N_NEWTON,
    OMEGA2,
    REVERSE_ELECTRODES,
    ROTATE_CLOCKWISE,
    TANK_RADIUS_CM,
    blob_metrics,
    delta_u,
    em_gprn,
    load_by_frequency,
    position_files,
    to_grid,
    true_map,
    truth_xy,
)
from general_utils.recon_utils.pyeit.eit.fem import calculate_ke, voltage_meter
from general_utils.recon_utils.pyeit.eit.utils import eit_scan_lines
from general_utils.recon_utils.pyeit.mesh.shape import circle, fix_points_fd
from general_utils.recon_utils.pyeit.mesh.utils import check_order

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "workspace" / "cem_wall"
ELECTRODE_WIDTH_CM = 1.0
CONTACT_Z = 1.0
WALL_CM = 2.0
FREQUENCY_KHZ = 50.0
N_OUTER = 3
CONTRASTS = (2.0, 6.0, 15.0, 30.0)
EX_MAT = eit_scan_lines(N_ELECTRODES, 1)


def electrode_centers() -> np.ndarray:
    """Hardware electrode k sits at the same point the point-electrode model uses."""
    points = fix_points_fd(circle, n_el=N_ELECTRODES) * TANK_RADIUS_CM
    order = np.roll(np.arange(N_ELECTRODES), -ROTATE_CLOCKWISE)
    if REVERSE_ELECTRODES:
        order = order[::-1]
    return points[order]


def build_mesh(centers: np.ndarray):
    """Wall nodes every 2 mm, coarser triangles toward the center."""
    spacing = 0.20
    theta = np.linspace(0.0, 2.0 * np.pi, int(np.ceil(2.0 * np.pi * TANK_RADIUS_CM / spacing)), endpoint=False)
    boundary = np.column_stack((TANK_RADIUS_CM * np.cos(theta), TANK_RADIUS_CM * np.sin(theta)))

    ys = np.arange(-TANK_RADIUS_CM, TANK_RADIUS_CM + spacing, spacing * np.sqrt(3.0) / 2.0)
    xs = np.arange(-TANK_RADIUS_CM, TANK_RADIUS_CM + spacing, spacing)
    xx, yy = np.meshgrid(xs, ys)
    xx = xx.copy()
    xx[1::2, :] += spacing / 2.0
    lattice = np.column_stack((xx.ravel(), yy.ravel()))
    lattice = lattice[np.linalg.norm(lattice, axis=1) <= TANK_RADIUS_CM - 0.35]

    candidates = lattice
    radius = np.linalg.norm(candidates, axis=1)
    inward = np.clip((8.0 - radius) / 8.0, 0.0, 1.0)
    target = 0.22 + (1.05 - 0.22) * np.where(radius >= 8.0, 0.0, inward ** 1.3)
    order = np.argsort(target, kind="mergesort")
    candidates = candidates[order]
    allowance = 0.78 * target[order]

    kept = np.empty((boundary.shape[0] + candidates.shape[0], 2), dtype=float)
    kept[: boundary.shape[0]] = boundary
    n_kept = boundary.shape[0]
    for point, limit in zip(candidates, allowance):
        if np.linalg.norm(kept[:n_kept] - point, axis=1).min() >= limit:
            kept[n_kept] = point
            n_kept += 1
    nodes = kept[:n_kept]
    triangles = check_order(nodes, Delaunay(nodes).simplices.copy())
    centroids = nodes[triangles].mean(axis=1)
    inside = np.linalg.norm(centroids, axis=1) <= TANK_RADIUS_CM + 1e-6
    triangles = triangles[inside]
    centroids = centroids[inside]

    angles = np.arctan2(boundary[:, 1], boundary[:, 0])
    center_angles = np.arctan2(centers[:, 1], centers[:, 0])
    half = (ELECTRODE_WIDTH_CM / 2.0) / TANK_RADIUS_CM
    members = []
    for center_angle in center_angles:
        delta = (angles - center_angle + np.pi) % (2.0 * np.pi) - np.pi
        chosen = np.flatnonzero(np.abs(delta) <= half)
        members.append(chosen[np.argsort(delta[chosen])])

    lengths = []
    for offset in ((0, 1), (1, 2), (2, 0)):
        lengths.append(np.linalg.norm(nodes[triangles[:, offset[0]]] - nodes[triangles[:, offset[1]]], axis=1))
    edge = np.median(np.column_stack(lengths), axis=1)
    wall = np.linalg.norm(centroids, axis=1) >= 9.0
    print(
        f"mesh nodes={nodes.shape[0]} elements={triangles.shape[0]} "
        f"wall_edge_mm={np.median(edge[wall]) * 10:.2f} "
        f"electrode_nodes={[len(item) for item in members]}"
    )
    mesh = {"node": nodes, "element": triangles, "perm": np.ones(triangles.shape[0])}
    geometry = {
        "boundary_index": np.arange(boundary.shape[0]),
        "members": members,
        "ke": calculate_ke(nodes, triangles),
    }
    return mesh, centroids, geometry


class CompleteElectrode:
    def __init__(self, mesh: dict, geometry: dict, contact_z: float = CONTACT_Z, rotate: int = 0):
        self.nodes = mesh["node"]
        self.triangles = mesh["element"]
        self.ke = geometry["ke"]
        self.contact_z = contact_z
        members = list(geometry["members"])
        if rotate:
            members = members[rotate:] + members[:rotate]
        self.members = members
        self.boundary_index = geometry["boundary_index"]
        self.n_nodes = self.nodes.shape[0]
        self.n_elem = self.triangles.shape[0]
        self.pairs = [
            voltage_meter(line, n_el=N_ELECTRODES, step=1, parser="fmmu") for line in EX_MAT
        ]
        self._coupling = self._boundary_coupling()

    def _boundary_coupling(self):
        """∫ φ_i φ_j, ∫ φ_i and edge length along each electrode arc."""
        blocks = []
        boundary = self.nodes[self.boundary_index]
        for member in self.members:
            rows, cols, mass, load, length = [], [], [], [], 0.0
            for left, right in zip(member[:-1], member[1:]):
                span = float(np.linalg.norm(boundary[right] - boundary[left]))
                a = int(self.boundary_index[left])
                b = int(self.boundary_index[right])
                length += span
                for i, j, weight in ((a, a, span / 3.0), (b, b, span / 3.0), (a, b, span / 6.0), (b, a, span / 6.0)):
                    rows.append(i)
                    cols.append(j)
                    mass.append(weight)
                load.append((a, span / 2.0))
                load.append((b, span / 2.0))
            blocks.append((rows, cols, mass, load, length))
        return blocks

    def solve(self, perm: np.ndarray):
        n = self.n_nodes
        size = n + N_ELECTRODES
        row = np.repeat(self.triangles, 3).ravel()
        col = np.repeat(self.triangles, 3, axis=0).ravel()
        data = (self.ke * perm[:, None, None]).ravel()
        matrix = coo_matrix((data, (row, col)), shape=(size, size)).tolil()
        scale = 1.0 / self.contact_z
        for electrode, (rows, cols, mass, load, length) in enumerate(self._coupling):
            for i, j, weight in zip(rows, cols, mass):
                matrix[i, j] += scale * weight
            for node, weight in load:
                matrix[node, n + electrode] -= scale * weight
                matrix[n + electrode, node] -= scale * weight
            matrix[n + electrode, n + electrode] += scale * length
        matrix[size - 1, :] = 0.0
        matrix[size - 1, n:] = 1.0
        dense = matrix.toarray()
        rhs_block = np.zeros((size, N_ELECTRODES))
        for drive, line in enumerate(EX_MAT):
            rhs_block[n + int(line[0]), drive] = 1.0
            rhs_block[n + int(line[1]), drive] = -1.0
        solution = np.linalg.solve(dense, rhs_block)
        potentials = solution[:n].T
        voltages = solution[n:].T
        measurements = self.measurements(voltages)
        jacobian = self.jacobian(potentials)
        return measurements, jacobian

    def measurements(self, voltages: np.ndarray) -> np.ndarray:
        blocks = []
        for drive, pairs in enumerate(self.pairs):
            blocks.append(voltages[drive, pairs[:, 0]] - voltages[drive, pairs[:, 1]])
        return np.concatenate(blocks)

    def jacobian(self, potentials: np.ndarray) -> np.ndarray:
        """∂(U_n - U_m) / ∂σ on each triangle. Rows follow the fmmu adjacent order."""
        grads = np.empty((N_ELECTRODES, self.n_elem, 2))
        area = None
        xy = self.nodes[self.triangles]
        x = xy[:, :, 0]
        y = xy[:, :, 1]
        twice = (x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0]) - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0])
        basis = np.empty((self.n_elem, 3, 2))
        basis[:, 0, 0] = y[:, 1] - y[:, 2]
        basis[:, 0, 1] = x[:, 2] - x[:, 1]
        basis[:, 1, 0] = y[:, 2] - y[:, 0]
        basis[:, 1, 1] = x[:, 0] - x[:, 2]
        basis[:, 2, 0] = y[:, 0] - y[:, 1]
        basis[:, 2, 1] = x[:, 1] - x[:, 0]
        basis /= twice[:, None, None]
        area = 0.5 * np.abs(twice)
        for drive in range(N_ELECTRODES):
            nodal = potentials[drive, self.triangles]
            grads[drive] = np.einsum("ei,eij->ej", nodal, basis)
        rows = []
        for drive, pairs in enumerate(self.pairs):
            adjoint = pairs[:, 1]
            dots = np.einsum("ej,mej->me", grads[drive], grads[adjoint])
            rows.append(dots * area)
        return np.vstack(rows)


def count_blobs(values: np.ndarray, triangles: np.ndarray) -> int:
    peak = float(np.max(values))
    if peak <= 0:
        return 0
    active = np.flatnonzero(values >= 0.5 * peak)
    if active.size == 0:
        return 0
    parent = {int(item): int(item) for item in active}

    def find(item):
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    node_owner = {}
    for element in active:
        for node in triangles[element]:
            node = int(node)
            if node in node_owner:
                left, right = find(int(element)), find(node_owner[node])
                parent[right] = left
            else:
                node_owner[node] = int(element)
    return len({find(int(item)) for item in active})


def conductivity(kind: str, shape: np.ndarray, contrast: float) -> np.ndarray:
    amount = contrast * np.clip(shape, 0.0, None)
    if kind == "conductor":
        return 1.0 + amount
    return 1.0 / (1.0 + amount)


def prediction(kind: str, measured: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if kind == "conductor":
        return -(measured - reference)
    return measured - reference


def reconstruct(model: CompleteElectrode, data: np.ndarray, kind: str):
    """Three EM solves. The contrast is chosen so the CEM voltages track the data."""
    reference, jacobian = model.solve(np.ones(model.n_elem))
    image = np.zeros(model.n_elem)
    chosen = 0.0
    for outer in range(N_OUTER):
        if outer == 0:
            step = -jacobian
        else:
            sigma = conductivity(kind, image, chosen)
            simulated, jacobian = model.solve(sigma)
            slope = np.square(sigma) if kind == "insulator" else np.ones_like(sigma)
            step = -jacobian * slope
        scale = float(np.max(np.abs(step)))
        if scale == 0:
            break
        image = em_gprn(step / scale, data / scale, gamma=GAMMA, omega2=OMEGA2)
        peak = float(image.max())
        if peak <= 0:
            break
        shape = image / peak
        best_corr = -np.inf
        chosen = CONTRASTS[0]
        for contrast in CONTRASTS:
            simulated, _ = model.solve(conductivity(kind, shape, contrast))
            predicted = prediction(kind, simulated, reference)
            corr = float(np.corrcoef(predicted, data)[0, 1])
            if corr > best_corr:
                best_corr = corr
                chosen = contrast
        print(f"  {kind} outer={outer} contrast={chosen:.0f} corr={best_corr:.3f}")
    return image, chosen


def finite_difference_check(model: CompleteElectrode) -> float:
    perm = np.ones(model.n_elem)
    base, jacobian = model.solve(perm)
    probes = [
        int(np.argmin(np.linalg.norm(model.nodes[model.triangles].mean(axis=1), axis=1))),
        int(np.argmax(np.linalg.norm(model.nodes[model.triangles].mean(axis=1), axis=1))),
    ]
    errors = []
    eps = 1e-4
    for element in probes:
        perm[element] += eps
        moved, _ = model.solve(perm)
        perm[element] -= eps
        column = (moved - base) / eps
        denom = np.linalg.norm(column)
        errors.append(float(np.linalg.norm(column - jacobian[:, element]) / denom))
    score = float(np.max(errors))
    print(f"jacobian finite-difference relative error={score:.3e}")
    return score


def show_images(path: Path, titles: list[str], images: list[np.ndarray], markers: list[np.ndarray]) -> None:
    fig, axes = plt.subplots(1, len(images), figsize=(3.1 * len(images), 3.4))
    if len(images) == 1:
        axes = [axes]
    for axis, title, image, marker in zip(axes, titles, images, markers):
        peak = np.nanmax(image) or 1.0
        axis.imshow(image, origin="lower", cmap="Blues", vmin=0, vmax=peak, extent=(-10, 10, -10, 10))
        axis.plot(marker[0], marker[1], "r+", markersize=8, markeredgewidth=1.2)
        axis.set_title(title, fontsize=8)
        axis.set_xticks([])
        axis.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def synthetic(model: CompleteElectrode, mesh, centroids) -> None:
    angle = CLOCKS["3oclock"]
    target = truth_xy(angle, TANK_RADIUS_CM - WALL_CM)
    disk = true_map(centroids, angle, TANK_RADIUS_CM - WALL_CM)
    cache = {}
    titles, images, markers = [], [], []
    for kind, perm in (("conductor", 1.0 + 8.0 * disk), ("insulator", 1.0 / (1.0 + 8.0 * disk))):
        measured, _ = model.solve(perm)
        reference, _ = model.solve(np.ones(model.n_elem))
        data = prediction(kind, measured, reference)
        image, contrast = reconstruct(model, data, kind)
        grid = to_grid(mesh, image, cache)
        blob = blob_metrics(image, centroids, target)
        blobs = count_blobs(image, mesh["element"])
        print(
            f"synthetic {kind} contrast={contrast:.0f} blobs={blobs} "
            f"COM={blob['com_error_cm']:.2f} cm res={blob['resolution_cm']:.2f} cm"
        )
        titles.append(f"synthetic {kind}\n{blobs} blobs, COM {blob['com_error_cm']:.2f} cm")
        images.append(grid)
        markers.append(target)
    show_images(OUT_DIR / "synthetic_wall2.png", titles, images, markers)


def real_wall(model: CompleteElectrode, mesh, centroids) -> None:
    cache = {}
    folders = {
        "3oclock": DATA_ROOT / "3'o clock",
        "6oclock": DATA_ROOT / "6'o clock",
        "9oclock": DATA_ROOT / "9'o clock",
        "12oclock": DATA_ROOT / "12'o clock",
    }
    fig, axes = plt.subplots(4, 4, figsize=(12.5, 12.5))
    for row, (clock, angle) in enumerate(CLOCKS.items()):
        files = position_files(folders[clock])
        target = truth_xy(angle, TANK_RADIUS_CM - WALL_CM)
        old = np.load(ROOT / "results" / "expectation maximum" / f"{clock}.npz", allow_pickle=True)
        wall_index = int(np.where(np.asarray(old["wall_cm"]) == WALL_CM)[0][0])
        freq_index = int(np.where(np.asarray(old["frequencies_khz"]) == FREQUENCY_KHZ)[0][0])
        for column, kind in enumerate(("conductor", "insulator")):
            previous = np.asarray(old[f"{kind}_grid"][wall_index, freq_index], dtype=float)
            measured = load_by_frequency(files[kind][int(WALL_CM)])[FREQUENCY_KHZ]
            reference = load_by_frequency(files[f"{kind}_ref"])[FREQUENCY_KHZ]
            data = delta_u(measured, reference, kind)
            started = time.perf_counter()
            image, contrast = reconstruct(model, data, kind)
            blob = blob_metrics(image, centroids, target)
            blobs = count_blobs(image, mesh["element"])
            print(
                f"{clock} {kind} contrast={contrast:.0f} blobs={blobs} "
                f"COM={blob['com_error_cm']:.2f} cm res={blob['resolution_cm']:.2f} cm "
                f"({time.perf_counter() - started:.0f}s)"
            )
            grid = to_grid(mesh, image, cache)
            panels = (
                (previous, f"{clock} {kind}\ncurrent linear"),
                (grid, f"CEM ×{N_OUTER}, contrast {contrast:.0f}\n{blobs} blobs, COM {blob['com_error_cm']:.2f} cm"),
            )
            for offset, (panel, title) in enumerate(panels):
                axis = axes[row, column * 2 + offset]
                peak = np.nanmax(panel) or 1.0
                axis.imshow(panel, origin="lower", cmap="Blues", vmin=0, vmax=peak, extent=(-10, 10, -10, 10))
                axis.plot(target[0], target[1], "r+", markersize=7, markeredgewidth=1.1)
                axis.set_xticks([])
                axis.set_yticks([])
                axis.set_title(title, fontsize=8)
    fig.suptitle(
        f"Wall {WALL_CM:.0f} cm at {FREQUENCY_KHZ:.0f} kHz. "
        f"Electrode width {ELECTRODE_WIDTH_CM:.0f} cm, contact z={CONTACT_Z:g}.",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "wall2_cem.png", dpi=140)
    plt.close(fig)
    print("saved", OUT_DIR / "wall2_cem.png")


def main() -> None:
    """Wall-2 cm images: CEM, electrode index advanced one position, NOSER penalty."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mesh, centroids, geometry = build_mesh(electrode_centers())
    model = CompleteElectrode(mesh, geometry, contact_z=CONTACT_Z, rotate=1)
    error = finite_difference_check(model)
    if error > 1e-2:
        raise RuntimeError("CEM Jacobian does not match a finite difference")
    _, jacobian = model.solve(np.ones(model.n_elem))
    jacobian = -jacobian
    scale = float(np.max(np.abs(jacobian)))
    sensitivity = np.linalg.norm(jacobian, axis=0)
    weights = sensitivity / np.median(sensitivity)
    cache = {}
    folders = {
        "3oclock": DATA_ROOT / "3'o clock",
        "6oclock": DATA_ROOT / "6'o clock",
        "9oclock": DATA_ROOT / "9'o clock",
        "12oclock": DATA_ROOT / "12'o clock",
    }
    fig, axes = plt.subplots(4, 4, figsize=(12.2, 12.2))
    for row, (clock, angle) in enumerate(CLOCKS.items()):
        files = position_files(folders[clock])
        target = truth_xy(angle, TANK_RADIUS_CM - WALL_CM)
        old = np.load(ROOT / "results" / "expectation maximum" / f"{clock}.npz", allow_pickle=True)
        wall_index = int(np.where(np.asarray(old["wall_cm"]) == WALL_CM)[0][0])
        freq_index = int(np.where(np.asarray(old["frequencies_khz"]) == FREQUENCY_KHZ)[0][0])
        for column, kind in enumerate(("conductor", "insulator")):
            measured = load_by_frequency(files[kind][int(WALL_CM)])[FREQUENCY_KHZ]
            reference = load_by_frequency(files[f"{kind}_ref"])[FREQUENCY_KHZ]
            data = delta_u(measured, reference, kind)
            image = em_gprn(jacobian / scale, data / scale, gamma=5.0, omega2=OMEGA2, weights=weights)
            blob = blob_metrics(image, centroids, target)
            print(
                f"{clock} {kind} blobs={count_blobs(image, mesh['element'])} "
                f"COM={blob['com_error_cm']:.2f} cm res={blob['resolution_cm']:.2f} cm"
            )
            grid = to_grid(mesh, image, cache)
            previous = np.asarray(old[f"{kind}_grid"][wall_index, freq_index], dtype=float)
            old_com = float(old[f"{kind}_com_error_cm"][wall_index, freq_index])
            panels = (
                (previous, f"{clock} {kind}\ncurrent, COM {old_com:.2f} cm"),
                (grid, f"CEM + boundary penalty\nCOM {blob['com_error_cm']:.2f} cm"),
            )
            for offset, (panel, title) in enumerate(panels):
                axis = axes[row, column * 2 + offset]
                peak = np.nanmax(panel) or 1.0
                axis.imshow(
                    panel, origin="lower", cmap="Blues", vmin=0, vmax=peak, extent=(-10, 10, -10, 10)
                )
                axis.plot(target[0], target[1], "r+", markersize=7, markeredgewidth=1.1)
                axis.set_xticks([])
                axis.set_yticks([])
                axis.set_title(title, fontsize=8)
    fig.suptitle(
        f"Wall {WALL_CM:.0f} cm at {FREQUENCY_KHZ:.0f} kHz. "
        "CEM, electrode index +1, penalty proportional to sensitivity.",
        fontsize=11,
    )
    fig.tight_layout()
    path = OUT_DIR / "wall2_cem_shift_noser.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print("saved", path)


if __name__ == "__main__":
    main()
