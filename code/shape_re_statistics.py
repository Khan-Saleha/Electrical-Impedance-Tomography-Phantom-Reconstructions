"""Three-way repeated-measures ANOVA and depth regressions for shape RE."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from statsmodels.stats.anova import AnovaRM

ROOT = Path(__file__).resolve().parents[1]
EM_DIR = ROOT / "results" / "expectation maximum"
RESULT_DIR = ROOT / "results" / "ANOVA statistics_EM"
CLOCKS = ("3oclock", "6oclock", "9oclock", "12oclock")


def load_long() -> pd.DataFrame:
    rows = []
    for clock in CLOCKS:
        data = np.load(EM_DIR / f"{clock}.npz", allow_pickle=True)
        wall = np.asarray(data["wall_cm"], dtype=float)
        freq = np.asarray(data["frequencies_khz"], dtype=float)
        for material in ("conductor", "insulator"):
            shape = np.asarray(data[f"{material}_re_shape"], dtype=float)
            for depth_i, wall_cm in enumerate(wall):
                for freq_i, freq_khz in enumerate(freq):
                    rows.append(
                        {
                            "clock": clock,
                            "material": material,
                            "wall_cm": wall_cm,
                            "freq_khz": freq_khz,
                            "shape_re": shape[depth_i, freq_i],
                        }
                    )
    return pd.DataFrame(rows)


def repeated_measures(frame: pd.DataFrame) -> pd.DataFrame:
    model = AnovaRM(
        frame,
        depvar="shape_re",
        subject="clock",
        within=["material", "wall_cm", "freq_khz"],
    ).fit()
    table = model.anova_table.reset_index().rename(columns={"index": "effect"})
    table["partial_eta_sq"] = (table["F Value"] * table["Num DF"]) / (
        table["F Value"] * table["Num DF"] + table["Den DF"]
    )
    return table


def fit_family(x: np.ndarray, y: np.ndarray) -> list[dict]:
    def exponential(distance, amplitude, rate, offset):
        return amplitude * np.exp(rate * distance) + offset

    fits = []
    linear = np.polyfit(x, y, 1)
    linear_hat = np.polyval(linear, x)
    fits.append(("linear", linear_hat, 2, linear))
    quadratic = np.polyfit(x, y, 2)
    quadratic_hat = np.polyval(quadratic, x)
    fits.append(("quadratic", quadratic_hat, 3, quadratic))
    try:
        start = (float(y.max() - y.min()), -0.2, float(y.min()))
        params, _ = curve_fit(exponential, x, y, p0=start, maxfev=20000)
        fits.append(("exponential", exponential(x, *params), 3, params))
    except RuntimeError:
        pass

    rows = []
    n = len(y)
    for name, hat, k, params in fits:
        residual = y - hat
        rss = float(np.sum(residual**2))
        tss = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - rss / tss if tss > 0 else np.nan
        aic = n * np.log(rss / n) + 2 * k
        rows.append(
            {
                "model": name,
                "r_squared": r2,
                "aic": aic,
                "rss": rss,
                "parameters": np.array2string(np.asarray(params), precision=4),
            }
        )
    return rows


def regression(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    collapsed = (
        frame.groupby(["material", "clock", "wall_cm"], as_index=False)["shape_re"].mean()
    )
    records = []
    curves = {}
    grid = np.linspace(2, 10, 80)
    for material, group in collapsed.groupby("material"):
        x = group["wall_cm"].to_numpy()
        y = group["shape_re"].to_numpy()
        fits = fit_family(x, y)
        best = min(fits, key=lambda row: row["aic"])
        for row in fits:
            row = dict(row)
            row["material"] = material
            row["n"] = len(y)
            row["selected"] = row["model"] == best["model"]
            records.append(row)
        curves[material] = (x, y, best["model"], grid)
    return pd.DataFrame(records), curves


def draw(frame: pd.DataFrame, curves: dict, selected: pd.DataFrame) -> None:
    means = frame.groupby(["material", "wall_cm"])["shape_re"].agg(["mean", "std", "count"])
    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    colors = {"conductor": "C0", "insulator": "C1"}
    for material, (x, y, model, grid) in curves.items():
        summary = means.loc[material]
        axis.errorbar(
            summary.index,
            summary["mean"],
            yerr=summary["std"] / np.sqrt(summary["count"]),
            fmt="o",
            color=colors[material],
            label=f"{material} mean",
            capsize=3,
        )
        if model == "linear":
            slope, intercept = np.polyfit(x, y, 1)
            hat = slope * grid + intercept
        elif model == "quadratic":
            coef = np.polyfit(x, y, 2)
            hat = np.polyval(coef, grid)
        else:
            def exponential(distance, amplitude, rate, offset):
                return amplitude * np.exp(rate * distance) + offset

            params, _ = curve_fit(
                exponential,
                x,
                y,
                p0=(float(y.max() - y.min()), -0.2, float(y.min())),
                maxfev=20000,
            )
            hat = exponential(grid, *params)
        r2 = float(selected.loc[(selected.material == material) & selected.selected, "r_squared"].iloc[0])
        axis.plot(grid, hat, color=colors[material], label=f"{material} {model} (R²={r2:.2f})")
    axis.set_xlabel("Distance from wall (cm)")
    axis.set_ylabel("Shape relative error")
    axis.set_title("Shape RE versus distance from the wall")
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "shape_re_vs_wall.png", dpi=150)
    plt.close(fig)


def main() -> None:
    frame = load_long()
    print(
        f"clocks={frame.clock.nunique()} materials={frame.material.nunique()} "
        f"depths={frame.wall_cm.nunique()} frequencies={frame.freq_khz.nunique()} "
        f"rows={len(frame)}"
    )
    table = repeated_measures(frame)
    table.to_csv(RESULT_DIR / "anova_shape_re.csv", index=False)
    print(table.to_string(index=False))
    selected, curves = regression(frame)
    selected.to_csv(RESULT_DIR / "regression_shape_re.csv", index=False)
    print(selected.to_string(index=False))
    draw(frame, curves, selected)
    print("saved", RESULT_DIR / "anova_shape_re.csv")
    print("saved", RESULT_DIR / "shape_re_vs_wall.png")


if __name__ == "__main__":
    main()
