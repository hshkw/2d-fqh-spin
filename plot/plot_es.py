#!/usr/bin/env python3
"""Plot entanglement spectra from an ITensor DMRG es.csv.

Expected columns:
    N,Sz,G,val   (spin-resolved)
or  N,G,val      (old spinless format)

Here val = -log(p), where p is the squared Schmidt coefficient.
The QN column Sz contains 2 * physical S_z.
No symmetry-sector shift of G is assumed unless explicitly requested.
"""

import argparse
import glob
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # For SSH/headless execution
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# Editable defaults: running `python plot/plot_es.py` uses these values.
# ----------------------------------------------------------------------
DEFAULT_CSV_PATTERN = (
    "results/fqh_ng_cylinder/edge/no_dep_n210r2.4/"
    "n179/asp*/m1000/es.csv"
)
DEFAULT_OUT = "plot/figures/n179fp_es"
DEFAULT_N_SECTORS = None        # e.g. [34, 35, 36], or None
DEFAULT_SZ_SECTORS = None       # ITensor QN 'Sz' = 2 * physical Sz; e.g. [0, 2]
DEFAULT_MAX_XI = 15.0          # None means no upper cut
DEFAULT_MAX_PANELS = 6         # automatic panel count when N unspecified
DEFAULT_RELATIVE_G = False     # plot G-G0, with G0 from lowest ES level

plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 15,
    "axes.labelsize": 17,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "axes.titlesize": 16,
    "legend.fontsize": 12,
})


def resolve_csv(pattern: str) -> Path:
    paths = [Path(p) for p in sorted(glob.glob(pattern, recursive=True)) if Path(p).is_file()]
    if not paths:
        raise FileNotFoundError(f"No es.csv matched: {pattern}")
    if len(paths) != 1:
        lines = "\n".join(str(p) for p in paths[:20])
        raise ValueError(f"Expected one es.csv, found {len(paths)}:\n{lines}")
    return paths[0]


def read_es(path: Path) -> tuple[pd.DataFrame, bool]:
    df = pd.read_csv(path)
    required = {"N", "G", "val"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")

    spinful = "Sz" in df.columns
    cols = ["N", "G", "val"] + (["Sz"] if spinful else [])
    df = df[cols].copy()

    for col in ("N", "G") + (("Sz",) if spinful else ()):
        df[col] = pd.to_numeric(df[col], errors="raise")
        if df[col].isna().any() or not np.isfinite(df[col]).all():
            raise ValueError(f"Non-finite quantum number in column {col}")
        if not np.all(df[col].to_numpy() == np.rint(df[col].to_numpy())):
            raise ValueError(f"Quantum numbers in {col} must be integers")
        df[col] = df[col].astype(int)

    df["val"] = pd.to_numeric(df["val"], errors="coerce")
    invalid = ~np.isfinite(df["val"].to_numpy(dtype=float))
    if invalid.any():
        print(f"Skipped {int(invalid.sum())} nonfinite ES levels")
        df = df.loc[~invalid].copy()
    if df.empty:
        raise ValueError("No finite entanglement-spectrum levels")
    if spinful:
        df["Sz_physical"] = df["Sz"] / 2.0
    return df, spinful


def select_es(df: pd.DataFrame, n_values, sz_values, max_xi):
    if n_values is not None:
        df = df.loc[df["N"].isin(n_values)]
    if sz_values is not None:
        if "Sz" not in df.columns:
            raise ValueError("--sz requires an es.csv with an Sz column")
        df = df.loc[df["Sz"].isin(sz_values)]
    if max_xi is not None:
        df = df.loc[df["val"] <= max_xi]
    if df.empty:
        raise ValueError("No ES levels survive the requested sector/energy filters")
    return df.copy()


def save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        file = out / f"{name}.{ext}"
        fig.savefig(file, dpi=300, bbox_inches="tight")
        print(f"Saved: {file}")
    plt.close(fig)


def spin_sector_palette(df: pd.DataFrame) -> dict[int, tuple]:
    """Assign a stable, distinct color to each available integer Sz sector."""
    if "Sz" not in df.columns:
        return {}
    sectors = sorted(int(x) for x in df["Sz"].unique())
    n = len(sectors)
    if n <= 10:
        cmap = plt.get_cmap("tab10")
        colors = [cmap(i) for i in range(n)]
    elif n <= 20:
        cmap = plt.get_cmap("tab20")
        colors = [cmap(i) for i in range(n)]
    else:
        # Many sectors are difficult to distinguish individually.
        print(f"Note: {n} spin sectors; consider --sz to select fewer sectors.")
        cmap = plt.get_cmap("turbo")
        colors = [cmap(v) for v in np.linspace(0.03, 0.97, n)]
    return dict(zip(sectors, colors))


def spin_sector_label(sz_qn: int) -> str:
    """Convert ITensor's integer Sz=2*S_z into a LaTeX label."""
    if sz_qn % 2 == 0:
        value = str(sz_qn // 2)
    else:
        value = rf"\frac{{{sz_qn}}}{{2}}"
    return rf"$S_{{z,A}}={value}$"


def add_spin_legend(ax, palette, df: pd.DataFrame):
    present = [sz for sz in palette if sz in set(df["Sz"].unique())]
    # In a very dense plot, keep the figure readable by placing a compact legend outside.
    ax.legend(
        loc="upper left", bbox_to_anchor=(1.01, 1.0),
        frameon=False, ncol=1 if len(present) <= 15 else 2,
        fontsize=11 if len(present) <= 12 else 9,
        markerscale=1.3, handletextpad=0.3, borderaxespad=0.0,
    )


def scatter_sector(ax, df: pd.DataFrame, palette: dict, g0: int):
    if palette:
        for sz, color in palette.items():
            part = df.loc[df["Sz"] == sz]
            if part.empty:
                continue
            ax.scatter(
                part["G"] - g0, part["val"],
                color=color, label=spin_sector_label(sz),
                s=18, linewidths=0, alpha=0.9, rasterized=True,
            )
    else:
        ax.scatter(
            df["G"] - g0, df["val"], color="black", s=18,
            linewidths=0, alpha=0.9, rasterized=True,
        )
    ax.set_xlabel(r"$G-G_0$" if g0 != 0 else r"$G$")
    ax.set_ylabel(r"$\xi=-\ln p$")
    ax.tick_params(direction="in", top=True, right=True)
    ax.set_ylim(bottom=0.0)


def plot_overview(df, out, spinful, g0, palette):
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    if spinful:
        scatter_sector(ax, df, palette, g0)
        ax.set_title("Entanglement spectrum: spin sectors")
        add_spin_legend(ax, palette, df)
    else:
        norm = Normalize(vmin=float(df["N"].min()) - 0.5,
                         vmax=float(df["N"].max()) + 0.5)
        sc = ax.scatter(
            df["G"] - g0, df["val"], c=df["N"], cmap="viridis",
            norm=norm, s=12, alpha=0.7, linewidths=0, rasterized=True,
        )
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label(r"$N_A$")
        ax.set_xlabel(r"$G-G_0$" if g0 != 0 else r"$G$")
        ax.set_ylabel(r"$\xi=-\ln p$")
        ax.tick_params(direction="in", top=True, right=True)
        ax.set_ylim(bottom=0)
        ax.set_title("Entanglement spectrum: all selected sectors")
    fig.tight_layout()
    save(fig, out, "es_overview")


def chosen_n_sectors(df: pd.DataFrame, explicit_n, max_panels: int) -> list[int]:
    available = sorted(int(n) for n in df["N"].unique())
    if explicit_n is not None:
        return available
    # Automatically prioritize sectors containing the lowest ES levels.
    minima = df.groupby("N")["val"].min().sort_values(kind="stable")
    chosen = sorted(int(x) for x in minima.index[:max_panels])
    print(f"Automatically selected N_A sectors for detailed plots: {chosen}")
    return chosen


def plot_per_n(df, out, spinful, g0, n_sectors, palette):
    for n in n_sectors:
        part = df.loc[df["N"] == n]
        fig, ax = plt.subplots(figsize=(7.0, 5.0))
        scatter_sector(ax, part, palette if spinful else {}, g0)
        ax.set_title(rf"Entanglement spectrum ($N_A={n}$)")
        if spinful:
            add_spin_legend(ax, palette, part)
        fig.tight_layout()
        save(fig, out, f"es_N{n}")


def plot_n_panels(df, out, spinful, g0, n_sectors, palette):
    if not n_sectors:
        return
    ncols = 2 if len(n_sectors) > 1 else 1
    nrows = math.ceil(len(n_sectors) / ncols)
    fig, axs = plt.subplots(
        nrows, ncols,
        figsize=(7.0 * ncols, 4.6 * nrows),
        squeeze=False, constrained_layout=True,
    )
    axes = list(axs.flat)
    for ax, n in zip(axes, n_sectors):
        part = df.loc[df["N"] == n]
        scatter_sector(ax, part, palette if spinful else {}, g0)
        ax.set_title(rf"$N_A={n}$")
        if spinful:
            add_spin_legend(ax, palette, part)
    for ax in axes[len(n_sectors):]:
        ax.set_visible(False)
    fig.suptitle("Entanglement spectrum by particle-number sector")
    save(fig, out, "es_by_N")


def main():
    parser = argparse.ArgumentParser(description="Plot ES from N,Sz,G,val CSV")
    parser.add_argument("csv", nargs="?", default=DEFAULT_CSV_PATTERN,
                        help="es.csv path (glob patterns like asp* supported)")
    parser.add_argument("--out", type=Path, default=Path(DEFAULT_OUT),
                        help="Directory for PNG/PDF output")
    parser.add_argument("--n", type=int, nargs="+", default=DEFAULT_N_SECTORS,
                        help="Select subsystem particle-number QN sectors")
    parser.add_argument("--sz", type=int, nargs="+", default=DEFAULT_SZ_SECTORS,
                        help="Select integer Sz QN sectors (Sz_QN=2*physical Sz)")
    parser.add_argument("--max-xi", type=float, default=DEFAULT_MAX_XI,
                        help="Maximum entanglement energy (default 15)")
    parser.add_argument("--no-xi-cut", action="store_true",
                        help="Include all finite entanglement energies")
    parser.add_argument("--max-panels", type=int, default=DEFAULT_MAX_PANELS,
                        help="Maximum automatic number of N sectors in detailed plots")
    parser.add_argument("--relative-g", action="store_true", default=DEFAULT_RELATIVE_G,
                        help="Shift G by the G of the lowest ES level")
    args = parser.parse_args()
    if args.max_panels < 1:
        parser.error("--max-panels must be >= 1")
    if not args.no_xi_cut and args.max_xi is not None and args.max_xi <= 0:
        parser.error("--max-xi must be > 0")

    csv = resolve_csv(args.csv)
    df, spinful = read_es(csv)
    print(f"Loaded {csv}: {len(df)} finite levels; spin resolved={spinful}")
    df = select_es(df, args.n, args.sz,
                   None if args.no_xi_cut else args.max_xi)
    g0 = int(df.loc[df["val"].idxmin(), "G"]) if args.relative_g else 0
    print(f"Plotting {len(df)} levels; G0={g0}")
    palette = spin_sector_palette(df)
    plot_overview(df, args.out, spinful, g0, palette)
    n_sectors = chosen_n_sectors(df, args.n, args.max_panels)
    plot_per_n(df, args.out, spinful, g0, n_sectors, palette)
    plot_n_panels(df, args.out, spinful, g0, n_sectors, palette)


if __name__ == "__main__":
    main()
