#!/usr/bin/env python3
"""Plot cross sections and (optionally) heatmaps of spin-resolved <n_i n_j>.

Input: ninj_corr.csv with columns i,j,upup,updn,dnup,dndn,ntot
       for every 1 <= i <= j <= N (including i=j).
For connected correlations, orb_occ_exp.csv is also required:
       columns j,ndn,nup,ntot (optional j=-1 sum row).

Run from the project root:
    python plot_ninj.py
or:
    python plot_ninj.py 'results/.../n24/asp*/m1000/ninj_corr.csv' \
        --out plot/figures/n24_ninj --mode both --refs center,peak
"""

import argparse
import glob
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Defaults for running simply: python plot_ninj.py
# Change these for your calculation. Command-line arguments override them.
# ---------------------------------------------------------------------------
DEFAULT_CSV_PATTERN = (
    "results/fqh_ng_cylinder/edge/no_dep_n210r2.4_sk/"
    "n211/asp*/m1000/ninj_corr.csv"
)
DEFAULT_OUT = "plot/figures/n24_ninj_test"
DEFAULT_MODE = "both"       # raw / connected / both
DEFAULT_PLOT = "cross"      # cross / heatmap / both
DEFAULT_REFS = "center"     # e.g. center,peak,12


plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 15,
    "axes.labelsize": 17,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
    "axes.titlesize": 17,
})

SPIN_CHANNELS = (
    ("upup", r"$\uparrow\uparrow$", "tab:red", "Reds"),
    ("updn", r"$\uparrow\downarrow$", "mediumpurple", "Purples"),
    ("dnup", r"$\downarrow\uparrow$", "tab:orange", "Oranges"),
    ("dndn", r"$\downarrow\downarrow$", "tab:blue", "Blues"),
)
ALL_CHANNELS = (*SPIN_CHANNELS, ("ntot", r"$\mathrm{total}$", "black", "Greys"))


def resolve_csv(pattern: str) -> Path:
    """Expand asp* and require exactly one matching input file."""
    matches = sorted(Path(p) for p in glob.glob(pattern) if Path(p).is_file())
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one ninj_corr.csv, found {len(matches)} for:\n"
            f"  {pattern}\nMatches: {matches}\n"
            "Check the path or narrow down the asp* wildcard."
        )
    return matches[0]


def load_correlations(path: Path) -> dict[str, np.ndarray]:
    df = pd.read_csv(path)
    required = ["i", "j", "upup", "updn", "dnup", "dndn", "ntot"]
    if missing := set(required) - set(df.columns):
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    if df.empty or df[required].isna().any().any():
        raise ValueError(f"Empty CSV or NaN values: {path}")

    i_values = df["i"].to_numpy(dtype=float)
    j_values = df["j"].to_numpy(dtype=float)
    if (not np.isfinite(i_values).all() or not np.isfinite(j_values).all()
            or not np.all(i_values == np.floor(i_values))
            or not np.all(j_values == np.floor(j_values))):
        raise ValueError("Orbital indices must be finite integers")

    ii, jj = i_values.astype(int), j_values.astype(int)
    n = int(max(ii.max(), jj.max()))
    if np.any(ii < 1) or np.any(ii > jj) or np.any(jj > n):
        raise ValueError("Expected 1 <= i <= j <= N")
    expected = n * (n + 1) // 2
    if len(df) != expected or df.duplicated(["i", "j"]).any():
        raise ValueError(
            f"Need all i<=j entries including diagonal: "
            f"N={n}, expected={expected}, found={len(df)}"
        )

    values = {k: df[k].to_numpy(dtype=float) for k in required[2:]}
    if any(not np.isfinite(v).all() for v in values.values()):
        raise ValueError("Correlation values must be finite")
    a, b = ii - 1, jj - 1
    out = {key: np.empty((n, n), dtype=float) for key in values}

    # Same-spin and total density correlations are symmetric.
    for key in ("upup", "dndn", "ntot"):
        out[key][a, b] = values[key]
        out[key][b, a] = values[key]

    # Cross-spin correlations transform into each other under i <-> j.
    # For i=j, updn=dnup; check rather than silently overwrite.
    diagonal = (a == b)
    diag_error = float(np.max(np.abs(
        values["updn"][diagonal] - values["dnup"][diagonal]
    )))
    if diag_error > 1e-7:
        raise ValueError(f"Cross-spin diagonal mismatch: {diag_error:.3e}")
    out["updn"][a, b] = values["updn"]
    out["updn"][b, a] = values["dnup"]
    out["dnup"] = out["updn"].T.copy()

    sum_spin = out["upup"] + out["updn"] + out["dnup"] + out["dndn"]
    error = float(np.max(np.abs(sum_spin - out["ntot"])))
    if error > 1e-7:
        raise ValueError(f"Spin components do not sum to ntot: {error:.3e}")
    print(f"Loaded {path} (N={n}, rows={len(df)}, max error={error:.2e})")
    return out


def read_gamma(directory: Path, override: float | None) -> float | None:
    if override is not None:
        if not np.isfinite(override) or override <= 0:
            raise ValueError("--gamma must be finite and positive")
        return override
    config = directory / "config.txt"
    if config.is_file():
        for line in config.read_text(encoding="utf-8").splitlines():
            left, sep, right = line.partition("=")
            if sep and left.strip() == "aspect":
                gamma = float(right.split("#", 1)[0].strip())
                if not np.isfinite(gamma) or gamma <= 0:
                    raise ValueError(f"Invalid aspect in {config}")
                return gamma
    return None


def load_occupations(path: Path, n: int) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. --mode connected/both and --refs peak need "
            "orb_occ_exp.csv in the CSV directory (or --orb-occ PATH)."
        )
    df = pd.read_csv(path)
    if missing := {"j", "ndn", "nup", "ntot"} - set(df.columns):
        raise ValueError(f"Missing occupation columns: {sorted(missing)}")
    df = df[df["j"] >= 1].sort_values("j").reset_index(drop=True)
    if len(df) != n or not np.array_equal(df["j"].to_numpy(), np.arange(1, n + 1)):
        raise ValueError("Occupation CSV must contain exactly j=1,...,N")
    if not np.isfinite(df[["ndn", "nup", "ntot"]].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite orbital occupations")
    return df


def connected_correlations(raw: dict[str, np.ndarray], occ: pd.DataFrame):
    up = occ["nup"].to_numpy()
    dn = occ["ndn"].to_numpy()
    tot = occ["ntot"].to_numpy()
    return {
        "upup": raw["upup"] - np.outer(up, up),
        "updn": raw["updn"] - np.outer(up, dn),
        "dnup": raw["dnup"] - np.outer(dn, up),
        "dndn": raw["dndn"] - np.outer(dn, dn),
        "ntot": raw["ntot"] - np.outer(tot, tot),
    }


def reference_indices(spec: str, n: int, occ: pd.DataFrame | None) -> list[int]:
    refs = []
    for token in spec.split(","):
        token = token.strip().lower()
        if token == "center":
            i0 = (n + 1) // 2  # even N: left of the two central orbitals
        elif token == "center-right":
            i0 = (n + 2) // 2
        elif token == "peak":
            if occ is None:
                raise ValueError("--refs peak requires orb_occ_exp.csv")
            i0 = int(occ.loc[occ["ntot"].idxmax(), "j"])
        else:
            try:
                i0 = int(token)
            except ValueError as exc:
                raise ValueError(f"Invalid reference orbital: {token}") from exc
        if not 1 <= i0 <= n:
            raise ValueError(f"Reference orbital {i0} must lie in [1, {n}]")
        if i0 not in refs:
            refs.append(i0)
    if not refs:
        raise ValueError("At least one reference orbital is required")
    return refs


def horizontal_axis(n: int, i0: int, gamma: float | None, axis: str):
    js = np.arange(1, n + 1)
    if gamma is None:
        x = js - i0 if axis == "relative" else js
        label = r"Orbital separation $j-i_0$" if axis == "relative" else r"Orbital index $j$"
    else:
        dx = np.sqrt(2 * np.pi * gamma / n)
        x = (js - i0) * dx if axis == "relative" else (js - 0.5) * dx
        label = (r"$(x_j-x_{i_0})/\ell_B$" if axis == "relative"
                 else r"$x_j/\ell_B$")
    return x, label


def save_figure(fig, outdir: Path, stem: str, show: bool):
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = outdir / f"{stem}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved: {path}")
    if show:
        plt.show()
    plt.close(fig)


def draw_slice(ax, arrays, keys, x, xlabel, i0, mode, axis):
    for key, label, color, _ in ALL_CHANNELS:
        if key not in keys:
            continue
        y = arrays[key][i0 - 1, :].copy()
        y[i0 - 1] = np.nan  # Exclude i=j without replacing it by zero.
        ax.plot(x, y, "-o", color=color, lw=1.6, markersize=3.3, label=label)
    if axis == "relative":
        ax.axvline(0, color="gray", lw=1.0, ls="--", alpha=0.75)
    else:
        ax.axvline(x[i0 - 1], color="gray", lw=1.0, ls="--", alpha=0.75)
    if mode == "connected":
        ax.axhline(0, color="gray", lw=1.0, ls=":", alpha=0.8)
    ax.set_xlabel(xlabel)
    if mode == "connected":
        ax.set_ylabel(r"$\langle\delta n_{i_0}\delta n_j\rangle$")
    elif keys == {"ntot"}:
        ax.set_ylabel(r"$\langle n_{i_0}n_j\rangle$")
    else:
        ax.set_ylabel(r"$\langle n_{i_0\sigma}n_{j\sigma'}\rangle$")
    ax.set_xlim(float(x.min()), float(x.max()))
    ax.tick_params(direction="in", top=True, right=True)
    ax.legend(frameon=False, loc="best", ncol=2 if len(keys) > 1 else 1)


def plot_cross_sections(arrays, outdir: Path, i0: int, mode: str,
                        gamma: float | None, axis: str, individual: bool, show: bool):
    n = arrays["ntot"].shape[0]
    x, xlabel = horizontal_axis(n, i0, gamma, axis)
    prefix = "ninj" if mode == "raw" else "ninj_connected"
    base = f"{prefix}_cross_i{i0:03d}"

    # Main comparison figure: four spin components and total density.
    fig, (ax_spin, ax_total) = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    draw_slice(ax_spin, arrays, {"upup", "updn", "dnup", "dndn"}, x, xlabel, i0, mode, axis)
    draw_slice(ax_total, arrays, {"ntot"}, x, xlabel, i0, mode, axis)
    ax_spin.set_title("Spin-resolved correlation")
    ax_total.set_title("Total density correlation")
    fig.suptitle(rf"Reference orbital $i_0={i0}$  ($j\ne i_0$)")
    save_figure(fig, outdir, base + "_summary", show)

    # Optional separate files for each spin channel.
    if individual:
        for key, _, _, _ in ALL_CHANNELS:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            draw_slice(ax, arrays, {key}, x, xlabel, i0, mode, axis)
            ax.set_title(f"{key}: reference orbital $i_0={i0}$")
            fig.tight_layout()
            save_figure(fig, outdir, base + f"_{key}", show)


def plot_heatmaps(arrays, outdir: Path, mode: str, gamma: float | None, show: bool):
    n = arrays["ntot"].shape[0]
    if gamma is None:
        extent = (0.5, n + 0.5, 0.5, n + 0.5)
        xlabel, ylabel = r"Orbital index $i$", r"Orbital index $j$"
    else:
        lx = np.sqrt(2 * np.pi * n * gamma)
        extent = (0.0, lx, 0.0, lx)
        xlabel, ylabel = r"$x_i/\ell_B$", r"$x_j/\ell_B$"

    if mode == "connected":
        s = max(np.max(np.abs(arrays[k])) for k, *_ in SPIN_CHANNELS)
        t = float(np.max(np.abs(arrays["ntot"])))
        spin_norm = TwoSlopeNorm(vmin=-max(s, 1e-12), vcenter=0, vmax=max(s, 1e-12))
        total_norm = TwoSlopeNorm(vmin=-max(t, 1e-12), vcenter=0, vmax=max(t, 1e-12))
    else:
        spin_norm = Normalize(vmin=0, vmax=max(max(np.max(arrays[k]) for k, *_ in SPIN_CHANNELS), 1e-12))
        total_norm = Normalize(vmin=0, vmax=max(np.max(arrays["ntot"]), 1e-12))

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.7), constrained_layout=True)
    for ax, (key, label, _, cmap) in zip(axes.flat, ALL_CHANNELS):
        norm = total_norm if key == "ntot" else spin_norm
        im = ax.imshow(
            arrays[key].T, origin="lower", extent=extent, aspect="equal",
            interpolation="nearest", cmap="RdBu_r" if mode == "connected" else cmap,
            norm=norm,
        )
        ax.set_title(label)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.tick_params(direction="in", top=True, right=True)
        cax = make_axes_locatable(ax).append_axes("right", size="4%", pad=0.08)
        fig.colorbar(im, cax=cax)
    axes.flat[-1].axis("off")
    prefix = "ninj" if mode == "raw" else "ninj_connected"
    save_figure(fig, outdir, prefix + "_summary_5panel", show)


def main():
    parser = argparse.ArgumentParser(description="Plot <n_i n_j> cross sections.")
    parser.add_argument("csv", nargs="?", default=DEFAULT_CSV_PATTERN,
                        help="Input ninj_corr.csv; glob asp* accepted")
    parser.add_argument("--out", type=Path, default=Path(DEFAULT_OUT),
                        help="Output directory")
    parser.add_argument("--mode", choices=("raw", "connected", "both"), default=DEFAULT_MODE)
    parser.add_argument("--plot", choices=("cross", "heatmap", "both"), default=DEFAULT_PLOT)
    parser.add_argument("--refs", default=DEFAULT_REFS,
                        help="Comma-separated reference orbitals: center,center-right,peak,12,...")
    parser.add_argument("--xaxis", choices=("relative", "absolute"), default="relative")
    parser.add_argument("--orb-occ", type=Path, default=None,
                        help="Override orb_occ_exp.csv for connected mode / peak reference")
    parser.add_argument("--gamma", type=float, default=None,
                        help="Aspect ratio; else read config.txt; else use orbital indices")
    parser.add_argument("--individual", action="store_true",
                        help="Also save separate cross-section plots for all five components")
    parser.add_argument("--show", action="store_true", help="Display figures interactively")
    args = parser.parse_args()

    csv = resolve_csv(args.csv)
    raw = load_correlations(csv)
    n = raw["ntot"].shape[0]
    gamma = read_gamma(csv.parent, args.gamma)
    print(f"gamma={gamma}, N={n}")

    need_occ = args.mode in ("connected", "both") or "peak" in args.refs.lower().split(",")
    occ = None
    if need_occ:
        occ_path = args.orb_occ if args.orb_occ is not None else csv.parent / "orb_occ_exp.csv"
        occ = load_occupations(occ_path, n)
    refs = reference_indices(args.refs, n, occ)
    print(f"Reference orbital(s): {refs}")

    for mode in (("raw", "connected") if args.mode == "both" else (args.mode,)):
        arrays = raw if mode == "raw" else connected_correlations(raw, occ)
        if args.plot in ("cross", "both"):
            for i0 in refs:
                plot_cross_sections(arrays, args.out, i0, mode, gamma,
                                    args.xaxis, args.individual, args.show)
        if args.plot in ("heatmap", "both"):
            plot_heatmaps(arrays, args.out, mode, gamma, args.show)


if __name__ == "__main__":
    main()
