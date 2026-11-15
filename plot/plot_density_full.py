
"""Plot spin-resolved local filling factors and orbital occupations.

Input: orb_occ_exp.csv with columns j,ndn,nup,ntot (j=-1 is a total row).
The 2D maps assume a state with conserved integer guiding-center momentum G,
so that off-diagonal orbital density-matrix elements vanish.

Use lengths in magnetic-length units (l_B = 1).
"""

import argparse
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable


def read_gamma_from_config(path: Path) -> float:
    if not path.is_file():
        raise FileNotFoundError(f"config.txt not found: {path}; use --gamma instead")
    match = re.search(
        r"^\s*aspect\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*$",
        path.read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )
    if match is None:
        raise ValueError(f"Could not find 'aspect = ...' in {path}")
    return float(match.group(1))


def load_occupations(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"j", "nup", "ndn", "ntot"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing CSV columns: {sorted(missing)}")

    df = df.loc[df["j"] >= 1, ["j", "nup", "ndn", "ntot"]].copy()
    df = df.sort_values("j")
    if df.empty:
        raise ValueError("No orbital occupation data after removing j=-1 row")
    if not np.array_equal(df["j"].to_numpy(), np.arange(1, len(df) + 1)):
        raise ValueError("Orbital indices j must be consecutive integers from 1 to N")
    return df


def calculate_profiles(df: pd.DataFrame, gamma: float, nx: int) -> dict:
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    if nx < 2:
        raise ValueError("nx must be at least 2")

    N = len(df)
    Lx = np.sqrt(2 * np.pi * N * gamma)
    Ly = np.sqrt(2 * np.pi * N / gamma)

    # Orbital centers: X_j = (j - 1/2) Lx / N
    xj = (df["j"].to_numpy() - 0.5) * Lx / N
    x = np.linspace(0.0, Lx, nx)

    # |phi_j(x,y)|^2 = exp[-(x-X_j)^2] / (sqrt(pi) * Ly)
    # with l_B = 1.
    envelope = np.exp(-((x[:, None] - xj[None, :]) ** 2))

    # Local filling factors: nu_sigma = 2*pi*rho_sigma
    factor = 2 * np.pi / (np.sqrt(np.pi) * Ly)

    nu_up = factor * (envelope @ df["nup"].to_numpy())
    nu_dn = factor * (envelope @ df["ndn"].to_numpy())
    nu_tot = factor * (envelope @ df["ntot"].to_numpy())

    return {
        "N": N,
        "Lx": Lx,
        "Ly": Ly,
        "xj": xj,
        "x": x,
        "nu_up": nu_up,
        "nu_dn": nu_dn,
        "nu_tot": nu_tot,
    }



def save_figure(
    fig,
    prefix: Path,
    name: str,
    show: bool = False,
) -> None:
    # Create output directory
    prefix.mkdir(parents=True, exist_ok=True)

    for extension in ("png", "pdf"):
        output_path = prefix / f"{name}.{extension}"

        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )
        print(f"Saved: {output_path}")

    if show:
        plt.show()

    plt.close(fig)



def plot_nu_2d(
    profile,
    Lx: float,
    Ly: float,
    ny: int,
    title: str,
    cmap: str,
    vmax: float,
    prefix: Path,
    name: str,
    show: bool,
) -> None:
    if ny < 2:
        raise ValueError("ny must be at least 2")

    # Density is uniform along y for fixed G.
    image = np.broadcast_to(profile[None, :], (ny, len(profile)))

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Round vmax up to the next multiple of 0.1
    vmax_plot = max(0.1, np.ceil(vmax / 0.1) * 0.1)

    im = ax.imshow(
        image,
        origin="lower",
        extent=(0.0, Lx, 0.0, Ly),
        cmap=cmap,
        vmin=0.0,
        vmax=vmax_plot,
        aspect="equal",
        interpolation="nearest",
    )

    ax.set_xlabel(r"$x/\ell_B$")
    ax.set_ylabel(r"$y/\ell_B$")
    ax.set_title(title)
    ax.tick_params(direction="in", top=True, right=True)

    # Colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.08)

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(title)

    # Ticks every 0.1
    ticks = np.arange(0.0, vmax_plot + 0.05, 0.1)
    cbar.set_ticks(ticks)

    fig.tight_layout()
    save_figure(fig, prefix, name, show=show)


def plot_orb_occ_1d(
    df: pd.DataFrame,
    xj,
    Lx: float,
    prefix: Path,
    show: bool,
) -> None:

    fig, ax = plt.subplots(figsize=(7, 4.5))

    lines = (
        ("nup",  r"$\langle n_{j\uparrow}\rangle$",   "tab:red"),
        ("ndn",  r"$\langle n_{j\downarrow}\rangle$", "tab:blue"),
        ("ntot", r"$\langle n_j\rangle$",             "tab:orange"),
    )

    for column, label, color in lines:
        ax.plot(
            xj,
            df[column],
            "-o",
            color=color,
            linewidth=1.5,
            markersize=2.8,
            label=label,
        )

    ax.set_xlabel(r"$x/\ell_B$")
    ax.set_ylabel("Orbital occupation")

    ax.set_xlim(0.0, Lx)
    ax.set_ylim(0.0, 1.0)  # 縦軸を 0-1.0 に固定

    ax.tick_params(direction="in", top=True, right=True)

    # Reference line: nu = 1/3
    ax.axhline(
        y=1/3,
        color="black",
        linestyle="--",
        linewidth=1.2,
        alpha=0.7,
        zorder=1,
    )

    # 凡例を中央に配置
    ax.legend(
        frameon=False,
        ncol=3,
        loc="upper center",
    )

    fig.tight_layout()
    save_figure(fig, prefix, "orb_occ_1d", show=show)

def plot_summary_4panel(
    df: pd.DataFrame,
    data: dict,
    prefix: Path,
    show: bool,
) -> None:
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    Lx = data["Lx"]
    Ly = data["Ly"]
    xj = data["xj"]

    # 2D 表示用に y 方向へ複製
    ny = 250
    image_up = np.broadcast_to(data["nu_up"][None, :], (ny, len(data["nu_up"])))
    image_dn = np.broadcast_to(data["nu_dn"][None, :], (ny, len(data["nu_dn"])))
    image_tot = np.broadcast_to(data["nu_tot"][None, :], (ny, len(data["nu_tot"])))

    # up / dn は同じ color scale
    spin_vmax = max(
        float(data["nu_up"].max()),
        float(data["nu_dn"].max()),
    )
    spin_vmax_plot = max(0.1, np.ceil(spin_vmax / 0.1) * 0.1)

    tot_vmax = float(data["nu_tot"].max())
    tot_vmax_plot = max(0.1, np.ceil(tot_vmax / 0.1) * 0.1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # --------------------------------------------------
    # up
    # --------------------------------------------------
    ax = axes[0, 0]
    im = ax.imshow(
        image_up,
        origin="lower",
        extent=(0.0, Lx, 0.0, Ly),
        cmap="Reds",
        vmin=0.0,
        vmax=spin_vmax_plot,
        aspect="equal",
        interpolation="nearest",
    )
    ax.set_title(r"$\nu_{\uparrow}(x,y)$")
    ax.set_xlabel(r"$x/\ell_B$")
    ax.set_ylabel(r"$y/\ell_B$")
    ax.tick_params(direction="in", top=True, right=True)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.08)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(r"$\nu_{\uparrow}(x,y)$")
    cbar.set_ticks(np.arange(0.0, spin_vmax_plot + 0.05, 0.1))

    # --------------------------------------------------
    # dn
    # --------------------------------------------------
    ax = axes[0, 1]
    im = ax.imshow(
        image_dn,
        origin="lower",
        extent=(0.0, Lx, 0.0, Ly),
        cmap="Blues",
        vmin=0.0,
        vmax=spin_vmax_plot,
        aspect="equal",
        interpolation="nearest",
    )
    ax.set_title(r"$\nu_{\downarrow}(x,y)$")
    ax.set_xlabel(r"$x/\ell_B$")
    ax.set_ylabel(r"$y/\ell_B$")
    ax.tick_params(direction="in", top=True, right=True)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.08)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(r"$\nu_{\downarrow}(x,y)$")
    cbar.set_ticks(np.arange(0.0, spin_vmax_plot + 0.05, 0.1))

    # --------------------------------------------------
    # tot
    # --------------------------------------------------
    ax = axes[1, 0]
    im = ax.imshow(
        image_tot,
        origin="lower",
        extent=(0.0, Lx, 0.0, Ly),
        # cmap="viridis",
        # cmap="Purples",
        # cmap="Oranges",
        cmap="Greens",
        vmin=0.0,
        vmax=tot_vmax_plot,
        aspect="equal",
        interpolation="nearest",
    )
    ax.set_title(r"$\nu(x,y)$")
    ax.set_xlabel(r"$x/\ell_B$")
    ax.set_ylabel(r"$y/\ell_B$")
    ax.tick_params(direction="in", top=True, right=True)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.08)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(r"$\nu(x,y)$")
    cbar.set_ticks(np.arange(0.0, tot_vmax_plot + 0.05, 0.1))

    # --------------------------------------------------
    # orb_occ
    # --------------------------------------------------
    ax = axes[1, 1]

    lines = (
        ("nup",  r"$\langle n_{j\uparrow}\rangle$",   "tab:red"),
        ("ndn",  r"$\langle n_{j\downarrow}\rangle$", "tab:blue"),
        # ("ntot", r"$\langle n_j\rangle$",             "tab:orange"),
        ("ntot", r"$\langle n_j\rangle$",             "tab:green"),
    )

    for column, label, color in lines:
        ax.plot(
            xj,
            df[column],
            "-o",
            color=color,
            linewidth=1.5,
            markersize=2.8,
            label=label,
        )

    ax.axhline(
        y=1/3,
        color="black",
        linestyle="--",
        linewidth=1.2,
        alpha=0.7,
        zorder=1,
    )

    ax.set_title("Orbital occupation")
    ax.set_xlabel(r"$x/\ell_B$")
    ax.set_ylabel("Orbital occupation")
    ax.set_xlim(0.0, Lx)
    ax.set_ylim(0.0, 1.0)
    ax.tick_params(direction="in", top=True, right=True)
    ax.legend(frameon=False, ncol=3, loc="upper center")

    fig.tight_layout()
    save_figure(fig, prefix, "summary_4panel", show=show)


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Generate three individual 2D local filling factor "
            "plots and one 1D orbital-occupation plot."
        )
    )

    parser.add_argument(
        "csv",
        type=Path,
        help="Input orb_occ_exp.csv",
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory prefix (default: <csv_dir>/figures)",
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="Aspect gamma; otherwise read config.txt",
    )

    parser.add_argument(
        "--nx",
        type=int,
        default=600,
    )

    parser.add_argument(
        "--ny",
        type=int,
        default=250,
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Show figures interactively",
    )

    args = parser.parse_args()

    # --- Load data -----------------------------------------
    df = load_occupations(args.csv)

    gamma = (
        args.gamma
        if args.gamma is not None
        else read_gamma_from_config(args.csv.parent / "config.txt")
    )

    data = calculate_profiles(df, gamma, args.nx)

    prefix = args.out or args.csv.parent / "figures"

    # --- Plot settings ------------------------------------
    plt.rcParams.update({
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "font.size": 16,
        "axes.labelsize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
    })

    # Use a common color scale for up and down.
    spin_vmax = max(
        float(data["nu_up"].max()),
        float(data["nu_dn"].max()),
    )

    specs = (
        ("nu_up",  r"$\nu_{\uparrow}(x,y)$",   "Reds",     spin_vmax),
        ("nu_dn",  r"$\nu_{\downarrow}(x,y)$", "Blues",    spin_vmax),
        ("nu_tot", r"$\nu(x,y)$",              "viridis",
        float(data["nu_tot"].max())),
    )

    # --- 2D filling factor plots --------------------------
    for name, title, cmap, vmax in specs:
        plot_nu_2d(
            data[name],
            data["Lx"],
            data["Ly"],
            args.ny,
            title,
            cmap,
            vmax,
            prefix,
            f"{name}_2d",
            args.show,
        )

    # --- 1D orbital occupation plot -----------------------
    plot_orb_occ_1d(
        df,
        data["xj"],
        data["Lx"],
        prefix,
        args.show,
    )

    # --- 4-panel summary figure ------------------------
    plot_summary_4panel(
        df,
        data,
        prefix,
        args.show,
    )

    print(
        f"N={data['N']}, "
        f"gamma={gamma}, "
        f"Lx/lB={data['Lx']:.6f}, "
        f"Ly/lB={data['Ly']:.6f}"
    )


if __name__ == "__main__":
    main()


"""
python plot/plot_density_full.py \
  results/fqh_ng_cylinder/edge/no_dep_sk/n120/asp2.0/m1000/orb_occ_exp.csv \
  --out plot/figures/n120_asp2.0_m1000
"""