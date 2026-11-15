import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_gamma_from_config(config_path: Path) -> float:
    if not config_path.exists():
        raise FileNotFoundError(f"config.txt not found: {config_path}")

    text = config_path.read_text(encoding="utf-8")
    match = re.search(r"aspect\s*=\s*([0-9eE+\-\.]+)", text)
    if match is None:
        raise ValueError(f"Could not find 'aspect = ...' in {config_path}")
    return float(match.group(1))


def compute_density_2d(df: pd.DataFrame, gamma: float, nx: int = 400, ny: int = 200):
    # j=-1 は総和なので除外
    df = df.loc[df["j"] >= 1].copy()
    df = df.sort_values("j")

    N = len(df)
    if N == 0:
        raise ValueError("No valid orbital data found.")

    # geometry (l_B = 1)
    Lx = np.sqrt(2 * np.pi * N * gamma)
    Ly = np.sqrt(2 * np.pi * N / gamma)

    # orbital centers: X_j = (j - 1/2) Lx / N
    j = df["j"].to_numpy()
    Xj = (j - 0.5) * Lx / N

    ntot = df["ntot"].to_numpy()

    # grid
    x = np.linspace(0.0, Lx, nx)
    y = np.linspace(0.0, Ly, ny)

    # rho(x) = 1/(sqrt(pi) Ly) sum_j n_j exp[-(x-Xj)^2]
    gaussian = np.exp(-(x[:, None] - Xj[None, :]) ** 2)
    rho_x = gaussian @ ntot
    rho_x /= (np.sqrt(np.pi) * Ly)

    # 2D density: uniform along y
    rho_xy = np.tile(rho_x[:, None], (1, ny))

    return x, y, rho_xy, Lx, Ly


def plot_density_2d(
    csv_path: Path,
    out_prefix: Path | None = None,
    gamma: float | None = None,
    nx: int = 400,
    ny: int = 200,
    show: bool = False,
):
    df = pd.read_csv(csv_path)

    required = {"j", "ntot"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if gamma is None:
        config_path = csv_path.with_name("config.txt")
        gamma = read_gamma_from_config(config_path)

    x, y, rho_xy, Lx, Ly = compute_density_2d(df, gamma, nx=nx, ny=ny)

    plt.rcParams.update({
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "font.size": 16,
        "axes.labelsize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
    })

    fig, ax = plt.subplots(figsize=(7, 4.5))

    im = ax.imshow(
        rho_xy.T,
        origin="lower",
        extent=[0, Lx, 0, Ly],
        aspect="auto",
        cmap="viridis",
    )

    ax.set_xlabel(r"$x/\ell_B$")
    ax.set_ylabel(r"$y/\ell_B$")
    ax.tick_params(direction="in", top=True, right=True)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$\rho(x,y)\,\ell_B^2$")

    fig.tight_layout()

    if out_prefix is None:
        out_prefix = csv_path.with_name("density_2d")

    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    for ext in ("png", "pdf"):
        output_path = Path(f"{out_prefix}.{ext}")
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")

    print(f"N = {len(df[df['j'] >= 1])}")
    print(f"gamma = {gamma}")
    print(f"Lx/lB = {Lx:.6f}")
    print(f"Ly/lB = {Ly:.6f}")

    if show:
        plt.show()

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot 2D charge density from orb_occ_exp.csv."
    )
    parser.add_argument("csv", type=Path, help="Path to orb_occ_exp.csv")
    parser.add_argument("--out", type=Path, default=None, help="Output prefix without extension")
    parser.add_argument("--gamma", type=float, default=None, help="Aspect ratio gamma")
    parser.add_argument("--nx", type=int, default=400, help="Number of x grid points")
    parser.add_argument("--ny", type=int, default=200, help="Number of y grid points")
    parser.add_argument("--show", action="store_true", help="Show figure on screen")
    args = parser.parse_args()

    plot_density_2d(
        args.csv,
        out_prefix=args.out,
        gamma=args.gamma,
        nx=args.nx,
        ny=args.ny,
        show=args.show,
    )


if __name__ == "__main__":
    main()