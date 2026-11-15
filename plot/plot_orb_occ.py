import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_gamma_from_config(csv_path: Path) -> float:
    """Read main.jl's aspect (= gamma) from config.txt next to the CSV."""
    config_path = csv_path.parent / "config.txt"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Could not find {config_path}. Specify --gamma explicitly."
        )

    for line in config_path.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition("=")
        if sep and key.strip() == "aspect":
            return float(value.strip())

    raise ValueError(f"No 'aspect = ...' setting found in {config_path}.")


def plot_orb_occ(
    csv_path: Path,
    out_prefix: Path | None = None,
    gamma: float | None = None,
    show: bool = False,
) -> None:
    df = pd.read_csv(csv_path)
    required = {"j", "nup", "ndn", "ntot"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    # j=-1 is the sum of occupations, not an orbital.
    df = df.loc[df["j"] >= 1].sort_values("j").copy()
    if df.empty:
        raise ValueError("No orbital occupation data found.")

    # The CSV has one data row per orbital; its orbital count is N.
    N = len(df)
    if not np.array_equal(df["j"].to_numpy(), np.arange(1, N + 1)):
        raise ValueError("Orbital indices j must run consecutively from 1 to N.")

    if gamma is None:
        gamma = read_gamma_from_config(csv_path)
    if not math.isfinite(gamma) or gamma <= 0:
        raise ValueError(f"gamma must be positive and finite (got {gamma}).")

    # All lengths are in magnetic length units (l_B).
    Lx_over_lB = math.sqrt(2 * math.pi * N * gamma)
    x_over_lB = (df["j"].to_numpy() - 1/2) * Lx_over_lB / N

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
    # curves = [
    #     ("nup",  r"$\langle n_{j\uparrow}\rangle$",   "red"),
    #     ("ndn",  r"$\langle n_{j\downarrow}\rangle$", "blue"),
    #     ("ntot", r"$\langle n_j\rangle$",             "orange"),
    # ]
    curves = [
        ("nup",  r"$\langle n_{j\uparrow}\rangle$",   "tab:red"),
        ("ndn",  r"$\langle n_{j\downarrow}\rangle$", "tab:blue"),
        ("ntot", r"$\langle n_j\rangle$",             "tab:orange"),
    ]
    for column, label, color in curves:
        ax.plot(
            x_over_lB,
            df[column],
            "-o",
            color=color,
            linewidth=1.5,
            markersize=3,
            label=label,
        )

    ax.set_xlabel(r"$x/\ell_B$")
    ax.set_ylabel("Orbital occupation")
    ax.set_xlim(0, Lx_over_lB)

    ymax = max(float(df[["nup", "ndn", "ntot"]].max().max()), 1.0)
    ax.set_ylim(0, ymax * 1.05)
    ax.tick_params(direction="in", top=True, right=True)
    ax.legend(frameon=False, ncol=3, loc="upper right")
    fig.tight_layout()

    if out_prefix is None:
        out_prefix = csv_path.with_name("orb_occ")
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    for extension in ("png", "pdf"):
        output_path = Path(f"{out_prefix}.{extension}")
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

        print(f"Saved: {output_path}")

    print(f"N = {N}, gamma = {gamma}, Lx/lB = {Lx_over_lB:.6f}")

    if show:
        plt.show()

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot FQH orbital occupations against x/l_B."
    )

    parser.add_argument(
        "csv",
        type=Path,
        help="Path to orb_occ_exp.csv",
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output prefix without extension",
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="Aspect ratio gamma (read from config.txt by default)",
    )

    parser.add_argument(
        "--show",
        action="store_true",
    )

    args = parser.parse_args()

    plot_orb_occ(
        args.csv,
        args.out,
        args.gamma,
        args.show,
    )


if __name__ == "__main__":
    main()
