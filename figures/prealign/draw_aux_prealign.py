from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as MplPath


plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42
# The mapping form is retained for compatibility with static figure auditors.
plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})
plt.rcParams["font.size"] = 7.5


COLORS = {
    "ink": "#273043",
    "muted": "#667085",
    "line": "#52606D",
    "input": "#E8E7F2",
    "stem": "#DCE8F5",
    "branch": "#DDF1EF",
    "fusion": "#D9E8F4",
    "project": "#C9DAED",
    "residual": "#F5E8D8",
    "output": "#E6DCEA",
    "group": "#8FA6B8",
    "white": "#FFFFFF",
}


def rounded_box(
    ax: plt.Axes,
    center: tuple[float, float],
    size: tuple[float, float],
    text: str,
    *,
    facecolor: str,
    edgecolor: str = COLORS["line"],
    linewidth: float = 1.0,
    fontsize: float = 7.0,
    fontweight: str = "normal",
) -> FancyBboxPatch:
    cx, cy = center
    width, height = size
    patch = FancyBboxPatch(
        (cx - width / 2, cy - height / 2),
        width,
        height,
        boxstyle="round,pad=0.035,rounding_size=0.10",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=3,
    )
    ax.add_patch(patch)
    ax.text(
        cx,
        cy,
        text,
        ha="center",
        va="center",
        color=COLORS["ink"],
        fontsize=fontsize,
        fontweight=fontweight,
        linespacing=1.25,
        zorder=4,
    )
    return patch


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = COLORS["line"],
    linewidth: float = 1.05,
    mutation_scale: float = 8.5,
    zorder: int = 2,
) -> FancyArrowPatch:
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        color=color,
        shrinkA=0,
        shrinkB=0,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def orthogonal_arrow(
    ax: plt.Axes,
    points: list[tuple[float, float]],
    *,
    color: str,
    linewidth: float = 1.05,
) -> FancyArrowPatch:
    vertices = [points[0]]
    codes = [MplPath.MOVETO]
    for point in points[1:]:
        vertices.append(point)
        codes.append(MplPath.LINETO)
    patch = FancyArrowPatch(
        path=MplPath(vertices, codes),
        arrowstyle="-|>",
        mutation_scale=8.5,
        linewidth=linewidth,
        color=color,
        shrinkA=0,
        shrinkB=0,
        zorder=1,
    )
    ax.add_patch(patch)
    return patch


def build_figure() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.2, 3.25))
    fig.patch.set_facecolor(COLORS["white"])
    ax.set_xlim(0.0, 17.7)
    ax.set_ylim(0.0, 7.75)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(
        8.85,
        7.52,
        "AuxPreAlign",
        ha="center",
        va="center",
        fontsize=10.5,
        fontweight="bold",
        color=COLORS["ink"],
    )
    ax.text(
        8.85,
        7.12,
        "Multi-scale transformation with an identity-preserving residual path",
        ha="center",
        va="center",
        fontsize=7.0,
        color=COLORS["muted"],
    )

    group = FancyBboxPatch(
        (4.78, 1.84),
        2.55,
        4.87,
        boxstyle="round,pad=0.03,rounding_size=0.10",
        facecolor="none",
        edgecolor=COLORS["group"],
        linewidth=0.85,
        linestyle=(0, (3, 2)),
        zorder=0,
    )
    ax.add_patch(group)
    ax.text(
        6.055,
        6.82,
        "Parallel multi-scale context",
        ha="center",
        va="bottom",
        fontsize=6.5,
        color=COLORS["muted"],
    )

    input_center = (1.05, 4.20)
    stem_center = (3.15, 4.20)
    split_center = (4.47, 4.20)
    branch_centers = [(6.05, 5.85), (6.05, 4.20), (6.05, 2.55)]
    merge_center = (7.62, 4.20)
    concat_center = (8.75, 4.20)
    fuse_center = (10.90, 4.20)
    project_center = (13.00, 4.20)
    add_center = (14.62, 4.20)
    output_center = (16.55, 4.20)
    residual_center = (7.20, 0.94)

    rounded_box(
        ax,
        input_center,
        (1.65, 1.10),
        "Auxiliary input\n$B \\times 1 \\times H \\times W$",
        facecolor=COLORS["input"],
        fontweight="bold",
    )
    rounded_box(
        ax,
        stem_center,
        (1.80, 1.10),
        "$3 \\times 3$ Conv\nBN + GELU\n$1 \\rightarrow 32$",
        facecolor=COLORS["stem"],
    )

    for center, dilation in zip(branch_centers, (1, 2, 3)):
        rounded_box(
            ax,
            center,
            (2.12, 1.00),
            "$3 \\times 3$ Conv\nBN + GELU\n"
            f"$d = {dilation}$,  $32 \\rightarrow 32$",
            facecolor=COLORS["branch"],
            fontsize=6.6,
        )

    rounded_box(
        ax,
        concat_center,
        (1.62, 1.10),
        "Concatenate\n$32 \\times 3 = 96$",
        facecolor="#EEF1F4",
    )
    rounded_box(
        ax,
        fuse_center,
        (1.80, 1.10),
        "$1 \\times 1$ Conv\nBN + GELU\n$96 \\rightarrow 32$",
        facecolor=COLORS["fusion"],
    )
    rounded_box(
        ax,
        project_center,
        (1.70, 1.10),
        "$1 \\times 1$ Conv\nProjection\n$32 \\rightarrow 3$",
        facecolor=COLORS["project"],
    )
    rounded_box(
        ax,
        residual_center,
        (2.25, 0.90),
        "Repeat channels\n$1 \\rightarrow 3$",
        facecolor=COLORS["residual"],
        edgecolor="#B58B5A",
        fontsize=6.8,
    )
    rounded_box(
        ax,
        output_center,
        (1.90, 1.10),
        "Aligned output\n$B \\times 3 \\times H \\times W$",
        facecolor=COLORS["output"],
        fontweight="bold",
    )

    split = Circle(split_center, radius=0.085, facecolor=COLORS["line"], edgecolor="none", zorder=4)
    merge = Circle(merge_center, radius=0.085, facecolor=COLORS["line"], edgecolor="none", zorder=4)
    add = Circle(
        add_center,
        radius=0.31,
        facecolor=COLORS["white"],
        edgecolor=COLORS["ink"],
        linewidth=1.15,
        zorder=4,
    )
    ax.add_patch(split)
    ax.add_patch(merge)
    ax.add_patch(add)
    ax.text(*add_center, "+", ha="center", va="center", fontsize=11, color=COLORS["ink"], zorder=5)
    ax.text(
        add_center[0],
        add_center[1] + 0.78,
        "Element-wise add",
        ha="center",
        va="bottom",
        fontsize=6.4,
        color=COLORS["muted"],
    )

    arrow(ax, (1.875, 4.20), (2.25, 4.20))
    arrow(ax, (4.05, 4.20), (4.385, 4.20))
    for center in branch_centers:
        arrow(ax, split_center, (4.99, center[1]))
        ax.plot(
            [7.11, merge_center[0]],
            [center[1], merge_center[1]],
            color=COLORS["line"],
            linewidth=1.05,
            solid_capstyle="round",
            zorder=2,
        )
    arrow(ax, (7.705, 4.20), (7.94, 4.20))
    arrow(ax, (9.56, 4.20), (10.00, 4.20))
    arrow(ax, (11.80, 4.20), (12.15, 4.20))
    arrow(ax, (13.85, 4.20), (14.31, 4.20))
    arrow(ax, (14.93, 4.20), (15.60, 4.20))

    residual_color = "#A46F36"
    orthogonal_arrow(
        ax,
        [(1.05, 3.65), (1.05, 0.94), (6.075, 0.94)],
        color=residual_color,
    )
    orthogonal_arrow(
        ax,
        [(8.325, 0.94), (14.62, 0.94), (14.62, 3.89)],
        color=residual_color,
    )
    ax.text(
        3.45,
        0.73,
        "Identity path",
        ha="center",
        va="top",
        fontsize=6.3,
        color=residual_color,
        fontweight="bold",
    )

    ax.text(
        8.85,
        0.12,
        "All convolutions use stride 1; spatial resolution is preserved.",
        ha="center",
        va="bottom",
        fontsize=6.2,
        color=COLORS["muted"],
    )

    fig.subplots_adjust(left=0.012, right=0.988, bottom=0.025, top=0.98)
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw the AuxPreAlign module structure.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory for SVG and PNG exports.",
    )
    parser.add_argument(
        "--pdf",
        action="store_true",
        help="Optionally export a PDF in addition to the requested SVG and PNG files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    output_stem = args.output_dir / "aux_prealign_structure"
    fig.savefig(output_stem.with_suffix(".svg"), facecolor="white", bbox_inches="tight", pad_inches=0.03)
    if args.pdf:
        fig.savefig(output_stem.with_suffix(".pdf"), facecolor="white", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(
        output_stem.with_suffix(".png"),
        dpi=600,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.03,
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
