from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as MplPath


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})
plt.rcParams["font.size"] = 7.0


COLORS = {
    "ink": "#25313D",
    "muted": "#66737F",
    "line": "#596775",
    "panel": "#D5DCE2",
    "neutral": "#F2F4F6",
    "rgb": "#DCEAF7",
    "rgb_edge": "#4E7EA8",
    "dsm": "#F6E6D4",
    "dsm_edge": "#BC7A36",
    "structure": "#DDF1EC",
    "structure_edge": "#3E8C7C",
    "fusion": "#E7E1F1",
    "fusion_edge": "#74649B",
    "output": "#DED8ED",
    "white": "#FFFFFF",
}

# Content-driven canvas: the two detail panels define the width, while the
# overview and detail rows define the height. No journal column width is imposed.
DETAIL_PANEL_WIDTHS_IN = (5.6, 6.4)
OVERVIEW_HEIGHT_IN = 2.55
DETAIL_HEIGHT_IN = 4.45
CANVAS_WIDTH_IN = sum(DETAIL_PANEL_WIDTHS_IN) + 0.35
CANVAS_HEIGHT_IN = OVERVIEW_HEIGHT_IN + DETAIL_HEIGHT_IN + 0.35


def setup_axis(ax: plt.Axes) -> None:
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")


def box(
    ax: plt.Axes,
    center: tuple[float, float],
    size: tuple[float, float],
    text: str,
    *,
    facecolor: str,
    edgecolor: str = COLORS["line"],
    fontsize: float = 6.4,
    linewidth: float = 0.85,
    fontweight: str = "normal",
    radius: float = 0.018,
    zorder: int = 3,
) -> FancyBboxPatch:
    cx, cy = center
    width, height = size
    patch = FancyBboxPatch(
        (cx - width / 2.0, cy - height / 2.0),
        width,
        height,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder,
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
        linespacing=1.13,
        zorder=zorder + 1,
    )
    return patch


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = COLORS["line"],
    linewidth: float = 0.85,
    mutation_scale: float = 7.5,
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


def routed_arrow(
    ax: plt.Axes,
    points: list[tuple[float, float]],
    *,
    color: str = COLORS["line"],
    linewidth: float = 0.8,
    mutation_scale: float = 7.0,
    linestyle: str = "-",
    zorder: int = 1,
) -> FancyArrowPatch:
    path = MplPath(points, [MplPath.MOVETO] + [MplPath.LINETO] * (len(points) - 1))
    patch = FancyArrowPatch(
        path=path,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        linestyle=linestyle,
        color=color,
        shrinkA=0,
        shrinkB=0,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def panel_heading(ax: plt.Axes, label: str, title: str, subtitle: str = "") -> None:
    ax.text(
        0.0,
        1.02,
        label,
        ha="left",
        va="bottom",
        fontsize=9.0,
        fontweight="bold",
        color=COLORS["ink"],
        transform=ax.transAxes,
    )
    ax.text(
        0.045,
        1.02,
        title,
        ha="left",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
        color=COLORS["ink"],
        transform=ax.transAxes,
    )
    if subtitle:
        ax.text(
            0.045,
            0.965,
            subtitle,
            ha="left",
            va="top",
            fontsize=6.4,
            color=COLORS["muted"],
            transform=ax.transAxes,
        )


def draw_overview(ax: plt.Axes) -> None:
    setup_axis(ax)
    panel_heading(
        ax,
        "a",
        "SPMF21 overview",
        "Four scale-specific fusion blocks share one architecture; only their spatial resolution differs.",
    )

    headers = [
        (0.235, "Scale-wise inputs"),
        (0.585, "Structure-conditioned\nfusion"),
        (0.865, "Fused features"),
    ]
    for x, label in headers:
        ax.text(
            x,
            0.90,
            label,
            ha="center",
            va="center",
            fontsize=7.0,
            fontweight="bold",
            color=COLORS["muted"],
            linespacing=1.05,
        )

    row_y = [0.70, 0.53, 0.36, 0.19]
    scales = ["1/4", "1/8", "1/16", "1/32"]
    for index, (y, scale) in enumerate(zip(row_y, scales), start=1):
        ax.text(
            0.015,
            y,
            f"scale {index}\n{scale}",
            ha="left",
            va="center",
            fontsize=6.1,
            color=COLORS["muted"],
            linespacing=1.02,
        )
        box(
            ax,
            (0.235, y + 0.052),
            (0.205, 0.042),
            f"$R_{{{index}}}$  ·  RGB  ·  256 ch",
            facecolor=COLORS["rgb"],
            edgecolor=COLORS["rgb_edge"],
            fontsize=6.0,
            fontweight="bold",
            radius=0.012,
        )
        box(
            ax,
            (0.235, y),
            (0.205, 0.042),
            f"$D_{{{index}}}$  ·  DSM  ·  256 ch",
            facecolor=COLORS["dsm"],
            edgecolor=COLORS["dsm_edge"],
            fontsize=6.0,
            fontweight="bold",
            radius=0.012,
        )
        box(
            ax,
            (0.235, y - 0.052),
            (0.205, 0.042),
            f"$S_{{{index}}}$  ·  structure  ·  256 ch",
            facecolor=COLORS["structure"],
            edgecolor=COLORS["structure_edge"],
            fontsize=5.9,
            fontweight="bold",
            radius=0.012,
        )
        box(
            ax,
            (0.585, y),
            (0.22, 0.105),
            rf"SPMF Fusion$_{{{index}}}$",
            facecolor=COLORS["fusion"],
            edgecolor=COLORS["fusion_edge"],
            fontsize=6.8,
            fontweight="bold" if index == 1 else "normal",
        )
        box(
            ax,
            (0.865, y),
            (0.15, 0.095),
            f"$F_{{{index}}}$\n256 ch",
            facecolor=COLORS["output"],
            edgecolor=COLORS["fusion_edge"],
            fontsize=6.5,
            fontweight="bold",
        )

        for source_y, target_y, color in (
            (y + 0.052, y + 0.03, COLORS["rgb_edge"]),
            (y, y, COLORS["dsm_edge"]),
            (y - 0.052, y - 0.03, COLORS["structure_edge"]),
        ):
            routed_arrow(
                ax,
                [(0.3375, source_y), (0.42, source_y), (0.475, target_y)],
                color=color,
            )
        arrow(ax, (0.695, y), (0.79, y), color=COLORS["fusion_edge"])

    ax.text(
        0.235,
        0.082,
        "$S_i$ from DSMStructureBranch12 (see b)",
        ha="center",
        va="bottom",
        fontsize=5.9,
        color=COLORS["structure_edge"],
        bbox={"facecolor": COLORS["white"], "edgecolor": "none", "pad": 0.8},
    )
    repeated = FancyBboxPatch(
        (0.465, 0.075),
        0.24,
        0.76,
        boxstyle="round,pad=0.008,rounding_size=0.015",
        facecolor="none",
        edgecolor=COLORS["fusion_edge"],
        linewidth=0.75,
        linestyle=(0, (3, 2)),
        zorder=0,
    )
    ax.add_patch(repeated)
    ax.text(
        0.585,
        0.082,
        "shared block expanded in c",
        ha="center",
        va="bottom",
        fontsize=5.9,
        color=COLORS["fusion_edge"],
        bbox={"facecolor": COLORS["white"], "edgecolor": "none", "pad": 0.8},
    )


def draw_structure_branch(ax: plt.Axes) -> None:
    setup_axis(ax)
    panel_heading(
        ax,
        "b",
        "DSMStructureBranch12",
        "DSM geometry and a scale-matched encoder tap jointly form each structure prior.",
    )

    box(
        ax,
        (0.08, 0.75),
        (0.14, 0.12),
        "DSM\n$1\\times H\\times W$",
        facecolor=COLORS["dsm"],
        edgecolor=COLORS["dsm_edge"],
        fontsize=6.3,
        fontweight="bold",
    )
    box(
        ax,
        (0.27, 0.75),
        (0.18, 0.12),
        "Min–max\nnormalization",
        facecolor=COLORS["neutral"],
        fontsize=6.3,
    )
    box(
        ax,
        (0.47, 0.54),
        (0.22, 0.13),
        "Local similarity\n$\\exp[-\\mathrm{Var}_{7\\times7}/(2\\sigma^2)]$",
        facecolor=COLORS["neutral"],
        fontsize=5.9,
    )
    box(
        ax,
        (0.52, 0.75),
        (0.14, 0.12),
        "Concat\n2 channels",
        facecolor=COLORS["neutral"],
        fontsize=6.0,
    )
    box(
        ax,
        (0.76, 0.75),
        (0.27, 0.15),
        "Geometry pyramid\nstem + strided stages\n$G_i$: 64 / 96 / 128 / 160 ch",
        facecolor=COLORS["structure"],
        edgecolor=COLORS["structure_edge"],
        fontsize=6.0,
        fontweight="bold",
    )

    arrow(ax, (0.15, 0.75), (0.18, 0.75), color=COLORS["dsm_edge"])
    arrow(ax, (0.36, 0.75), (0.45, 0.75))
    routed_arrow(ax, [(0.36, 0.72), (0.39, 0.72), (0.39, 0.54), (0.36, 0.54)])
    arrow(ax, (0.47, 0.605), (0.52, 0.69))
    arrow(ax, (0.59, 0.75), (0.625, 0.75), color=COLORS["structure_edge"])

    ax.text(
        0.76,
        0.635,
        "$H_i, W_i = (H,W)/(4,8,16,32)$",
        ha="center",
        va="top",
        fontsize=5.5,
        color=COLORS["muted"],
    )

    box(
        ax,
        (0.08, 0.23),
        (0.14, 0.115),
        "$T_i$\nencoder tap",
        facecolor=COLORS["dsm"],
        edgecolor=COLORS["dsm_edge"],
        fontsize=6.2,
        fontweight="bold",
    )
    box(
        ax,
        (0.285, 0.23),
        (0.20, 0.115),
        "Resize + tap adapter\n$1\\times1$, $3\\times3$ CNA",
        facecolor=COLORS["neutral"],
        fontsize=5.8,
    )
    box(
        ax,
        (0.52, 0.23),
        (0.18, 0.115),
        "Confidence\n$C_i=\\sigma(\\cdot)$, 256 ch",
        facecolor=COLORS["structure"],
        edgecolor=COLORS["structure_edge"],
        fontsize=5.8,
    )
    box(
        ax,
        (0.73, 0.45),
        (0.18, 0.105),
        "$1\\times1$ projection\n$G_i \\rightarrow 256$ ch",
        facecolor=COLORS["neutral"],
        fontsize=5.9,
    )
    box(
        ax,
        (0.84, 0.23),
        (0.29, 0.14),
        "$S_i=P_i(G_i)\\,\\odot$\n$[1+(2C_i-1)]$",
        facecolor=COLORS["structure"],
        edgecolor=COLORS["structure_edge"],
        fontsize=6.3,
        fontweight="bold",
    )

    arrow(ax, (0.15, 0.23), (0.185, 0.23), color=COLORS["dsm_edge"])
    arrow(ax, (0.385, 0.23), (0.43, 0.23), color=COLORS["structure_edge"])
    arrow(ax, (0.61, 0.23), (0.695, 0.23), color=COLORS["structure_edge"])
    routed_arrow(ax, [(0.76, 0.675), (0.76, 0.555), (0.73, 0.555), (0.73, 0.502)])
    arrow(ax, (0.79, 0.415), (0.80, 0.30), color=COLORS["structure_edge"])
    ax.text(
        0.50,
        0.07,
        "CNA: convolution + normalization + GELU   •   tap gradients are detached by default",
        ha="center",
        va="center",
        fontsize=5.6,
        color=COLORS["muted"],
    )


def draw_fusion_block(ax: plt.Axes) -> None:
    setup_axis(ax)
    panel_heading(
        ax,
        "c",
        "Repeated single-scale SPMF fusion block",
        "A shared block computes 64-channel evidence and routes the original modality features.",
    )

    y_rgb, y_structure, y_dsm = 0.78, 0.52, 0.26
    input_specs = [
        (y_rgb, "$R_i$\n256 ch", COLORS["rgb"], COLORS["rgb_edge"]),
        (y_structure, "$S_i$\n256 ch", COLORS["structure"], COLORS["structure_edge"]),
        (y_dsm, "$D_i$\n256 ch", COLORS["dsm"], COLORS["dsm_edge"]),
    ]
    for y, label, fill, edge in input_specs:
        box(
            ax,
            (0.075, y),
            (0.12, 0.105),
            label,
            facecolor=fill,
            edgecolor=edge,
            fontsize=6.2,
            fontweight="bold",
        )

    for y, name, edge in (
        (y_rgb, "$P_R$\n$256\\rightarrow64$", COLORS["rgb_edge"]),
        (y_structure, "$P_S$\n$256\\rightarrow64$", COLORS["structure_edge"]),
        (y_dsm, "$P_D$\n$256\\rightarrow64$", COLORS["dsm_edge"]),
    ):
        box(
            ax,
            (0.235, y),
            (0.13, 0.105),
            name,
            facecolor=COLORS["neutral"],
            edgecolor=edge,
            fontsize=6.0,
        )
        arrow(ax, (0.135, y), (0.17, y), color=edge)

    box(
        ax,
        (0.415, 0.68),
        (0.18, 0.105),
        "RGB affine\n$\\gamma_R,\\beta_R$",
        facecolor=COLORS["structure"],
        edgecolor=COLORS["structure_edge"],
        fontsize=6.0,
    )
    box(
        ax,
        (0.415, 0.36),
        (0.18, 0.105),
        "DSM affine\n$\\gamma_D,\\beta_D$",
        facecolor=COLORS["structure"],
        edgecolor=COLORS["structure_edge"],
        fontsize=6.0,
    )
    routed_arrow(ax, [(0.30, 0.52), (0.315, 0.52), (0.315, 0.68), (0.325, 0.68)], color=COLORS["structure_edge"])
    routed_arrow(ax, [(0.30, 0.52), (0.315, 0.52), (0.315, 0.36), (0.325, 0.36)], color=COLORS["structure_edge"])

    ax.scatter(
        [0.57],
        [y_rgb],
        s=260,
        facecolor=COLORS["white"],
        edgecolor=COLORS["rgb_edge"],
        linewidth=0.9,
        zorder=4,
    )
    ax.scatter(
        [0.57],
        [y_dsm],
        s=260,
        facecolor=COLORS["white"],
        edgecolor=COLORS["dsm_edge"],
        linewidth=0.9,
        zorder=4,
    )
    ax.text(0.57, y_rgb, "A", ha="center", va="center", fontsize=6.5, fontweight="bold", color=COLORS["ink"], zorder=5)
    ax.text(0.57, y_dsm, "A", ha="center", va="center", fontsize=6.5, fontweight="bold", color=COLORS["ink"], zorder=5)
    ax.text(0.57, 0.91, "$z'=z(1+\\gamma)+\\beta$", ha="center", va="center", fontsize=5.8, color=COLORS["muted"])
    arrow(ax, (0.30, y_rgb), (0.538, y_rgb), color=COLORS["rgb_edge"])
    arrow(ax, (0.30, y_dsm), (0.538, y_dsm), color=COLORS["dsm_edge"])
    arrow(ax, (0.505, 0.68), (0.555, 0.748), color=COLORS["structure_edge"])
    arrow(ax, (0.505, 0.36), (0.555, 0.292), color=COLORS["structure_edge"])

    box(
        ax,
        (0.72, y_rgb),
        (0.20, 0.12),
        "Concat $[R'_i,S'_i]$\nEvidence head",
        facecolor=COLORS["rgb"],
        edgecolor=COLORS["rgb_edge"],
        fontsize=5.9,
    )
    box(
        ax,
        (0.72, y_dsm),
        (0.20, 0.12),
        "Concat $[D'_i,S'_i]$\nEvidence head",
        facecolor=COLORS["dsm"],
        edgecolor=COLORS["dsm_edge"],
        fontsize=5.9,
    )
    arrow(ax, (0.602, y_rgb), (0.62, y_rgb), color=COLORS["rgb_edge"])
    arrow(ax, (0.602, y_dsm), (0.62, y_dsm), color=COLORS["dsm_edge"])
    ax.plot([0.30, 0.595], [0.52, 0.52], color=COLORS["structure_edge"], linewidth=0.8, zorder=1)
    ax.text(
        0.455,
        0.535,
        "structure semantic $S'_i$",
        ha="center",
        va="bottom",
        fontsize=5.4,
        color=COLORS["structure_edge"],
    )
    routed_arrow(ax, [(0.595, 0.52), (0.595, 0.69), (0.62, 0.69)], color=COLORS["structure_edge"])
    routed_arrow(ax, [(0.595, 0.52), (0.595, 0.35), (0.62, 0.35)], color=COLORS["structure_edge"])

    box(
        ax,
        (0.90, 0.52),
        (0.15, 0.145),
        "Stack logits\nSoftmax: RGB / DSM\n$W_i^R+W_i^D=1$",
        facecolor=COLORS["fusion"],
        edgecolor=COLORS["fusion_edge"],
        fontsize=5.7,
        fontweight="bold",
    )
    routed_arrow(ax, [(0.82, y_rgb), (0.85, y_rgb), (0.85, 0.57), (0.825, 0.57)], color=COLORS["rgb_edge"])
    routed_arrow(ax, [(0.82, y_dsm), (0.85, y_dsm), (0.85, 0.47), (0.825, 0.47)], color=COLORS["dsm_edge"])

    box(
        ax,
        (0.63, 0.075),
        (0.46, 0.115),
        "$F_i=W_i^R\\odot R_i+W_i^D\\odot D_i$   (256 ch)",
        facecolor=COLORS["output"],
        edgecolor=COLORS["fusion_edge"],
        fontsize=6.4,
        fontweight="bold",
    )
    routed_arrow(ax, [(0.90, 0.447), (0.90, 0.14), (0.82, 0.14)], color=COLORS["fusion_edge"])
    routed_arrow(
        ax,
        [(0.075, 0.725), (0.075, 0.13), (0.40, 0.13)],
        color=COLORS["rgb_edge"],
        linestyle=(0, (3, 2)),
    )
    routed_arrow(
        ax,
        [(0.075, 0.207), (0.075, 0.02), (0.40, 0.02)],
        color=COLORS["dsm_edge"],
        linestyle=(0, (3, 2)),
    )
    ax.text(
        0.25,
        0.095,
        "original-feature skips",
        ha="center",
        va="center",
        fontsize=5.4,
        color=COLORS["muted"],
    )


def build_figure() -> plt.Figure:
    fig = plt.figure(
        figsize=(CANVAS_WIDTH_IN, CANVAS_HEIGHT_IN),
        facecolor=COLORS["white"],
    )
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[OVERVIEW_HEIGHT_IN, DETAIL_HEIGHT_IN],
        width_ratios=DETAIL_PANEL_WIDTHS_IN,
        left=0.03,
        right=0.99,
        bottom=0.035,
        top=0.94,
        hspace=0.20,
        wspace=0.075,
    )
    ax_a = fig.add_subplot(grid[0, :])
    ax_b = fig.add_subplot(grid[1, 0])
    ax_c = fig.add_subplot(grid[1, 1])

    fig.suptitle(
        "SPMF21: multi-scale structure-conditioned multimodal fusion",
        x=0.515,
        y=0.985,
        ha="center",
        va="top",
        fontsize=10.2,
        fontweight="bold",
        color=COLORS["ink"],
    )
    draw_overview(ax_a)
    draw_structure_branch(ax_b)
    draw_fusion_block(ax_c)

    divider_y = 0.5 * (ax_a.get_position().y0 + max(ax_b.get_position().y1, ax_c.get_position().y1))
    divider = plt.Line2D(
        [0.03, 0.99],
        [divider_y, divider_y],
        transform=fig.transFigure,
        color=COLORS["panel"],
        linewidth=0.7,
        zorder=0,
    )
    fig.add_artist(divider)
    return fig


def save_figure(fig: plt.Figure, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / "spmf21_structure"
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(
        base.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.03,
        pil_kwargs={"compression": "tiff_lzw"},
    )


def main() -> None:
    output_dir = Path(__file__).resolve().parent
    fig = build_figure()
    save_figure(fig, output_dir)
    plt.close(fig)


if __name__ == "__main__":
    main()
