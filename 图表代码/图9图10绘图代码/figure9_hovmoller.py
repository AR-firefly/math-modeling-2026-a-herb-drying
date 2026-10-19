from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.colors import LinearSegmentedColormap

MOISTURE_WORKBOOK = Path(__file__).with_name("result3.xlsx")
DRYING_END_H = 57.2107
ISOLINE_LEVELS = [0.5, 1.0, 1.5, 2.0]
TARGET_MOISTURE = 0.15


def read_moisture_grid(workbook):
    wide = pd.read_excel(workbook, header=0)
    radius_cm = np.asarray([float(col) for col in wide.columns[1:]], dtype=float)
    elapsed_s = np.asarray(wide.iloc[:, 0], dtype=float)
    flat = pd.DataFrame({
        "time_h": np.repeat(elapsed_s, radius_cm.size) / 3600.0,
        "r_cm": np.tile(radius_cm, elapsed_s.size),
        "moisture": wide.iloc[:, 1:].to_numpy(dtype=float).reshape(-1),
    })
    return flat.dropna()


def configure_cjk_rendering():
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS"
    ]
    plt.rcParams["axes.unicode_minus"] = False


def build_r_t_mesh(time_h, r_cm):
    radius_axis = np.linspace(float(np.min(r_cm)), float(np.max(r_cm)), 250)
    time_axis = np.linspace(float(np.min(time_h)), max(DRYING_END_H, float(np.max(time_h))), 350)
    return radius_axis, np.meshgrid(radius_axis, time_axis)


def paint_moisture_contours(ax, radius_mesh, time_mesh, concentration):
    floor, ceiling = np.nanmin(concentration), np.nanmax(concentration)
    DEEP_SAGE = LinearSegmentedColormap.from_list(
        "deep_sage",
        ["#eef8ec", "#eef8ec", "#eef8ec", "#71b673", "#c04b86", "#b7437d"]
    )
    filled = ax.contourf(radius_mesh, time_mesh, concentration,
                         levels=np.linspace(floor, ceiling, 25), cmap=DEEP_SAGE, alpha=0.80)

    isolines = [level for level in ISOLINE_LEVELS if floor < level < ceiling]
    if isolines:
        contours = ax.contour(radius_mesh, time_mesh, concentration, levels=isolines,
                              colors="white", linewidths=0.8, alpha=0.9)
        ax.clabel(contours, inline=True, fontsize=9, fmt="%.1f")

    if floor <= TARGET_MOISTURE <= ceiling:
        target = ax.contour(radius_mesh, time_mesh, concentration, levels=[TARGET_MOISTURE],
                            colors="#ff4d4f", linewidths=2.0)
        ax.clabel(target, fmt={TARGET_MOISTURE: "0.15"}, fontsize=9, colors="#ff4d4f")
    return filled


def mark_drying_endpoint(ax, radius_axis):
    span = radius_axis[-1] - radius_axis[0]
    ax.scatter(radius_axis[0], DRYING_END_H, s=18, color="#ff4d4f", zorder=10)
    ax.text(radius_axis[0] + 0.25 * span, DRYING_END_H - 1.2,
            f"{DRYING_END_H:.4f} h", color="#ff4d4f", fontsize=9, ha="left", va="center")


def render_moisture_map(time_h, radius_axis, radius_mesh, time_mesh, concentration):
    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=180)
    filled = paint_moisture_contours(ax, radius_mesh, time_mesh, concentration)
    mark_drying_endpoint(ax, radius_axis)

    ax.set_xlim(radius_axis[0], radius_axis[-1])
    ax.set_ylim(max(DRYING_END_H, float(np.max(time_h))), float(np.min(time_h)))
    ax.set_xlabel(r"到中心距离 $r$ (cm)", fontsize=11)
    ax.set_ylabel(r"时间 $t$ (h)", fontsize=11)
    ax.tick_params(direction="out", labelsize=9)

    colorbar = fig.colorbar(filled, ax=ax, pad=0.05, shrink=0.93)
    colorbar.set_label(r"水分浓度 $C$ (kg/kg)", fontsize=10)
    colorbar.ax.tick_params(labelsize=9)
    fig.tight_layout()
    return fig


def render_figure9():
    if not MOISTURE_WORKBOOK.exists():
        raise FileNotFoundError(
            f"找不到 {MOISTURE_WORKBOOK.name}。请把该 Excel 文件与本脚本放在同一文件夹。"
        )

    configure_cjk_rendering()
    samples = read_moisture_grid(MOISTURE_WORKBOOK)
    time_h = samples["time_h"].to_numpy()
    r_cm = samples["r_cm"].to_numpy()
    concentration_flat = samples["moisture"].to_numpy()

    if len(samples) < 4:
        raise ValueError("有效数据点过少，无法绘制时空分布图。")

    radius_axis, (radius_mesh, time_mesh) = build_r_t_mesh(time_h, r_cm)
    concentration = griddata((r_cm, time_h), concentration_flat,
                             (radius_mesh, time_mesh), method="linear")

    fig = render_moisture_map(time_h, radius_axis, radius_mesh, time_mesh, concentration)
    output_dir = Path(__file__).parent
    fig.savefig(output_dir / "fig_q3_hovmoller.png", dpi=600, bbox_inches="tight")
    fig.savefig(output_dir / "fig_q3_hovmoller.pdf", bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print("图9已生成：fig_q3_hovmoller.png / .pdf")


if __name__ == "__main__":
    render_figure9()
