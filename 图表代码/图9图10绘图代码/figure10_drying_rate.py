from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


INVENTORY_CSV = Path(__file__).with_name("W_history.csv")
DRYING_END_H = 57.2107


def read_water_inventory(csv_path):
    table = pd.read_csv(csv_path).dropna().sort_values("t_s")
    return table["t_s"].to_numpy(dtype=float) / 3600.0, table["W"].to_numpy(dtype=float)


def configure_cjk_rendering():
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS"
    ]
    plt.rcParams["axes.unicode_minus"] = False


def derive_drying_curves(time_h, water_mass):
    rate = np.maximum(-np.gradient(water_mass, time_h * 3600.0), 0.0)
    removed_fraction = (water_mass[0] - water_mass) / water_mass[0]
    return rate, removed_fraction


def mark_drying_extremes(ax_rate, ax_removed, time_h, peak_time, peak_value, final_removed):
    ax_rate.scatter(peak_time, peak_value, color="#ff5a5f", marker="*", s=80, zorder=10)
    ax_rate.annotate(
        f"{peak_value:.3e}",
        xy=(peak_time, peak_value),
        xytext=(peak_time + 1.2, peak_value * 0.97),
        fontsize=9,
        color="#ff5a5f",
        arrowprops={"arrowstyle": "-", "color": "#ff5a5f", "lw": 0.8},
    )
    ax_rate.axvline(peak_time, linestyle="--", linewidth=0.9, color="gray", alpha=0.8)
    ax_removed.text(
        time_h[-1] - 0.5,
        final_removed - 0.02,
        f"{final_removed * 100:.2f}%",
        fontsize=10,
        color="#3498db",
        ha="right",
        va="top",
    )
    ax_rate.axvline(DRYING_END_H, linestyle=":", linewidth=0.8, color="gray", alpha=0.5)


def style_black_frame(ax_rate, ax_removed):
    ax_rate.set_xlabel(r"时间 $t$ (h)", fontsize=11, color="black")
    ax_rate.set_ylabel(r"干燥速率 $-\mathrm{d}W/\mathrm{d}t$ (kg/s)", fontsize=11, color="black")
    ax_removed.set_ylabel(r"脱水率 $1-W/W_0$", fontsize=11, color="black")
    ax_rate.tick_params(axis="x", colors="black", labelsize=9)
    ax_rate.tick_params(axis="y", colors="black", labelsize=9)
    ax_removed.tick_params(axis="y", colors="black", labelsize=9)
    ax_rate.spines["left"].set_color("black")
    ax_rate.spines["bottom"].set_color("black")
    ax_removed.spines["right"].set_color("black")
    ax_rate.spines["left"].set_linewidth(1.0)
    ax_rate.spines["bottom"].set_linewidth(1.0)
    ax_removed.spines["right"].set_linewidth(1.0)
    ax_rate.spines["top"].set_visible(False)
    ax_removed.spines["top"].set_visible(False)


def render_drying_chart(time_h, rate, removed_fraction, peak_time, peak_value, final_removed):
    fig, ax_rate = plt.subplots(figsize=(8.0, 5.0), dpi=180)
    line_rate, = ax_rate.plot(
        time_h, rate, linewidth=2.0, color="#27ae60", label=r"$-\mathrm{d}W/\mathrm{d}t$",
    )
    ax_rate.fill_between(time_h, 0, rate, color="#f3bfd7", alpha=0.65)

    ax_removed = ax_rate.twinx()
    line_removed, = ax_removed.plot(
        time_h, removed_fraction, linewidth=2.0, color="#3498db", label="脱水率",
    )
    ax_removed.set_ylim(0, 1.05)

    mark_drying_extremes(ax_rate, ax_removed, time_h, peak_time, peak_value, final_removed)
    style_black_frame(ax_rate, ax_removed)

    ax_rate.set_xlim(min(0.0, float(np.min(time_h))), max(DRYING_END_H, float(np.max(time_h))))
    ax_rate.grid(False)
    ax_removed.grid(False)
    lines = [line_rate, line_removed]
    ax_rate.legend(lines, [line.get_label() for line in lines],
                   loc="center right", frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


def render_figure10():
    if not INVENTORY_CSV.exists():
        raise FileNotFoundError(
            f"找不到 {INVENTORY_CSV.name}。请把该文件与本脚本放在同一文件夹。"
        )

    configure_cjk_rendering()
    time_h, water_mass = read_water_inventory(INVENTORY_CSV)

    if len(time_h) < 3 or np.any(np.diff(time_h) <= 0):
        raise ValueError("至少需要3个严格递增的有效时间点。")
    if water_mass[0] <= 0:
        raise ValueError("初始总含水质量必须大于0。")

    rate, removed_fraction = derive_drying_curves(time_h, water_mass)
    peak_index = int(np.argmax(rate))
    peak_time, peak_value = time_h[peak_index], rate[peak_index]
    final_removed = removed_fraction[-1]

    fig = render_drying_chart(time_h, rate, removed_fraction, peak_time, peak_value, final_removed)
    output_dir = Path(__file__).parent
    fig.savefig(output_dir / "fig_q3_drying_rate.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    print("图10已生成：fig_q3_drying_rate.png")
    print(f"峰值干燥速率 = {peak_value:.6e} kg/s")
    print(f"峰值时间 = {peak_time:.6f} h")
    print(f"最终脱水率 = {final_removed * 100:.4f}%")
    print(f"烘干终点 = {DRYING_END_H:.4f} h")


if __name__ == "__main__":
    render_figure10()
