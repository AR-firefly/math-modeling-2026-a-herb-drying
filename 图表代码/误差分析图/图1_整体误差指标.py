import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from plot_common import CHART_EXPORT_DIR, configure_cjk_rendering

configure_cjk_rendering()

temperature_metrics = [0.2594, 0.6358]
moisture_metrics = [0.0189, 0.0986]
relative_metrics = [0.741, 0.827]

bar_blue, bar_pink, bar_mint = "#8CC4E1", "#EE9DB4", "#9ED8C8"
stroke_gray, grid_gray, ref_red = "#333333", "#E3E3E3", "#D7655B"

fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), dpi=160)


def draw_metric_panel(ax, values, labels, colors, ylabel, ylim, caption, decimals=4):
    bars = ax.bar(labels, values, width=0.58, color=colors,
                  edgecolor=stroke_gray, linewidth=0.9, zorder=3)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_ylim(*ylim)
    ax.grid(axis="y", color=grid_gray, linewidth=0.8, zorder=0)
    ax.tick_params(axis="both", labelsize=11)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, value + (ylim[1] - ylim[0]) * 0.025,
                f"{value:.{decimals}f}", ha="center", va="bottom", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(0.5, -0.14, caption, transform=ax.transAxes,
            ha="center", va="top", fontsize=12)


draw_metric_panel(axes[0], temperature_metrics,
                  ["平均绝对误差", "均方根误差"], [bar_blue, bar_pink],
                  "误差（℃）", (0, 0.80), "(a)  温度场")

draw_metric_panel(axes[1], moisture_metrics,
                  ["平均绝对误差", "均方根误差"], [bar_blue, bar_pink],
                  "误差（kg/kg）", (0, 0.12), "(b)  水分浓度场")

bars = axes[2].bar(["温度场", "水分浓度场"], relative_metrics,
                   width=0.58, color=[bar_blue, bar_mint], edgecolor=stroke_gray,
                   linewidth=0.9, zorder=3)
axes[2].axhline(1.0, color=ref_red, linestyle="--", linewidth=1.6, zorder=2)
axes[2].legend(handles=[Line2D([0], [0], color=ref_red, lw=1.6,
                              linestyle="--", label="1%参考线")],
               loc="upper left", frameon=False, fontsize=10)
axes[2].set_ylabel("平均相对误差（%）", fontsize=12)
axes[2].set_ylim(0, 1.18)
axes[2].grid(axis="y", color=grid_gray, linewidth=0.8, zorder=0)
axes[2].tick_params(axis="both", labelsize=11)
for bar, value in zip(bars, relative_metrics):
    axes[2].text(bar.get_x() + bar.get_width()/2, value + 0.035,
                 f"{value:.3f}%", ha="center", va="bottom", fontsize=11)
axes[2].spines["top"].set_visible(False)
axes[2].spines["right"].set_visible(False)
axes[2].text(0.5, -0.14, "(c)  整体相对误差", transform=axes[2].transAxes,
             ha="center", va="top", fontsize=12)

fig.subplots_adjust(left=0.06, right=0.99, top=0.97, bottom=0.20, wspace=0.22)
png_path = CHART_EXPORT_DIR / "图1_整体误差指标.png"
fig.savefig(png_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"已生成：{png_path}")
