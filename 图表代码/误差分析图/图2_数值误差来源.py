import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from plot_common import CHART_EXPORT_DIR, configure_cjk_rendering

configure_cjk_rendering()

error_sources = [
    r"空间网格误差（$N=21$ 对 $N=81$）",
    "时间步长误差（30 s 对 1 s）",
    "终点记录误差（离散值对插值值）",
]
relative_percent = [0.0659, 0.0441, 0.0105]
source_colors = ["#67C5A8", "#75B6E3", "#EC88A5"]
source_rows = [2, 1, 0]
stroke_gray, grid_gray = "#3B3B3B", "#E5E5E5"

fig, ax = plt.subplots(figsize=(12.8, 6.8), dpi=160)

for row, percent, tone in zip(source_rows, relative_percent, source_colors):
    ax.hlines(row, 0, percent, color=tone, alpha=0.42, linewidth=5, zorder=2)
    ax.scatter(percent, row, s=330, color=tone, edgecolor=stroke_gray,
               linewidth=1.1, zorder=3)
    ax.text(percent + 0.003, row, f"{percent:.4f}%", va="center",
            ha="left", fontsize=16, color="#222222")

ax.axvline(0.1, color="#888888", linestyle="--", linewidth=1.6, zorder=1)
ax.legend(handles=[Line2D([0], [0], color="#888888", linestyle="--",
                          lw=1.6, label="0.1% 参考线")],
          loc="upper left", frameon=False, fontsize=14)

ax.set_yticks(source_rows)
ax.set_yticklabels(error_sources, fontsize=14)
ax.set_xlim(0, 0.112)
ax.set_ylim(-0.65, 2.65)
ax.set_xlabel("相对误差（%）", fontsize=15)
ax.set_xticks([0, 0.02, 0.04, 0.06, 0.08, 0.10])
ax.tick_params(axis="x", labelsize=13)
ax.grid(axis="x", color=grid_gray, linewidth=0.9, zorder=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0)

ax.text(0.11, -0.42, "最终烘干时间：57.2107 h",
        ha="right", va="center", fontsize=14, color="#B94A60",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#F9E3E8",
                  edgecolor="none", alpha=0.9))

fig.subplots_adjust(left=0.35, right=0.97, top=0.95, bottom=0.17)
png_path = CHART_EXPORT_DIR / "图2_数值误差来源.png"
fig.savefig(png_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"已生成：{png_path}")
