import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 10

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

FLUX_TABLE = Path(__file__).with_name("图5_表面通量数据.csv")
table = pd.read_csv(FLUX_TABLE, encoding="utf-8-sig")


def locate_column(columns, keywords):
    for key in keywords:
        if key in columns:
            return key
    for col in columns:
        s = str(col).lower().replace(" ", "")
        for key in keywords:
            if str(key).lower().replace(" ", "") in s:
                return col
    return None


elapsed_col = locate_column(table.columns, ["t", "time", "时间", "时间/s"])
radius_col = locate_column(table.columns, ["r", "radius", "半径", "径向位置"])
heat_col = locate_column(table.columns, ["q", "heat_flux", "heatflux", "热通量", "表面热通量"])
mass_col = locate_column(table.columns, ["j", "mass_flux", "massflux", "传质通量", "表面传质通量"])

if elapsed_col is None or heat_col is None or mass_col is None:
    raise ValueError(
        f"未识别到时间、q、j 列。当前列名：{list(table.columns)}\n"
        "若 Excel 仅含 T、C，请先按边界条件计算表面热通量 q 和传质通量 j。"
    )

if radius_col is not None:
    outermost = table.groupby(elapsed_col)[radius_col].idxmax()
    surface_rows = table.loc[outermost, [elapsed_col, heat_col, mass_col]].sort_values(elapsed_col).copy()
else:
    surface_rows = table[[elapsed_col, heat_col, mass_col]].drop_duplicates(elapsed_col).sort_values(elapsed_col).copy()

elapsed_s = surface_rows[elapsed_col].to_numpy(dtype=float)
heat_flux = surface_rows[heat_col].to_numpy(dtype=float)
mass_flux = surface_rows[mass_col].to_numpy(dtype=float)

in_window = (elapsed_s >= 0) & (elapsed_s <= 1800)
elapsed_s, heat_flux, mass_flux = elapsed_s[in_window], heat_flux[in_window], mass_flux[in_window]
mass_micro = mass_flux * 1e6

total_heat = integrate(heat_flux, elapsed_s)
total_mass = integrate(mass_flux, elapsed_s)
print(f"累计表面换热量 Q = {total_heat:.4f} J/m²")
print(f"累计表面传质量 M = {total_mass:.8f} kg/m²")

fig, axes = plt.subplots(2, 1, figsize=(8.0, 7.0), sharex=True)

ax = axes[0]
ax.plot(elapsed_s, heat_flux, marker="o", markersize=3.2, linewidth=1.7)
ax.fill_between(elapsed_s, 0, heat_flux, alpha=0.13)
ax.set_ylabel(r"表面热通量 $q$/$(\mathrm{W\,m^{-2}})$")
ax.set_title("表面热通量（预热阶段 0–1800 s）", loc="left", fontsize=10)
ax.text(0.01, 0.97, "(a)", transform=ax.transAxes, va="top", fontweight="bold")
ax.text(0.20, 0.25, "阴影面积：累计传入热量\n"+r"$Q=\int_0^t q\,\mathrm{d}t$",
        transform=ax.transAxes, fontsize=9)
ax.annotate(rf"$t={elapsed_s[-1]:.0f}\ \mathrm{{s}}:\ q={heat_flux[-1]:.2f}\ \mathrm{{W/m^2}}$",
            xy=(elapsed_s[-1], heat_flux[-1]), xytext=(0.70, 0.75), textcoords="axes fraction",
            fontsize=8.5, arrowprops=dict(arrowstyle="-", linewidth=0.7))
ax.grid(alpha=0.16)

ax = axes[1]
ax.plot(elapsed_s, mass_micro, marker="o", markersize=3.2, linewidth=1.7)
ax.fill_between(elapsed_s, mass_micro, np.min(mass_micro), alpha=0.13)
ax.set_xlabel("时间 $t$/s")
ax.set_ylabel(r"表面传质通量 $j$/$(10^{-6}\,\mathrm{kg\,m^{-2}\,s^{-1}})$")
ax.set_title("表面传质通量（预热阶段 0–1800 s）", loc="left", fontsize=10)
ax.text(0.01, 0.97, "(b)", transform=ax.transAxes, va="top", fontweight="bold")
ax.text(0.05, 0.25, "阴影面积：累计传出质量\n"+r"$M=\int_0^t j\,\mathrm{d}t$",
        transform=ax.transAxes, fontsize=9)
ax.annotate(rf"$t={elapsed_s[-1]:.0f}\ \mathrm{{s}}:\ j={mass_micro[-1]:.2f}\times10^{{-6}}$",
            xy=(elapsed_s[-1], mass_micro[-1]), xytext=(0.70, 0.30), textcoords="axes fraction",
            fontsize=8.5, arrowprops=dict(arrowstyle="-", linewidth=0.7))
ax.grid(alpha=0.16)
ax.set_xlim(0, 1800)

plt.tight_layout()
plt.savefig(Path(__file__).with_name("图5_表面热通量与传质通量演化.png"),
            dpi=600, bbox_inches="tight", facecolor="white")
plt.show()
