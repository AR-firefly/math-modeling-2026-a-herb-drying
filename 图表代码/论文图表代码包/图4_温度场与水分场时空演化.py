from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.interpolate import interp1d

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 9

RESULT_WORKBOOK = Path(__file__).with_name("result2_full_60s.xlsx")
SNAPSHOT_SECONDS = [600, 1200, 1800]
GRID_SIDE = 350


def read_wide_sheet(sheet_name, value_name):
    wide = pd.read_excel(RESULT_WORKBOOK, sheet_name=sheet_name, header=0)
    radius_cm = np.asarray([float(col) for col in wide.columns[1:]], dtype=float)
    elapsed_s = np.asarray(wide.iloc[:, 0], dtype=float)
    return pd.DataFrame({
        "t": np.repeat(elapsed_s, radius_cm.size),
        "r": np.tile(radius_cm, elapsed_s.size),
        value_name: wide.iloc[:, 1:].to_numpy(dtype=float).reshape(-1),
    })


table = read_wide_sheet("温度", "T").merge(read_wide_sheet("水分浓度", "C"), on=["t", "r"]).dropna()


def locate_column(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    for col in columns:
        s = str(col).lower().replace(" ", "")
        for c in candidates:
            if str(c).lower().replace(" ", "") in s:
                return col
    return None


elapsed_col = locate_column(table.columns, ["t", "time", "时间", "时间/s", "time(s)"])
radius_col = locate_column(table.columns, ["r", "radius", "半径", "r/r", "径向位置"])
temp_col = locate_column(table.columns, ["T", "temperature", "温度", "温度/℃"])
moist_col = locate_column(table.columns, ["C", "moisture", "含水率", "干基含水率", "水分浓度"])

if None in [elapsed_col, radius_col, temp_col, moist_col]:
    raise ValueError(f"未识别到 t、r、T、C 四列。当前列名：{list(table.columns)}")

snapshots = {}

for target_s in SNAPSHOT_SECONDS:
    recorded_s = np.asarray(table[elapsed_col].unique(), dtype=float)
    nearest_s = recorded_s[np.argmin(np.abs(recorded_s - target_s))]
    rows = table[np.isclose(table[elapsed_col].astype(float), nearest_s)].copy().sort_values(radius_col)
    radius_cm = rows[radius_col].to_numpy(dtype=float)
    temperature = rows[temp_col].to_numpy(dtype=float)
    moisture = rows[moist_col].to_numpy(dtype=float)
    snapshots[target_s] = {"r": radius_cm / np.max(radius_cm), "T": temperature, "C": moisture}

temp_floor = min(s["T"].min() for s in snapshots.values())
temp_ceil = max(s["T"].max() for s in snapshots.values())
moist_floor = min(s["C"].min() for s in snapshots.values())
moist_ceil = max(s["C"].max() for s in snapshots.values())

unit_axis = np.linspace(-1, 1, GRID_SIDE)
XX, YY = np.meshgrid(unit_axis, unit_axis)
radial_position = np.sqrt(XX**2 + YY**2)
inside_disk = radial_position <= 1.0

disks_T, disks_C = {}, {}
for snap_s in SNAPSHOT_SECONDS:
    r, T, C = snapshots[snap_s]["r"], snapshots[snap_s]["T"], snapshots[snap_s]["C"]
    kind = "cubic" if len(r) >= 4 else "linear"
    interp_T = interp1d(r, T, kind=kind, bounds_error=False, fill_value=(T[0], T[-1]))
    interp_C = interp1d(r, C, kind=kind, bounds_error=False, fill_value=(C[0], C[-1]))
    disk_T = interp_T(np.clip(radial_position, 0, 1))
    disk_C = interp_C(np.clip(radial_position, 0, 1))
    disk_T[~inside_disk] = np.nan
    disk_C[~inside_disk] = np.nan
    disks_T[snap_s], disks_C[snap_s] = disk_T, disk_C

fig = plt.figure(figsize=(9.0, 7.0))
gs = fig.add_gridspec(3, 4, width_ratios=[1,1,1,0.07],
                      height_ratios=[1,1,0.95], hspace=0.13, wspace=0.08)

axes_T = []
for i, t in enumerate(SNAPSHOT_SECONDS):
    ax = fig.add_subplot(gs[0, i]); axes_T.append(ax)
    ax.imshow(disks_T[t], origin="lower", extent=(-1,1,-1,1),
              cmap="coolwarm", vmin=temp_floor, vmax=temp_ceil, interpolation="bicubic")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(rf"$t={t}\ \mathrm{{s}}$")
axes_T[0].set_ylabel("温度 $T$/℃")
cax1 = fig.add_subplot(gs[0,3])
cb1 = fig.colorbar(ScalarMappable(norm=Normalize(temp_floor,temp_ceil), cmap="coolwarm"), cax=cax1)
cb1.set_label("温度 $T$/℃")

axes_C = []
for i, t in enumerate(SNAPSHOT_SECONDS):
    ax = fig.add_subplot(gs[1, i]); axes_C.append(ax)
    ax.imshow(disks_C[t], origin="lower", extent=(-1,1,-1,1),
              cmap="YlGnBu", vmin=moist_floor, vmax=moist_ceil, interpolation="bicubic")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
axes_C[0].set_ylabel("干基含水率 $C$ /(kg/kg)")
cax2 = fig.add_subplot(gs[1,3])
cb2 = fig.colorbar(ScalarMappable(norm=Normalize(moist_floor,moist_ceil), cmap="YlGnBu"), cax=cax2)
cb2.set_label("干基含水率 $C$ /(kg/kg)")

axT = fig.add_subplot(gs[2,0:2])
axC = fig.add_subplot(gs[2,2:4])
linestyles = ["-", "--", ":"]
for t, ls in zip(SNAPSHOT_SECONDS, linestyles):
    axT.plot(snapshots[t]["r"], snapshots[t]["T"], ls=ls, lw=1.6, label=f"{t} s")
    axC.plot(snapshots[t]["r"], snapshots[t]["C"], ls=ls, lw=1.6, label=f"{t} s")

axT.set_xlabel("无量纲半径 $r/R$"); axT.set_ylabel("温度 $T$/℃")
axT.set_title("径向温度分布"); axT.legend(frameon=False); axT.grid(alpha=0.18)
axC.set_xlabel("无量纲半径 $r/R$"); axC.set_ylabel("干基含水率 $C$ /(kg/kg)")
axC.set_title("径向水分分布"); axC.legend(frameon=False); axC.grid(alpha=0.18)

plt.savefig(Path(__file__).with_name("图4_温度场与水分场时空演化.png"),
            dpi=600, bbox_inches="tight", facecolor="white")
plt.show()
