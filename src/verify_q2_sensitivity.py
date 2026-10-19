# -*- coding: utf-8 -*-
"""
verify_q2_sensitivity.py —— 检验 ③灵敏度（建模文档 §10.1 的完整清单）

扫描清单（单参数 + 关键二维组合）：
    h ×{0.5,1,2} · h_m ×{0.5,1,2} · D₀ ×{0.5,1,2} · β∈{0.40,0.45,0.50}
    E_D∈{3657.5,3850,4042.5} · ρ(C) 整体 ×{0.9,1.0,1.1} · c_p(C) 整体 ×{0.9,1.0,1.1}
    组合 h_m×D₀（3×3）· 驱动平台 T_a/E0–E2 × C_a/E0–E2（4×4，含对比 6）

**Δt 口径（必须声明）**：灵敏度算例用 Δt_I=1/16 s、Δt_II=1/2 s（比生产粗 4 倍）。
理由：灵敏度要的是**方向与量级**（t_dry 的位移是 10²–10⁴ s 量级），不是 4 位小数的交付精度；
生产的 Δt 由 ②数值收敛单独定案（`results/q2_dt_convergence.json`）。**该口径不用于交付值。**

基线（所有参数取题目值）只跑一次（`sens_h__1.0`），其余扫描共用该基线。

产物：`results/q2_sensitivity.json`（单参数 + 组合）、`results/q2_scenarios.json`（驱动情景）
用法：python src/verify_q2_sensitivity.py [--workers 8]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

if getattr(sys.stdout, "reconfigure", None):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from q2_parallel import run_cases, load_part, save_json, PARTS   # noqa: E402

RES = ROOT / "results"

SPECS = [
    ("h", "H = 25×{v} W/(m²·K)", ["0.5", "1.0", "2.0"],
     "h ↑ → 表面对流换热 ↑ → 升温更快、表面温度更接近 T_a；对水分无直接影响（只经 T 进 D）"),
    ("hm", "h_m = 8e-7×{v} m/s", ["0.5", "1.0", "2.0"],
     "h_m ↑ → 表面传质系数 ↑ → 干燥更快、t_dry 更短；同时抬高 C 的下界速度（§4.4）"),
    ("D0", "D₀ = 2.4e-3×{v} m²/s", ["0.5", "1.0", "2.0"],
     "D₀ ↑ → 内扩散 ↑ → 整体干燥更快；对高含水区影响有限（那里 D 已大），主要压尾部"),
    ("beta", "β = {v}（e^(−β/C)）", ["0.40", "0.45", "0.50"],
     "β ↑ → 低含水区 D 指数式↓ → 尾部显著拉长（§4.2 的「尾部主控」）"),
    ("ED", "E_D = {v} K（e^(−E_D/T)）", ["3657.5", "3850", "4042.5"],
     "E_D ↑ → D 对 T 更敏感；恒温段 T≈T_a=50.165 ℃ 恒定，故只经温度场的早段间接影响 t_dry"),
    ("omega", "ρ(C) 整体 ×{v}", ["0.9", "1.0", "1.1"],
     "ρ ↑ → 热惯性 ↑（α↓）→ 预热段升温更慢、表面温度更低 → 早段 D 略小 → t_dry 略增"),
    ("cp", "c_p(C) 整体 ×{v}", ["0.9", "1.0", "1.1"],
     "同 ρ：c_p 只经 ω 进方程；方向与 ρ 相同"),
    ("hmD0", "h_m×{v} 的组合", ["0.5x0.5", "0.5x2", "2x0.5", "2x2"],
     "表面速率由 h_m 与 D 的比值控制（§7.3）：h_m×D₀ 同向加倍会叠加，反向则部分抵消"),
]

SCEN_TA = ["freeze_last", "tail_mean", "row_min", "row_max"]
SCEN_CA = ["freeze_last", "tail_mean", "row_min", "row_max"]
SCEN_LABEL = {
    "freeze_last": "E0 冻结末值（主口径）",
    "tail_mean": "E1 末段 1 h 均值",
    "row_min": "E2 末段下界",
    "row_max": "E2 末段上界",
}


def _row(meta):
    return dict(
        C_max_3h=meta.get("C_max_3h"), C_surf_3h=meta.get("C_surf_3h"),
        C_center_3h=meta.get("C_at_3h_kgkg", [None])[0],
        T_center_3h=meta.get("T_center_3h"), T_surf_3h=meta.get("T_surf_3h"),
        t_dry_s=meta.get("t_dry_linear"), t_dry_h=(meta.get("t_dry_linear") or np.nan) / 3600.0,
        mass_ratio=meta.get("mass_ratio"), wall_s=meta.get("wall_s"),
        floor_count=meta.get("floor_count"), n_violations=meta.get("n_violations"),
        picard_mean=(meta.get("picard") or {}).get("mean_iters"),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cases = ["sens_h__1.0"]                      # 基线（其余「题目值」档共用，不重跑）
    for name, _, vals, _ in SPECS:
        for v in vals:
            if name == "hmD0":
                cases.append("sens_%s__%s" % (name, v))
            elif v == "1.0":
                continue                          # 题目值 → 用基线
            elif name == "beta" and abs(float(v) - 0.45) < 1e-12:
                continue                          # β 的题目值 → 用基线
            elif name == "ED" and abs(float(v) - 3850.0) < 1e-9:
                continue                          # E_D 的题目值 → 用基线
            else:
                cases.append("sens_%s__%s" % (name, v))
    cases = sorted(set(cases))
    scens = ["scen_%s__%s" % (a, b) for a in SCEN_TA for b in SCEN_CA]
    scens = [s for s in scens if s != "scen_freeze_last__freeze_last"]
    allc = cases + scens
    print("[sens] 共 %d 个算例（单参数 %d + 情景 %d）" % (len(allc), len(cases), len(scens)))
    r = run_cases(allc, workers=args.workers, force=args.force)
    failed = [k for k, v in r.items() if not v["ok"]]
    if failed:
        print("[WARN] 失败：%s" % failed)

    base_z, base_m = load_part("sens_h__1.0")
    base = _row(base_m)
    out = {
        "design": "Δt_I=1/16 s、Δt_II=1/2 s（**灵敏度口径，粗于生产，不用于交付值**）；"
                  "Δr=0.01 cm；求解到 C_max ≤ 0.15 + 3 h 余量",
        "baseline": dict(case="sens_h__1.0", note="全部参数取题目值（附录 3 + 附录 2 的 h/h_m）",
                         **base),
        "baseline_t_dry_h": base["t_dry_h"],
        "scans": {},
        "acceptance": None,
    }
    for name, label, vals, direction in SPECS:
        rows = {}
        if name in ("h", "hm", "D0", "omega", "cp"):
            rows["1.0"] = dict(case="sens_h__1.0（基线复用）", **base)
        elif name == "beta":
            rows["0.45"] = dict(case="sens_h__1.0（基线复用）", **base)
        elif name == "ED":
            rows["3850"] = dict(case="sens_h__1.0（基线复用）", **base)
        for v in vals:
            if v in rows:
                continue
            cid = "sens_%s__%s" % (name, v)
            _, m = load_part(cid)
            rows[v] = dict(case=cid, **_row(m))
        # 相对基线的变化（论文要「方向 + 量级」）
        for v, d in rows.items():
            if d.get("t_dry_s") and base.get("t_dry_s"):
                d["dt_dry_s_vs_base"] = d["t_dry_s"] - base["t_dry_s"]
                d["dt_dry_h_vs_base"] = d["dt_dry_s_vs_base"] / 3600.0
                d["dt_dry_rel_pct"] = 100.0 * d["dt_dry_s_vs_base"] / base["t_dry_s"]
            for k in ("C_max_3h", "C_surf_3h", "T_center_3h"):
                if d.get(k) is not None and base.get(k) is not None:
                    d["d_%s_vs_base" % k] = d[k] - base[k]
        out["scans"][name] = dict(label=label, direction=direction, rows=rows)

    # ---- 驱动情景 4×4 ----
    scen = {"design": "T_a × C_a 的 4×4 平台情景（对比 6 + §10.1 的 E0/E1/E2）",
            "labels": SCEN_LABEL, "grid": {}, "t_dry_matrix_h": {}, "t_dry_matrix_s": {},
            "note": "E0=冻结末值（主口径）；E1=末段 1 h 均值；E2=末段上下界。"
                    "**论文必须以区间表述 t_dry**（§14.3 第 16 条）"}
    for a in SCEN_TA:
        scen["t_dry_matrix_h"][a] = {}
        scen["t_dry_matrix_s"][a] = {}
        for b in SCEN_CA:
            cid = "scen_%s__%s" % (a, b)
            if cid == "scen_freeze_last__freeze_last":
                d = dict(case="sens_h__1.0（基线复用）", **base)
            else:
                _, m = load_part(cid)
                d = dict(case=cid, **_row(m))
            scen["grid"]["%s|%s" % (a, b)] = d
            scen["t_dry_matrix_h"][a][b] = d.get("t_dry_h")
            scen["t_dry_matrix_s"][a][b] = d.get("t_dry_s")
    td = np.array([[scen["t_dry_matrix_s"][a][b] for b in SCEN_CA] for a in SCEN_TA], float)
    scen["t_dry_interval_s"] = [float(np.nanmin(td)), float(np.nanmax(td))]
    scen["t_dry_interval_h"] = [float(np.nanmin(td)) / 3600.0, float(np.nanmax(td)) / 3600.0]
    save_json(RES / "q2_sensitivity.json", out)
    save_json(RES / "q2_scenarios.json", scen)

    print("\n[灵敏度] 基线 t_dry = %.4f h（C_max(3h)=%.6f, C_surf(3h)=%.6f）"
          % (base["t_dry_h"], base["C_max_3h"] or np.nan, base["C_surf_3h"] or np.nan))
    for name, label, vals, _ in SPECS:
        line = "  %-6s" % name
        for v, d in out["scans"][name]["rows"].items():
            line += "  %s: t_dry=%s h" % (v, ("%.4f" % d["t_dry_h"]) if d.get("t_dry_h") else "-")
        print(line)
    print("[情景] t_dry 情景区间 = [%.4f, %.4f] h（%.4f, %.4f d）"
          % (scen["t_dry_interval_h"][0], scen["t_dry_interval_h"][1],
             scen["t_dry_interval_h"][0] / 24, scen["t_dry_interval_h"][1] / 24))


if __name__ == "__main__":
    main()
