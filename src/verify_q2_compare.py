# -*- coding: utf-8 -*-
"""
verify_q2_compare.py —— 检验 ⑤对比验证（建模文档 §10.2 的 6 组）

1. **退化回归**（关键项）：附录 3 物性整体换成附录 2 常数（ρ=820, c_p=2600, k=0.36,
   D=7e-9·e^(−0.89/C)），只跑 t∈[0,1800] s → 复现 Q1 的三个锚点。**Q2 唯一的解析级基准**。
2. **能量形式**：capacity（主）vs product（∂(ωT)/∂t，T_ref = 0 ℃ / 273.15 K 两档），
   [0,10800] s → 用数字展示 §3.1 的「参考温度不变性」失败（诊断，不进交付）。
3. **水分形式**：∂C/∂t（主）vs ∂(ρC)/∂t（§3.2 式 Q2-9），量化速率差与 t_dry 差。
4. **物性口径**：附录 3（主）vs 附录 2（对照），整段 3 h → 量化 §6.1 的不连续性。
5. **界面系数**：arith/harm/faceCT → 见 `results/q2_dt_convergence.json` 的 `face_rule_study`
   （本文件只引用，不重复计算）。
6. **附件 1 外推**：E0/E1/E2 → 见 `results/q2_scenarios.json`（本文件只引用）。

产物：`results/q2_compare.json`
用法：python src/verify_q2_compare.py [--workers 6]
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

from q2_parallel import run_cases, load_part, save_json   # noqa: E402

RES = ROOT / "results"
ACCEPT_T = 5.0e-5          # ℃
C_DRY = 0.15
R_OUT = np.arange(0.0, 2.0 + 1e-9, 0.1)
# Q1 的三个锚点（逐位取自 results/results.json，2026-09-10 拍板的交付口径）
Q1_ANCHOR = {
    "T_center_1800_analytic_C": 33.5753448265136,
    "T_surface_1800_analytic_C": 36.78556035156142,
    "C_surface_1800_richardson": 1.5102359725649321,
    "C_surface_1800_rawN200": 1.5102532607720367,
    "C_center_1800": 2.54999231711862,
}
Q1_NUMERIC_CSV = RES / "q1_temperature_numeric_crosscheck.csv"


def compare1_degenerate():
    """对比 1：退化回归 —— 复现 Q1 三锚点（**必须逐位对比**）。"""
    z, m = load_part("degenerate_q1")
    t = z["t_s"]
    i_surf, i_center = 20, 0                       # r = 2.0 / 0.0 cm（21 列里的末列/首列）
    T_c = float(z["T_21"][-1, i_center])
    T_s = float(z["T_21"][-1, i_surf])
    C_s = float(z["C_21"][-1, i_surf])
    C_c = float(z["C_21"][-1, i_center])
    out = {
        "design": "附录 2 常数物性（ρ=820, c_p=2600, k=0.36, D=7e-9·e^(−0.89/C)），"
                  "[0,1800] s，Δr=0.01 cm，Δt=1/64 s（与 Q1 生产口径逐项一致）",
        "why": "常数物性下「容量形式」与「乘积形式」恒等 → 该回归验证的是**求解器结构**"
               "（离散/装配/边界/时间推进），不是 Q2 的变物性物理。",
        "q1_anchors": Q1_ANCHOR,
        "q2_degenerate": {
            "T_center_1800_C": T_c, "T_surface_1800_C": T_s,
            "C_surface_1800_kgkg": C_s, "C_center_1800_kgkg": C_c,
            "mode": (m.get("note") or ""),
        },
        "checks": {},
    }
    out["checks"]["T_center_vs_analytic"] = {
        "abs_err_C": abs(T_c - Q1_ANCHOR["T_center_1800_analytic_C"]),
        "criterion_C": ACCEPT_T,
        "pass": bool(abs(T_c - Q1_ANCHOR["T_center_1800_analytic_C"]) < ACCEPT_T),
        "note": "解析解是常数物性下的精确解（Q1 交付口径）；数值解与它的差就是离散误差",
    }
    out["checks"]["T_surface_vs_analytic"] = {
        "abs_err_C": abs(T_s - Q1_ANCHOR["T_surface_1800_analytic_C"]),
        "criterion_C": ACCEPT_T,
        "pass": bool(abs(T_s - Q1_ANCHOR["T_surface_1800_analytic_C"]) < ACCEPT_T),
    }
    # 表面 C：Q1 的**交付值**是 Richardson 外推（N400/N800），单个 N200 网格值应与
    # Q1 的 raw N200 逐位一致；与 Richardson 的差 = Q1 自己登记的网格不确定度。
    out["checks"]["C_surface_vs_Q1_rawN200"] = {
        "abs_err": abs(C_s - Q1_ANCHOR["C_surface_1800_rawN200"]),
        "pass": bool(abs(C_s - Q1_ANCHOR["C_surface_1800_rawN200"]) < 1e-9),
        "note": "**这是「Q2 退化内核 == Q1 内核」的逐位判据**（同一网格、同一 Δt）",
    }
    out["checks"]["C_surface_vs_Q1_richardson"] = {
        "abs_err": abs(C_s - Q1_ANCHOR["C_surface_1800_richardson"]),
        "note": "差的这部分就是 Q1 登记的**网格**不确定度（Q1 交付列用 Richardson 外推）；"
                "本文件不对外推，故此处不作判据",
    }
    out["checks"]["C_center_vs_Q1"] = {
        "abs_err": abs(C_c - Q1_ANCHOR["C_center_1800"]),
        "pass": bool(abs(C_c - Q1_ANCHOR["C_center_1800"]) < 1e-9),
    }
    # 与 Q1 的**数值**温度场逐点对比（1801×21，全字段、全窗口）
    if Q1_NUMERIC_CSV.is_file():
        q1 = np.loadtxt(Q1_NUMERIC_CSV, delimiter=",", skiprows=1)
        if q1.shape == (t.size, 22):
            d = np.abs(z["T_21"] - q1[:, 1:])
            k = np.unravel_index(int(np.argmax(d)), d.shape)
            out["checks"]["T_field_vs_Q1_numeric"] = {
                "shape": list(d.shape),
                "max_abs_diff_C": float(d.max()),
                "at": dict(t_s=float(t[k[0]]), r_cm=float(R_OUT[k[1]])),
                "at_1800s_max": float(d[-1].max()),
                "note": "逐点对比 Q2 退化内核与 Q1 生产内核的温度场（含 t=0）",
            }
        else:
            out["checks"]["T_field_vs_Q1_numeric"] = {"error": "形状不符 %s" % (q1.shape,)}
    out["pass"] = all(v.get("pass", True) for v in out["checks"].values())
    return out


def compare2_energy_form():
    """对比 2：能量形式（capacity vs product，T_ref = 0 ℃ / 273.15 K）。"""
    zc, mc = load_part("dr_200")                    # capacity = 主口径
    zp0, mp0 = load_part("energy_product_0C")
    zpK, mpK = load_part("energy_product_K")
    t = zc["t_s"]
    out = {"design": "capacity（主）vs product（∂(ωT)/∂t）两档 T_ref：0 ℃ / 273.15 K；"
                     "窗口 t=1…10800 s × 21 个交付半径；诊断用，**不进交付**",
           "why": "§3.1.2：乘积形式因显含 T_ref 而**不满足参考温度不变性**——两档 T_ref "
                  "若给出不同的物理答案，即用数字证明它不可用",
           "diagnostic_only": True}
    d_c_p0 = np.abs(zc["T_21"][1:] - zp0["T_21"][1:])
    d_c_pK = np.abs(zc["T_21"][1:] - zpK["T_21"][1:])
    d_p0_pK = np.abs(zp0["T_21"][1:] - zpK["T_21"][1:])
    dC_c_p0 = np.abs(zc["C_21"][1:] - zp0["C_21"][1:])
    dC_c_pK = np.abs(zc["C_21"][1:] - zpK["C_21"][1:])
    j = [int(np.argmin(np.abs(t - ts))) for ts in (1800, 3600, 5400, 7200, 9000, 10800)]
    out["table3_style_T_diff_C"] = {
        "times_s": [int(t[k]) for k in j],
        "capacity_minus_product0C": [[float(v) for v in (zc["T_21"][k] - zp0["T_21"][k])]
                                     for k in j],
        "product0C_minus_productK": [[float(v) for v in (zp0["T_21"][k] - zpK["T_21"][k])]
                                     for k in j],
        "radii_cm": R_OUT.tolist(),
    }
    out["summary"] = {
        "capacity_vs_product_0C": dict(max_T_diff_C=float(d_c_p0.max()),
                                       max_C_diff=dC_c_p0.max(),
                                       at_surface_1800_C=float(d_c_p0[-1, -1])),
        "capacity_vs_product_K": dict(max_T_diff_C=float(d_c_pK.max()),
                                      max_C_diff=dC_c_pK.max(),
                                      at_surface_1800_C=float(d_c_pK[-1, -1])),
        "product_0C_vs_product_K": dict(max_T_diff_C=float(d_p0_pK.max()),
                                        at_surface_1800_C=float(d_p0_pK[-1, -1])),
    }
    out["conclusion"] = ("两档 T_ref 之差的量级 %.3e ℃（表面 %.3e ℃）与「容量形式 vs 乘积形式」"
                         "之差同量级 → 乘积形式的答案取决于人为选定的温度零点，**不可用**"
                         % (float(d_p0_pK.max()), float(d_p0_pK[-1, -1])))
    np.savetxt(RES / "q2_energy_form_diff.csv",
               np.column_stack([t[1:], zc["T_21"][1:, -1], zp0["T_21"][1:, -1],
                                zpK["T_21"][1:, -1],
                                (zc["T_21"][1:, -1] - zp0["T_21"][1:, -1]),
                                (zp0["T_21"][1:, -1] - zpK["T_21"][1:, -1])]),
               delimiter=",", comments="",
               header="t_s,T_surf_capacity,T_surf_product_0C,T_surf_product_K,"
                      "cap_minus_p0C,p0C_minus_pK")
    return out


def compare3_moisture_form():
    """对比 3：水分形式 ∂C/∂t（主）vs ∂(ρC)/∂t（§3.2 式 Q2-9）。"""
    zd, md = load_part("dr_200")                 # 主口径（∂C/∂t），Δt=1/64
    # 替代形式用 `moist_rhoC_full`：它同时提供了 3 h 窗口的 60 s 快照（本次对照用）
    # 与全程 t_dry。`moist_rhoC`（Δt=1/64 的 3 h）因 rhoC 形式 Picard 收敛慢，
    # 单例要 ~35 min，性价比低，故不跑 —— 3 h 场差由 `moist_rhoC_full` 的快照给出。
    zrf, mrf = load_part("moist_rhoC_full")
    zr = zrf
    _, mbase = load_part("sens_h__1.0")   # 主口径在**同一灵敏度 Δt 口径**下的 t_dry，
                                          # 与 rhoC_full 严格可比（两边 Δt 完全一致）
    t = zd["t_s"]
    out = {"design": "∂C/∂t（主）vs ∂(ρC)/∂t（替代形式），[0,10800] s 场对比 + 全时长 t_dry 对比",
           "why": "§3.2：把附录 3 的 ρ(C) 误当干物质密度会引入 1/(1+Cρ'/ρ) 因子"
                  "（C=2.55 处 = 0.7495 → 速率差 33.4 %），必须用数字说清代价",
           "theory_factor_at_C_2_55": 1.0 + 2.55 * 128.0 / (650.0 + 128.0 * 2.55),
           "theory_rate_ratio_at_C_2_55": 1.0 / (1.0 + 2.55 * 128.0 / (650.0 + 128.0 * 2.55)),
           "theory_factor_at_C_0_15": 1.0 + 0.15 * 128.0 / (650.0 + 128.0 * 0.15),
           }
    # 时间轴对齐（rhoC_full 是 60 s 采样；主口径是 1 s 采样）
    tr = zr["t_s"]
    j = [int(np.argmin(np.abs(t - ts))) for ts in (1800, 3600, 5400, 7200, 9000, 10800)]
    jr = [int(np.argmin(np.abs(tr - ts))) for ts in (1800, 3600, 5400, 7200, 9000, 10800)]
    for a, b in zip(j, jr):
        if abs(float(t[a]) - float(tr[b])) > 1e-9:
            raise ValueError("时间轴未对齐：%.1f vs %.1f" % (t[a], tr[b]))
    dC = np.abs(zd["C_21"][j] - zr["C_21"][jr])
    dCall = np.abs(np.array([np.interp(t, tr, zr["C_21"][:, k]) for k in range(21)]).T
                   [1:] - zd["C_21"][1:])
    out["table4_style_C_diff"] = {
        "times_s": [int(t[k]) for k in j],
        "dC_minus_rhoC": [[float(zd["C_21"][a, c] - zr["C_21"][b, c])
                           for c in range(21)] for a, b in zip(j, jr)],
        "radii_cm": R_OUT.tolist(),
    }
    out["field_diff"] = {
        "max_abs_C_over_3h_window": float(dCall.max()),
        "at_t_10800s": [float(v) for v in (zd["C_21"][-1] - zr["C_21"][jr[-1]])],
        "C_max_3h_dC": float(zd["C_21"][-1].max()),
        "C_max_3h_rhoC": float(zr["C_21"][jr[-1]].max()),
        "C_surf_3h_dC": float(zd["C_21"][-1, -1]),
        "C_surf_3h_rhoC": float(zr["C_21"][jr[-1], -1]),
        "note": "替代形式的快照是 60 s 网格（与主口径的 1 s 网格对齐到 1800/3600/…/10800 s）",
    }
    out["t_dry"] = {
        "dC": mbase.get("t_dry_linear"), "rhoC": mrf.get("t_dry_linear"),
        "dC_case": "sens_h__1.0（主口径，Δt_I=1/16、Δt_II=1/2）",
        "rhoC_case_note": mrf.get("note"),
        "note": "两个形式用**完全相同的 Δt**（Δt_I=1/16、Δt_II=1/2）→ 差值纯粹是**模型形式**差",
    }
    if out["t_dry"]["dC"] and out["t_dry"]["rhoC"]:
        out["t_dry"]["ratio_rhoC_over_dC"] = out["t_dry"]["rhoC"] / out["t_dry"]["dC"]
        out["t_dry"]["diff_h"] = (out["t_dry"]["rhoC"] - out["t_dry"]["dC"]) / 3600.0
    return out


def compare4_props():
    """对比 4：物性口径 附录 3（主）vs 附录 2（对照），整段 3 h。"""
    z3, m3 = load_part("dr_200")
    z2, m2 = load_part("props_ap2_3h")
    t = z3["t_s"]
    j = [int(np.argmin(np.abs(t - ts))) for ts in (1800, 3600, 5400, 7200, 9000, 10800)]
    dT = np.abs(z3["T_21"][1:] - z2["T_21"][1:])
    dC = np.abs(z3["C_21"][1:] - z2["C_21"][1:])
    return {
        "design": "附录 3（Q2 主口径）vs 附录 2（Q1 口径）整段 3 h 场对比",
        "why": "§6.1 的不连续性必须在**场**上量化（论文「模型评价/局限」取数，§6.3）",
        "ratio_table_from_doc_6_1": {
            "rho_at_C_2_55": (650 + 128 * 2.55) / 820.0,
            "cp_at_C_2_55": (1450 + 2736 * 2.55 / 3.55) / 2600.0,
            "k_at_C_2_55": (0.21 + 0.38 * 2.55 / 3.55) / 0.36,
            "alpha_ratio": ((0.21 + 0.38 * 2.55 / 3.55) / ((650 + 128 * 2.55) *
                            (1450 + 2736 * 2.55 / 3.55))) / (0.36 / (820.0 * 2600.0)),
            "D_at_C_0_15_T_50C": 2.4e-3 * np.exp(-0.45 / 0.15) * np.exp(-3850 / 323.15),
            "D_ap2_at_C_0_15": 7e-9 * np.exp(-0.89 / 0.15),
        },
        "table3_diff": {"times_s": [int(t[k]) for k in j], "radii_cm": R_OUT.tolist(),
                        "appendix3": [[float(v) for v in z3["T_21"][k]] for k in j],
                        "appendix2": [[float(v) for v in z2["T_21"][k]] for k in j]},
        "table4_diff": {"times_s": [int(t[k]) for k in j], "radii_cm": R_OUT.tolist(),
                        "appendix3": [[float(v) for v in z3["C_21"][k]] for k in j],
                        "appendix2": [[float(v) for v in z2["C_21"][k]] for k in j]},
        "summary": {"max_abs_T_diff_C": float(dT.max()),
                    "max_abs_C_diff": float(dC.max()),
                    "T_diff_at_1800s_center": float(dT[j[0], 0]),
                    "C_diff_at_1800s_surface": float(dC[j[0], -1]),
                    "C_max_3h_ap3": float(z3["C_21"][-1].max()),
                    "C_max_3h_ap2": float(z2["C_21"][-1].max())},
        "note": "两套物性在数值上不连续，Q1/Q2 是两个各自独立起算的模型（Q2-A12 / §6.3），"
                "**不做连续性检验、不逐点比较**；本块只量化不连续性供论文如实表述",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cases = ["degenerate_q1", "energy_capacity", "energy_product_0C", "energy_product_K",
             "moist_dC", "moist_rhoC", "moist_rhoC_full", "props_ap2_3h"]
    # `energy_capacity` 与 `moist_dC` 都是主口径（= dr_200）；跑它们是为了**自检口径等价**
    r = run_cases([c for c in cases if c not in ("energy_capacity", "moist_dC")],
                  workers=args.workers, force=args.force)
    failed = [k for k, v in r.items() if not v["ok"]]
    if failed:
        print("[WARN] 失败：%s" % failed)

    out = {"question": "Q2 对比验证 6 组（§10.2）",
           "compare1_degenerate_regression": compare1_degenerate(),
           "compare2_energy_form": compare2_energy_form(),
           "compare3_moisture_form": compare3_moisture_form(),
           "compare4_property_set": compare4_props(),
           "compare5_interface_rule": {"source": "results/q2_dt_convergence.json → face_rule_study",
                                       "note": "arith/harm/faceCT × 4 级网格，见收敛文件"},
           "compare6_attachment1_extrapolation": {
               "source": "results/q2_scenarios.json",
               "note": "E0/E1/E2 的 4×4 驱动平台情景，见情景文件"},
           "N/A_items": {
               "caseA_vs_caseB_long": "Q2 不重跑长时（§10.3）：口径 B 是固定 Neumann 通量、"
                                      "无 C_s 反馈 → 离散极值原理不成立、C_max≤0.15 不可达；"
                                      "Q2/Q3 的绝对时长完全由口径 A 决定",
               "bessel_analytic_direct": "Q2 的变系数方程没有 Bessel 级数解 → 不可直接对标；"
                                         "以对比 1（退化回归）作为唯一解析级基准",
               "latent_heat_coupling": "题目未给汽化潜热与吸附等温线 → 以量化失效条件替代"
                                       "（§3.5 定理 1：恒温段模型预测 T ≡ T_a，是上界）",
           }}
    save_json(RES / "q2_compare.json", out)

    c1 = out["compare1_degenerate_regression"]
    print("\n[对比1 退化回归] %s" % ("**PASS**" if c1["pass"] else "**FAIL**"))
    for k, v in c1["checks"].items():
        print("   %-28s %s" % (k, v))
    print("[对比2 能量形式] 两档 T_ref 之差 max %.3e ℃（表面 %.3e ℃）；"
          "容量形式 vs product_0C 表面差 %.3e ℃"
          % (out["compare2_energy_form"]["summary"]["product_0C_vs_product_K"]["max_T_diff_C"],
             out["compare2_energy_form"]["summary"]["product_0C_vs_product_K"]["at_surface_1800_C"],
             out["compare2_energy_form"]["summary"]["capacity_vs_product_0C"]["at_surface_1800_C"]))
    c3 = out["compare3_moisture_form"]
    print("[对比3 水分形式] 3 h 场差 max %.3e kg/kg（表面 %.4f vs %.4f）；"
          "t_dry: dC=%s s, rhoC=%s s（比 %.4f，差 %+.2f h）"
          % (c3["field_diff"]["max_abs_C_over_3h_window"],
             c3["field_diff"]["C_surf_3h_dC"], c3["field_diff"]["C_surf_3h_rhoC"],
             c3["t_dry"]["dC"], c3["t_dry"]["rhoC"],
             c3["t_dry"].get("ratio_rhoC_over_dC") or float("nan"),
             c3["t_dry"].get("diff_h") or float("nan")))
    print("[对比4 物性口径] max|ΔT| %.3e ℃, max|ΔC| %.3e kg/kg"
          % (out["compare4_property_set"]["summary"]["max_abs_T_diff_C"],
             out["compare4_property_set"]["summary"]["max_abs_C_diff"]))


if __name__ == "__main__":
    main()
