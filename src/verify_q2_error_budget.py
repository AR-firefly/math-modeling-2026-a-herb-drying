# -*- coding: utf-8 -*-
"""
verify_q2_error_budget.py —— 检验 ④误差分析（建模文档 §10 第 4 类）

**主误差源必须实测，不得照搬 Q1**（Q1 的结论是「温度→时间步、水分表面列→空间」，
Q2 的 D、α、C 变幅全变，两场结论**可能相反**）。本脚本把五类误差的**实测**数值汇总，
逐项给出「离散 / 模型 / 数据 / 参数」的归因与量级，并明确：

  · **主误差源**：T 与 C **分开**判定（用实测的误差量级对比，不用理论预期）
  · **表面列不确定度单列**（Q1 A7 的强制项），随结果报告
  · N/A 项写明理由（§10.3）

输入（全部为已冻结的检验产物，本脚本**不重新求解**）：
  results/q2_dt_convergence.json、results/q2_convergence.json、
  results/q2_compare.json、results/q2_sensitivity.json、results/q2_scenarios.json、
  results/q2_parts/interp_step.json + .npz

产物：results/q2_error_budget.json
用法：python src/verify_q2_error_budget.py [--run-interp]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if getattr(sys.stdout, "reconfigure", None):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from q2_parallel import run_cases, load_part, save_json   # noqa: E402

RES = ROOT / "results"
ACCEPT = 5.0e-5
R_OUT = np.arange(0.0, 2.0 + 1e-9, 0.1)


def _read(name):
    p = RES / name
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def interp_error():
    """数据误差：附件 1 的读取方式（分段线性 vs 零阶保持）。"""
    try:
        zl, _ = load_part("dr_200")
        zs, _ = load_part("interp_step")
    except Exception as e:                                     # noqa: BLE001
        return {"status": "missing", "error": str(e)}
    if zl["t_s"].shape != zs["t_s"].shape:
        return {"status": "shape_mismatch"}
    dT = np.abs(zl["T_21"] - zs["T_21"])
    dC = np.abs(zl["C_21"] - zs["C_21"])
    return {
        "design": "附件 1 分段线性插值（主）vs 零阶保持（左端采样值）",
        "max_T_diff_C": float(dT.max()),
        "max_C_diff_kgkg": float(dC.max()),
        "T_surface_col_max": float(dT[:, -1].max()),
        "C_surface_col_max": float(dC[:, -1].max()),
        "note": "零阶保持引入最长 60 s 的**系统性滞后**（不是随机误差），"
                "方向：驱动量偏低 → 温度偏低、干燥偏慢。主口径取分段线性（与 Q1 §6.1 一致）。",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-interp", action="store_true",
                    help="先跑 interp_step 对照算例（约 4 min）")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    if args.run_interp:
        run_cases(["interp_step"], workers=args.workers)

    dtc = _read("q2_dt_convergence.json")
    conv = _read("q2_convergence.json")
    cmp_ = _read("q2_compare.json")
    sens = _read("q2_sensitivity.json")
    scen = _read("q2_scenarios.json")
    if not (dtc and conv):
        raise SystemExit("缺 q2_dt_convergence.json / q2_convergence.json —— "
                         "先跑 src/verify_q2_convergence.py")

    # ---- 离散误差（实测）----
    dtI = dtc["reference"]["levels"]
    dtII = dtc["dt_II_study"]["reference"]
    grid = conv["uncertainty"]
    r = conv["uncertainty"]["richardson_p=1"]
    disc = {
        "time_step_dt_I": {
            "source": "results/q2_dt_convergence.json → reference.levels（以 1/128 + Richardson "
                      "p=1 为基准，全窗口 t=1…10800 s × 21 半径逐点）",
            "delivered_dt_s": 1.0 / 64,
            "T_max_C": dtI["64"]["T_max"], "T_at": dtI["64"]["T_at"],
            "C_max_kgkg": dtI["64"]["C_max"], "C_at": dtI["64"]["C_at"],
            "T_surface_col_max": dtI["64"]["T_surface_col_max"],
            "C_surface_col_max": dtI["64"]["C_surface_col_max"],
            "meets_acceptance": dtI["64"]["meets_acceptance"],
            "orders": dtc["orders"],
        },
        "time_step_dt_II": {
            "source": "results/q2_dt_convergence.json → dt_II_study（窗口 t=14400…28800 s）",
            "delivered_dt_s": 1.0 / 8,
            "T_max_C": dtII["8"]["T_max"], "C_max_kgkg": dtII["8"]["C_max"],
            "cmax_traj_max_abs_diff": dtII["8"]["cmax_traj_max_abs_diff_vs_dt2_8"],
            "meets_acceptance": dtII["8"]["meets_acceptance"],
            "orders": dtc["dt_II_study"]["orders"],
        },
        "space_grid": {
            "source": "results/q2_convergence.json（Δr 四级，全窗口逐点）",
            "delivered_dr_cm": 0.01,
            "T_max_C_conservative": grid["p=1"]["T_C_full_window_max"],
            "C_max_kgkg_conservative": grid["p=1"]["C_kgkg_full_window_max"],
            "T_surface_col_max": grid["p=1"]["T_surface_col_max"],
            "C_surface_col_max": grid["p=1"]["C_surface_col_max"],
            "richardson_T_max_C": r["T_full_window_max"],
            "richardson_C_max_kgkg": r["C_full_window_max"],
            "richardson_T_surface_col_max": r["T_surface_col_max"],
            "richardson_C_surface_col_max": r["C_surface_col_max"],
            "orders": conv["orders"],
            "early_thin_layer": dtc["early_thin_layer_t_le_2s"],
        },
    }

    # ---- 模型误差（实测 + 文献方向）----
    cmp1 = cmp_["compare1_degenerate_regression"]
    cmp2 = cmp_["compare2_energy_form"]
    cmp3 = cmp_["compare3_moisture_form"]
    cmp4 = cmp_["compare4_property_set"]
    model = {
        "energy_form_capacity_vs_product": {
            "source": "q2_compare.json → compare2_energy_form（诊断，不进交付）",
            "capacity_vs_product_0C_surface_1800_C":
                cmp2["summary"]["capacity_vs_product_0C"]["at_surface_1800_C"],
            "product_ref_0C_vs_273K_surface_1800_C":
                cmp2["summary"]["product_0C_vs_product_K"]["at_surface_1800_C"],
            "statement": "乘积形式的两档 T_ref 之差的量级与「容量形式 vs 乘积形式」之差同量级 → "
                         "**该形式不满足参考温度不变性，不可用**（§3.1.2）",
        },
        "moisture_form_dC_vs_rhoC": {
            "source": "q2_compare.json → compare3_moisture_form",
            "theory_rate_ratio_at_C_2_55": cmp3["theory_rate_ratio_at_C_2_55"],
            "t_dry_ratio_rhoC_over_dC": cmp3["t_dry"].get("ratio_rhoC_over_dC"),
            "t_dry_diff_h": cmp3["t_dry"].get("diff_h"),
            "statement": "∂(ρC)/∂t 形式在 C=2.55 处速率差 33.4 %（§3.2），已在场与 t_dry 上量化",
        },
        "property_set_discontinuity": {
            "source": "q2_compare.json → compare4_property_set",
            "max_abs_T_diff_C_3h": cmp4["summary"]["max_abs_T_diff_C"],
            "max_abs_C_diff_3h": cmp4["summary"]["max_abs_C_diff"],
            "statement": "附录 2/3 是两套并列经验公式（§6.1）；Q1 与 Q2 各自独立起算，"
                         "**不做连续性检验**",
        },
        "latent_heat_missing": {
            "source": "建模文档 §3.5 定理 1 / §14.3 第 13 条（不可实测量化，给方向）",
            "direction": "本模型恒温段预测 T ≡ T_a（= 上界）→ D 被 e^(−3850/T) 抬高 → "
                         "**t_dry 偏短**",
            "magnitude_reference": "Q1 登记的潜热/显热比 5.2~8.9",
            "status": "不可实现（题目未给汽化潜热与吸附等温线）→ 以量化失效条件替代",
        },
        "1d_axisymmetric": {
            "source": "Q1 登记的 2D 端面效应（R/L = 0.08）",
            "status": "**未在变物性下重做**（§14.3 第 15 条）——如实登记，不作数值声明",
        },
    }

    # ---- 数据误差 ----
    Ti = interp_error()
    data = {
        "attachment1_interpolation": Ti,
        "attachment1_freeze_extrapolation": {
            "source": "results/q2_scenarios.json（E0/E1/E2 的 4×4 平台情景）",
            "t_dry_E0_s": scen["t_dry_matrix_s"]["freeze_last"]["freeze_last"] if scen else None,
            "t_dry_interval_s": scen["t_dry_interval_s"] if scen else None,
            "t_dry_interval_h": scen["t_dry_interval_h"] if scen else None,
            "statement": "决策 1（冻结末值）的敏感性以**情景区间**表述，不报单一值（§14.3 第 16 条）",
        },
        "attachment1_sampling": {
            "dt_s": 60.0, "n": 241,
            "statement": "附件 1 只有 60 s 采样；段内用分段线性 → 截断误差为 0（线性函数）；"
                         "真实曲线若弯曲则误差 O(Δ²T_a) 不可估（题目未给更高采样）",
        },
    }

    # ---- 参数误差 ----
    par = {
        "source": "results/q2_sensitivity.json（题目未给误差棒 → 扫描代替）",
        "note": "h、h_m 是**推断项**（附录 3 未给，沿用附录 2），必须扫描；"
                "D₀/β/E_D 为附录 3 给定值，扫描用于量化敏感方向与量级",
        "t_dry_baseline_h": sens["baseline_t_dry_h"] if sens else None,
        "scans": {k: {v: (d or {}).get("t_dry_h") for v, d in s["rows"].items()}
                  for k, s in (sens or {}).get("scans", {}).items()},
    }

    # ---- 主误差源判定（实测对比，T 与 C 分开）----
    e_T_time = dtI["64"]["T_max"]
    e_T_space = grid["p=1"]["T_C_full_window_max"]
    e_C_time = dtI["64"]["C_max"]
    e_C_space = grid["p=1"]["C_kgkg_full_window_max"]
    e_T_space_surf = grid["p=1"]["T_surface_col_max"]
    e_C_space_surf = grid["p=1"]["C_surface_col_max"]
    verdict = {
        "T_field": {
            "time_step_max_C": e_T_time, "space_max_C": e_T_space,
            "dominant": "时间步" if e_T_time >= e_T_space else "空间网格",
            "ratio_time_over_space": e_T_time / e_T_space if e_T_space else None,
        },
        "C_field": {
            "time_step_max": e_C_time, "space_max": e_C_space,
            "dominant": "时间步" if e_C_time >= e_C_space else "空间网格",
            "ratio_time_over_space": e_C_time / e_C_space if e_C_space else None,
            "surface_col": {"time_max": dtI["64"]["C_surface_col_max"],
                            "space_max": e_C_space_surf,
                            "note": "表面列单列（Q1 A7 强制项）"},
        },
        "note": "**本判定完全由实测数值给出，未照搬 Q1 的任何结论**（§10 第 4 类的要求）。"
                "比较对象：交付口径（Δt=1/64、Δr=0.01 cm）下的时间离散误差 vs 网格（最细一对）"
                "不确定度。",
    }

    # ---- t_dry 的总不确定度预算（§11.3 的落实；**放大系数必须实测**）----
    import json as _json
    r2p = RES / "results2.json"
    r2 = _json.loads(r2p.read_text(encoding="utf-8")) if r2p.is_file() else {}
    amp_doc = 9.536e5                                      # 建模文档 §7.4 的估计
    slope = abs(r2.get("t_dry", {}).get("dCmax_dt_at_crossing") or float("nan"))
    amp_meas = (1.0 / slope) if slope == slope and slope > 0 else float("nan")
    # Δt_II 对 t_dry 的贡献：**直接比 1/8 与 1/16 的 C_max 轨迹**（不是自比）
    dtII_cmax = None
    try:
        z8, _ = load_part("dtII_8")
        z16, _ = load_part("dtII_16")
        tt = np.linspace(max(z8["cmax_t"][0], z16["cmax_t"][0]),
                         min(z8["cmax_t"][-1], z16["cmax_t"][-1]), 400)
        dtII_cmax = float(np.abs(np.interp(tt, z8["cmax_t"], z8["cmax_v"])
                                 - np.interp(tt, z16["cmax_t"], z16["cmax_v"])).max())
    except Exception:                                             # noqa: BLE001
        dtII_cmax = None
    # 尾部网格不确定度：**实测**（tailgrid_100/400 与同 Δt 的 sens_h__1.0 比 t_dry）
    tail_grid = None
    try:
        tb = _read("q2_parts/sens_h__1.0.json")
        t1 = _read("q2_parts/tailgrid_100.json")
        t4 = _read("q2_parts/tailgrid_400.json")
        if tb and t1 and t4:
            td_b = tb["t_dry_linear"]
            d100 = abs(t1["t_dry_linear"] - td_b)
            d400 = abs(t4["t_dry_linear"] - td_b)
            p_tail = (np.log(d100 / d400) / np.log(2.0)) if d400 > 0 else float("nan")
            tail_grid = dict(
                t_dry_dr_0_02_cm=t1["t_dry_linear"], t_dry_dr_0_01_cm=td_b,
                t_dry_dr_0_005_cm=t4["t_dry_linear"],
                abs_diff_N100N200_s=d100, abs_diff_N200N400_s=d400,
                observed_order=(float(p_tail) if p_tail == p_tail else None),
                conservative_uncertainty_s=d400,
                richardson_p_measured_s=(t4["t_dry_linear"] + d400 / (2.0 ** p_tail - 1.0))
                if (p_tail == p_tail and p_tail > 0) else None,
                note="**真实的尾部网格不确定度**（不是用 t≤2 s 表面列的 6.6e-4 代理）；"
                     "三档网格的 Δt 与灵敏度口径逐项相同（Δt_I=1/16、Δt_II=1/2）",
            )
    except Exception as e:                                        # noqa: BLE001
        tail_grid = {"error": str(e)}
    grid_C_10800 = grid["p=1"]["C_kgkg_full_window_max"]
    tdu = {
        "t_dry_s": r2.get("t_dry", {}).get("value_s"),
        "dCmax_dt_at_crossing_meas": -slope,
        "amplification_measured_s_per_kgkg": amp_meas,
        "amplification_doc_estimate": amp_doc,
        "amplification_ratio_meas_over_doc": (amp_meas / amp_doc) if amp_meas == amp_meas else None,
        "why_not_doc_value": (
            "文档 §7.4 用 LDF 首特征值 × D(0.15) 估 9.536e5 s/(kg/kg)；实测跨越点的 "
            "|dC_max/dt| = %.3e kg/(kg·s) → 放大系数 %.3e s/(kg/kg)，**比文档估计大 %.2f 倍**。"
            "原因：跨越点上 C_max = 0.15 但 C 的体均值更低，D 应按更低含水率取值（更小），"
            "且 D 沿演化持续下降 → 尾部实际比 LDF 估计更慢。**故 t_dry 的不确定度按实测放大"
            "系数报，不套用文档的 9.536e5**"
            % (slope, amp_meas, (amp_meas / amp_doc) if amp_meas == amp_meas else float("nan"))),
        "contributions": {
            "localization": {"abs_s": r2.get("t_dry", {}).get("localization_error_s"),
                             "note": "线性 vs 三次 Hermite 之差；Δt_II=1/8 的括住区间仅 0.125 s"},
            "time_step_dt_II": {"dC_max_window_max": dtII_cmax,
                                "implied_t_s": (dtII_cmax * amp_meas)
                                if (dtII_cmax is not None and amp_meas == amp_meas) else None,
                                "note": "Δt_II 1/8 vs 1/16 的 C_max 轨迹最大偏差 × 实测放大系数"},
            "space_grid_tail_measured": tail_grid,
            "space_grid_early_surface_proxy": {
                "dC_max_proxy_at_10800s": grid_C_10800,
                "implied_t_s": grid_C_10800 * amp_meas if amp_meas == amp_meas else None,
                "note": "若误用 t≤2 s 表面列的网格不确定度（6.56e-4）作代理会得到 ~2.2e3 s —— "
                        "**那是错的**（尾部场光滑，网格误差小两个量级）。以 "
                        "space_grid_tail_measured 为准。"},
            "driver_scenario": {"interval_h": scen["t_dry_interval_h"] if scen else None,
                                "note": "E0/E1/E2 的 4×4 平台情景（§14.3 第 16 条）"},
        },
        "statement": ("t_dry 的**数值**不确定度（局部化 + Δt + 网格）在 10¹ s 量级；"
                      "**模型/数据**的不确定度（驱动平台情景）达 10⁰–10¹ h 量级 —— "
                      "必须以「未舍入值 + 不确定度 + 情景区间」三者同时报告，"
                      "**不得声称 4 位小数有效**"),
    }

    out = {
        "question": "Q2 误差分析（§10 第 4 类）",
        "acceptance_line": ACCEPT,
        "t_dry_uncertainty": tdu,
        "discrete_errors": disc,
        "model_errors": model,
        "data_errors": data,
        "parameter_errors": par,
        "dominant_source_verdict": verdict,
        "surface_column_uncertainty": {
            "note": "**交付值的不确定度**必须随结果报告（Q1 A10.1 的同族要求）",
            "grid_conservative_C_kgkg": e_C_space_surf,
            "grid_richardson_C_kgkg": r["C_surface_col_max"],
            "time_step_at_1_64_C_kgkg": dtI["64"]["C_surface_col_max"],
            "per_time_series": "results/q2_dtI_surface_column_error.csv",
            "statement": "表面列（r = 2.0 cm）的网格不确定度与时间步残差**单列**；"
                         "超出 ① 层分辨率（1e-4）时不得声称该列四位有效",
        },
        "N_A_items": {
            "caseA_vs_caseB_long_window": "Q2 不重跑长时（§10.3）：口径 B 为固定 Neumann 通量、"
                                          "无 C_s 反馈 → 离散极值原理不成立、C_max ≤ 0.15 不可达",
            "bessel_analytic_direct": "变系数方程无 Bessel 级数解 → 由「对比 1 退化回归」替代",
            "latent_heat_coupling": "题目未给潜热与等温线 → 以量化失效条件替代（§3.5 定理 1）",
            "q4_shrinking_domain": "属 Q4 范围",
            "2d_axisymmetric_redo": "未在变物性下重做（§14.3 第 15 条）",
            "parameter_fitting": "题目未给误差棒 → 不做任何参数拟合（§14.4 第 21 条）",
        },
    }
    save_json(RES / "q2_error_budget.json", out)

    print("\n[误差] T 场：时间步 %.3e ℃ vs 空间 %.3e ℃ → 主控 = %s"
          % (e_T_time, e_T_space, verdict["T_field"]["dominant"]))
    print("[误差] C 场：时间步 %.3e vs 空间 %.3e → 主控 = %s；表面列(空间) %.3e, (时间) %.3e"
          % (e_C_time, e_C_space, verdict["C_field"]["dominant"],
             e_C_space_surf, dtI["64"]["C_surface_col_max"]))
    print("[误差] 数据项（插值方式）：max|ΔT| %.3e ℃，max|ΔC| %.3e"
          % (Ti.get("max_T_diff_C", float("nan")), Ti.get("max_C_diff_kgkg", float("nan"))))
    print("[误差] t_dry 放大系数：实测 %.4e s/(kg/kg)，文档估 %.4e → 实测/文档 = %.2f 倍"
          % (tdu["amplification_measured_s_per_kgkg"], tdu["amplification_doc_estimate"],
             tdu["amplification_ratio_meas_over_doc"]))
    c = tdu["contributions"]
    tg = c.get("space_grid_tail_measured") or {}
    print("       Δt_II 贡献 %s s；尾部网格贡献(实测) %s s；导向情景区间 %s h"
          % (c["time_step_dt_II"]["implied_t_s"], tg.get("conservative_uncertainty_s"),
             c["driver_scenario"]["interval_h"]))


if __name__ == "__main__":
    main()
