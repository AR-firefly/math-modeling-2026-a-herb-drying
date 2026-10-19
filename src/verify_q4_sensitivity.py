from __future__ import annotations

import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np


def utf8_stdout():
    _SCREEN = sys.stdout
    if getattr(_SCREEN, "reconfigure", None):
        _SCREEN.reconfigure(encoding="utf-8", errors="replace")


utf8_stdout()

_SELF_FILE = Path(__file__).resolve()
_SUBMIT = _SELF_FILE.parents[1]
sys.path.insert(0, "%s/Q4" % _SUBMIT.as_posix())

from solver_q4 import (
    solve_q4, driver_at, radius_at, att2, C_DRY, HM_DEFAULT, H_DEFAULT,
    D0_AP4, BETA_AP4, E_D_AP4, R0, R_MIN, LENGTH_CM, C_INIT, CA_FREEZE,
    T_FREEZE_S, TA_FREEZE_C,
)

RES = _SUBMIT / "results"
N_SENS = 400
DT1, DT2 = 1.25, 5.0
DH_VAP = 2.26e6
RHO_D_BORROW = (760.0 + 90.0 * C_INIT) / (1.0 + C_INIT)


def t_dry_of(**kw):
    o = solve_q4(N=N_SENS, dt1=DT1, dt2=DT2, t_switch=T_FREEZE_S, R_mode="shrink",
                 T_mode="prescribed", props="app4", bc="robin", face_rule="arith",
                 stop_on_dry=True, snap_every_s=0.0, cmax_track=True, audit=True,
                 verbose=False, **kw)
    if o["status"] != "ok":
        return None, o
    tt, cm = o["cmax"]["t"], o["cmax"]["cmax"]
    i = int(np.flatnonzero(cm <= C_DRY)[0])
    if i == 0:
        return float(tt[0]), o
    t0, t1, c0, c1 = tt[i - 1], tt[i], cm[i - 1], cm[i]
    return float(t0 + (C_DRY - c0) / (c1 - c0) * (t1 - t0)), o


def _scan_single_and_2d(sens, base, rows):
    def scan(tag, **kw):
        t0 = time.time()
        td, _ = t_dry_of(**kw)
        rows[tag] = dict(t_dry_s=td, t_dry_h=(td / 3600.0) if td else None,
                         delta_h=((td - base) / 3600.0) if td else None,
                         delta_pct=(100.0 * (td - base) / base) if td else None,
                         wall_s=time.time() - t0)
        print("   %-26s t_dry %s h  Δ %+8.4f h (%+6.2f %%)  %.1f s"
              % (tag, ("%.4f" % (td / 3600.0)) if td else "None",
                 (td - base) / 3600.0 if td else 0.0,
                 100.0 * (td - base) / base if td else 0.0, rows[tag]["wall_s"]),
              flush=True)
    print("[1] 单参数扫描（h、h_m、D₀、β、E_D、ρc_p）")
    for m in (0.5, 2.0):
        scan("h_x%g" % m, H=H_DEFAULT * m)
    for m in (0.5, 2.0):
        scan("hm_x%g" % m, HM=HM_DEFAULT * m)
    for m in (0.5, 2.0):
        scan("D0_x%g" % m, D0=D0_AP4 * m)
    for v in (0.27, 0.33):
        scan("beta_%g" % v, BETA=v)
    for v in (3657.5, 4042.5):
        scan("E_D_%g" % v, E_D=v)
    for m in (0.9, 1.1):
        scan("rho_scale_%g" % m, RHO_SCALE=m)
    for m in (0.9, 1.1):
        scan("cp_scale_%g" % m, CP_SCALE=m)
    sens["single_parameter"] = rows

    print("[2] 二维组合 h_m × h（3×3 = 9 组，§11.4）")
    grid2 = {}
    for mh in (0.5, 1.0, 2.0):
        for mm in (0.5, 1.0, 2.0):
            tag = "h_x%g_hm_x%g" % (mh, mm)
            td, _ = t_dry_of(H=H_DEFAULT * mh, HM=HM_DEFAULT * mm)
            grid2[tag] = dict(t_dry_s=td, t_dry_h=(td / 3600.0) if td else None,
                              delta_pct=(100.0 * (td - base) / base) if td else None)
            print("   %-16s t_dry %8.4f h  Δ %+6.2f %%"
                  % (tag, td / 3600.0, 100.0 * (td - base) / base), flush=True)
    v2 = [v["t_dry_h"] for v in grid2.values() if v["t_dry_h"]]
    sens["h_hm_2d"] = dict(
        grid=grid2, min_h=min(v2), max_h=max(v2),
        spread_h=max(v2) - min(v2),
        spread_pct=100.0 * (max(v2) - min(v2)) / min(v2),
        degeneracy=(" 在预设 T 口径下 `h` 恒无影响（能量方程不求解） 3×3 的 9 组"
                    "退化为 3 个不同值（每列相同）。这是「预设 T 口径」的直接证据，"
                    "必须如实报告：本模型对 `h` 的灵敏度低于真实情况（§11.4）。"),
        direction=("h_m   表面更快  t_dry （实测 hm×0.5 +6.57 %、hm×2 −2.00 %，"
                   "次线性，因 h_m 同时抬高表面浓度梯度的驱动压差）；"
                   "h 在本口径下影响恒为 0（实测 h×0.5 / h×2 的 `delta_h` 逐位为 "
                   "0.0；预设 T 口径下能量方程不求解、`h` 无处进入 —— 属该口径的必然，"
                   "不是「影响被弱化」）。"))
    return sens, rows, grid2


def _scan_scenarios(sens, base):
    print("[3] 驱动外推情景 E0/E1/E2（T_a 与 C_a 平台，§12.2 组 6）")
    scen = {}
    for ts in ("freeze_last", "tail_mean", "row_min", "row_max"):
        for cs in ("freeze_last", "tail_mean", "row_min", "row_max"):
            tag = "Ta_%s__Ca_%s" % (ts, cs)
            if ts == "freeze_last" and cs == "freeze_last":
                continue
            td, _ = t_dry_of(ta_scenario=ts, ca_scenario=cs)
            scen[tag] = dict(t_dry_h=(td / 3600.0) if td else None,
                             delta_pct=(100.0 * (td - base) / base) if td else None)
            print("   %-34s t_dry %s h  Δ %s %%"
                  % (tag, ("%.4f" % (td / 3600.0)) if td else "None",
                     ("%+.2f" % (100.0 * (td - base) / base)) if td else "-"),
                  flush=True)
    v6 = [v["t_dry_h"] for v in scen.values() if v["t_dry_h"]]
    sens["driver_scenarios"] = dict(rows=scen, min_h=min(v6) if v6 else None,
                                    max_h=max(v6) if v6 else None,
                                    spread_h=(max(v6) - min(v6)) if v6 else None,
                                    note="E6 情景区间（§12.2 组 6）；论文以区间表述")
    return sens, scen, v6


def _e1_zero_d(base):
    print("[4] E1 潜热亏缺：0-D 集总对照（借用 ΔH_vap，只进误差分析节）")
    o_base = solve_q4(N=N_SENS, dt1=DT1, dt2=DT2, t_switch=T_FREEZE_S,
                      R_mode="shrink", T_mode="prescribed", props="app4",
                      stop_on_dry=True, snap_every_s=60.0, cmax_track=True,
                      audit=False, verbose=False)
    snap_t, snap_R, snap_C = o_base["snap_t"], o_base["snap_R"], o_base["snap_C"]
    tt, cm = o_base["cmax"]["t"], o_base["cmax"]["cmax"]
    Cs_of_t = np.interp(snap_t, tt, o_base["cmax"]["cR"])
    Cmax_of_t = np.interp(snap_t, tt, cm)
    t_end_e1 = float(o_base["t_end"])
    dtE = 60.0
    tE = np.arange(0.0, t_end_e1 + 1e-9, dtE)
    RE = radius_at(tE, "shrink")
    CsE = np.interp(tE, snap_t, Cs_of_t)
    TaE = np.array([driver_at(t)[0] for t in tE])
    CaE = np.array([driver_at(t)[1] for t in tE])
    Cmid = np.interp(tE, snap_t, Cmax_of_t)
    c_eff = (1850.0 + 2150.0 * Cmid / (Cmid + 1.0))
    Tm = np.empty(tE.size)
    Tm[0] = 28.0
    for k in range(tE.size - 1):
        R_ = RE[k]
        cap = RHO_D_BORROW * R_ * R_ / 2.0 * c_eff[k]
        q_conv = R_ * H_DEFAULT * (TaE[k] - Tm[k])
        q_evap = RHO_D_BORROW * R_ * HM_DEFAULT * (CsE[k] - CaE[k]) * DH_VAP
        Tm[k + 1] = Tm[k] + dtE * (q_conv - q_evap) / cap
    dT = TaE - Tm
    dT_qs = (RHO_D_BORROW * HM_DEFAULT * DH_VAP / H_DEFAULT) * (CsE - CaE)
    P_evap = 2.0 * np.pi * RE * RHO_D_BORROW * HM_DEFAULT * (CsE - CaE) * DH_VAP
    i_dry = int(np.argmin(np.abs(tE - base)))
    P_avg = float(np.trapezoid(P_evap[:i_dry + 1], tE[:i_dry + 1]) / base)
    dT_avg = P_avg / (H_DEFAULT * 2.0 * np.pi * R0)
    t_eff_series = np.column_stack([tE, TaE - dT, TaE - dT_qs])
    return tE, TaE, dT_qs, P_avg, dT_avg


def _e1_adopt(tE, TaE, dT_qs, dT_avg, base):
    def t_dry_with_Teff(T_eff_series, t_end_ov):
        import solver_q4 as S
        t0, T0, C0 = S.att1()
        t_new = np.arange(0.0, T_FREEZE_S + 1e-9, 60.0)
        T_new = np.interp(t_new, tE, T_eff_series)
        C_new = np.interp(t_new, t0, C0)
        shift = float(TaE[0] - T_eff_series[0])
        freeze_orig = S.TA_FREEZE_C
        S._ATT1_CACHE = (t_new, T_new, C_new)
        S.TA_FREEZE_C = freeze_orig - shift
        S._DRIVER_CACHE.clear()
        try:
            oo = solve_q4(N=N_SENS, dt1=DT1, dt2=DT2, t_switch=T_FREEZE_S,
                          R_mode="shrink", T_mode="prescribed", props="app4",
                          stop_on_dry=False, t_end_override=t_end_ov,
                          snap_every_s=0.0, cmax_track=True, audit=False, verbose=False)
        finally:
            S._ATT1_CACHE = (t0, T0, C0)
            S.TA_FREEZE_C = freeze_orig
            S._DRIVER_CACHE.clear()
        cm2 = oo["cmax"]["cmax"]
        ii = np.flatnonzero(cm2 <= C_DRY)
        if ii.size == 0:
            return None, float(oo["t_final"]), float(cm2.min())
        j = int(ii[0])
        if j == 0:
            return float(oo["cmax"]["t"][0]), float(oo["t_final"]), float(cm2.min())
        ta_, tb_ = oo["cmax"]["t"][j - 1], oo["cmax"]["t"][j]
        ca_, cb_ = cm2[j - 1], cm2[j]
        return (float(ta_ + (C_DRY - ca_) / (cb_ - ca_) * (tb_ - ta_)),
                float(tE[-1]), float(cm2.min()))

    T_eff_inst = TaE - dT_qs
    T_CAP_E1 = 259200.0
    t_a, tf_a, cmin_a = t_dry_with_Teff(T_eff_inst, T_CAP_E1)
    T_eff_avg = TaE - dT_avg
    t_b, tf_b, cmin_b = t_dry_with_Teff(T_eff_avg, T_CAP_E1)
    E1 = (t_b - base) / base if t_b else None
    return t_a, cmin_a, t_b, cmin_b, E1, T_CAP_E1


def _e1_report(tE, dT_qs, P_avg, dT_avg, T_CAP_E1, t_a, cmin_a, t_b, cmin_b, E1):
    print("   δT_qs 峰值 %.2f K（t=%.0f s）；平均 ΔT = %.4f K（P_avg = %.4f W/m）"
          % (dT_qs.max(), tE[int(np.argmax(dT_qs))], dT_avg, P_avg), flush=True)
    print("   读法(a) 瞬时 δT(t)：72 h 窗口内 %s（C_max 终值 %.5f）"
          % ("达标 t_dry=%s h" % ("%.4f" % (t_a / 3600.0)) if t_a else "未达标",
             cmin_a), flush=True)
    print("   读法(b) 平均 ΔT = %.4f K：t_dry^corr %s h；E1 = %s %%"
          % (dT_avg, ("%.4f" % (t_b / 3600.0)) if t_b else "未达标",
             ("%+.2f" % (100 * E1)) if E1 is not None else "-"), flush=True)
    e1 = dict(method=("0-D 集总对照（§11.2 (Q4-34)）；`C_s(t)` 与 `R(t)` 取自主模型，"
                      "不含新的水分模型。两种读法：(a) 瞬时 δT(t)（准稳态表面平衡）；"
                      "(b) 平均 ΔT（先把蒸发功率在干燥窗内做时间平均，再取常数偏移）。"),
              borrowed_parameters=dict(dH_vap_J_per_kg=DH_VAP, rho_d_kg_m3=RHO_D_BORROW,
                                       note="题目未给，借自文献 / 附录 4 折算式；"
                                            "只作量级估计，绝不进交付数值"),
              delta_T_qs_max_K=float(dT_qs.max()),
              delta_T_qs_max_at_s=float(tE[int(np.argmax(dT_qs))]),
              delta_T_qs_final_K=float(dT_qs[-1]),
              P_evap_avg_W_per_m=P_avg,
              delta_T_avg_K=float(dT_avg),
              reading_a_instantaneous=dict(
                  t_dry_h=(t_a / 3600.0) if t_a else None,
                  reached_within_s=T_CAP_E1, C_max_final=cmin_a,
                  verdict=("在 72 h 窗口内未达标" if not t_a else "达标"),
                  note=("瞬时读法在早段把表面蒸发通量按 `C_s(0) = 2.55` 全额计入，"
                        "δT 峰值 %.1f K，材料温度被压到远低于初值 —— 这是 0-D 集总"
                        "的失效模式（真实药材有内部导热供热，表面不会瞬间掉到湿球温度）。"
                        "该读法只作定性说明，不作为 E1 数值。" % dT_qs.max())),
              reading_b_averaged=dict(
                  t_dry_h=(t_b / 3600.0) if t_b else None,
                  reached_within_s=T_CAP_E1, C_max_final=cmin_b),
              E1_relative=E1,
              E1_direction="t_dry 偏短（潜热缺失  温度偏高  D 偏高）",
              E1_adopted="读法 (b)（平均 ΔT）；读法 (a) 的结果如实登记但不采用",
              not_main_result=True,
              note="E1 的修正结果不得作为主结果（§11.6 纪律 3）")
    return e1


def _e2_e3_blocks(grid2):
    e2 = dict(R_over_L=R0 / (LENGTH_CM / 100.0),
              area_ratio=1.0 + R0 / (LENGTH_CM / 100.0),
              upper_bound_pct=100.0 * R0 / (LENGTH_CM / 100.0),
              direction="t_dry 偏长（1D 无限长漏掉端面传质）",
              note=("与 E1/E3 方向相反、部分相消（§11.3）；题目未给轴向数据，"
                    "各向同性收缩下 R/L 不变。"))

    v3 = [v["t_dry_h"] for v in grid2.values() if v["t_dry_h"]]
    e3 = dict(h_hm_3x3_spread_h=max(v3) - min(v3),
              h_hm_3x3_spread_pct=100.0 * (max(v3) - min(v3)) / min(v3),
              direction="双向",
              borrowed_params=False)
    return v3, e2, e3


def _e4_e5_e6_blocks(RES, v6):
    e4 = dict(source="results/q4_convergence.json", direction="双向")
    pconv = RES / "q4_convergence.json"
    if pconv.exists():
        cj = json.loads(pconv.read_text(encoding="utf-8"))
        e4["grid_N400_vs_N800_t_ge_3600s"] = cj["A_grid_independence"][
            "diff_vs_N800"]["400"]["max_abs_t_ge_3600s"]
        e4["grid_N400_vs_N800_fullwindow"] = cj["A_grid_independence"][
            "diff_vs_N800"]["400"]["max_abs"]
        e4["dt_prod_vs_finest_t_ge_3600s"] = cj["B_time_convergence"][
            "diff_vs_finest"]["I1.25_II5"]["max_abs_t_ge_3600s"]
        e4["dt_prod_vs_finest_fullwindow"] = cj["B_time_convergence"][
            "diff_vs_finest"]["I1.25_II5"]["max_abs"]
        e4["t_dry_grid_spread_h"] = cj["A_grid_independence"]["t_dry_spread_h"]
        e4["t_dry_dt_spread_h"] = cj["B_time_convergence"]["t_dry_spread_h"]
        e4["statement"] = ("t ≥ 1 h：Δξ 与 Δt 的联合离散不确定度均 < 5×10⁻⁵（② 层线）；"
                           "t < 1 h：由初始边界层主导，最大差 %.2e（表面列 @ t=60 s）"
                           % cj["A_grid_independence"]["diff_vs_N800"]["400"]["max_abs"])
    tt2, RR2 = att2()
    from scipy.interpolate import PchipInterpolator
    tc = np.linspace(0.0, min(259200.0, 200000.0), 4001)
    lin = np.interp(tc, tt2, RR2)
    pchp = PchipInterpolator(tt2, RR2)(tc)
    dR = np.abs(lin - pchp)
    quant = 0.001 / 100.0
    tt_int = np.arange(0.0, 259200.0 + 1e-9, 1800.0)
    e5 = dict(R_quantization_m=quant,
              pchip_monotone_ok=bool(np.all(np.diff(pchp) <= 1e-15)),
              pchip_range_ok=bool(pchp.min() >= R_MIN - 1e-12 and pchp.max() <= R0 + 1e-12),
              max_abs_diff_linear_vs_pchip_m=float(dR.max()),
              max_rel_diff=float((dR / np.maximum(lin, 1e-30)).max()),
              quant_equivalent_pct=100.0 * quant / R_MIN,
              direction="双向，小",
              note=("量化台阶 0.001 cm = %.4f %% of R_min；PCHIP 通过 `0.01198 ≤ R ≤ 0.02`"
                    "且 `Ṙ ≤ 0` 逐点校验（不合格则判 PCHIP 不合格，§12.2 组 5）。"
                    "主口径用分段线性（§4.3），PCHIP 仅作对照。"
                    % (100.0 * quant / R_MIN)))
    e6 = dict(ta_scenario_spread_h=(max(v6) - min(v6)) if v6 else None,
              ca_scenario_min_h=min(v6) if v6 else None,
              ca_scenario_max_h=max(v6) if v6 else None,
              direction="偏长为主（冻结压低后期驱动力）",
              borrowed_params=False)
    return e4, e5, e6, dR, quant


def _write_outputs(sens, base, e1, e2, e3, e4, e5, e6, E1, t_all):
    budget = {"generated_at": datetime.now().isoformat(timespec="seconds"),
              "baseline_t_dry_h": base / 3600.0, "baseline_t_dry_s": base,
              "E1_latent_heat": e1, "E2_end_face": e2, "E3_h_hm": e3,
              "E4_discrete": e4, "E5_R_interp": e5, "E6_driver": e6,
              "reporting_rule": ("E1–E4 分别报告，不做「总误差 = 各项相加」"
                                 "（方向不同、量级不同、互不独立）；论文以区间 + 方向"
                                 "表述（§11.1）；E1 的借用参数结果只进误差分析节。"),
              "interval_statement": (
                  "主结果 t_dry = %.4f h（E0 主情景、N=%d、Δt=%.3g/%.3g s）；"
                  "考虑 E1（本模型缺潜热  t_dry 偏短 %s  真值变长，上端 = base×(1+E1)）"
                  "与 E2（本模型 1D 无限长  t_dry 偏长 ≤ +%.1f %%  真值变短，"
                  "下端 = base×(1−E2)）的方向性偏差后，合理区间约为 [%.2f, %.2f] h"
                  "（方向性上界，非统计置信区间）。"
                  % (base / 3600.0, N_SENS, DT1, DT2,
                     ("%.1f %%" % (100 * abs(E1))) if E1 is not None else "待定",
                     e2["upper_bound_pct"],
                     base / 3600.0 * (1.0 - e2["upper_bound_pct"] / 100.0),
                     base / 3600.0 * (1.0 + (E1 if E1 else 0.0))))}
    (RES / "q4_error_budget.json").write_text(
        json.dumps(budget, ensure_ascii=False, indent=2), encoding="utf-8")
    sens["wall_s"] = time.time() - t_all
    (RES / "q4_sensitivity.json").write_text(
        json.dumps(sens, ensure_ascii=False, indent=2), encoding="utf-8")


def _print_summary(grid2, v3, v6, E1, dR, quant, sens):
    print("[summary] h_m×h 3×3：t_dry ∈ [%.4f, %.4f] h（极差 %.4f h = %.2f %%）；%s"
          % (min(v3), max(v3), max(v3) - min(v3),
             100.0 * (max(v3) - min(v3)) / min(v3), grid2 and "方向解释见 JSON"))
    print("[summary] 驱动情景：t_dry ∈ [%.4f, %.4f] h（极差 %.4f h）"
          % (min(v6), max(v6), max(v6) - min(v6)) if v6 else "")
    print("[summary] E1 = %s %%（借用 ΔH_vap/ρ_d，只进误差分析节）；E2 = ≤ +8.0 %%"
          % (("%+.2f" % (100 * E1)) if E1 is not None else "-"))
    print("[summary] E5：PCHIP vs 分段线性 max |ΔR| = %.3e m；量化 0.001 cm = %.3f %% of R_min"
          % (dR.max(), 100.0 * quant / R_MIN))
    print("[out] results/q4_sensitivity.json、results/q4_error_budget.json；总耗时 %.1f s"
          % sens["wall_s"])


def main():
    import os
    skip = os.environ.get("MM_Q4_SENS_SKIP") == "1"
    RES.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    if skip:
        prev = json.loads((RES / "q4_sensitivity.json").read_text(encoding="utf-8"))
        sens = prev
        base = prev["baseline"]["t_dry_s"]
        rows = prev["single_parameter"]
        grid2 = prev["h_hm_2d"]["grid"]
        scen = prev["driver_scenarios"]["rows"]
        v3 = [v["t_dry_h"] for v in grid2.values() if v["t_dry_h"]]
        v6 = [v["t_dry_h"] for v in scen.values() if v["t_dry_h"]]
        sens["baseline"]["note"] = ("MM_Q4_SENS_SKIP=1：复用上次的单参数/二维/情景扫描结果，"
                                    "仅重算 E1 与误差预算（2026-09-11 的 E1 读法修正）")
        print("[skip] 复用上次扫描结果（基线 %.6f h），仅重算 E1 与误差预算"
              % (base / 3600.0), flush=True)
    else:
        base, _ = t_dry_of()
        print("[基线] t_dry = %.6f s = %.6f h（N=%d、Δt=%.3g/%.3g s）"
              % (base, base / 3600.0, N_SENS, DT1, DT2), flush=True)
        sens = {"baseline": dict(t_dry_s=base, t_dry_h=base / 3600.0,
                                 N=N_SENS, dt1=DT1, dt2=DT2)}
        rows = {}

    if not skip:
        sens, rows, grid2 = _scan_single_and_2d(sens, base, rows)
        sens, scen, v6 = _scan_scenarios(sens, base)

    tE, TaE, dT_qs, P_avg, dT_avg = _e1_zero_d(base)
    t_a, cmin_a, t_b, cmin_b, E1, T_CAP_E1 = _e1_adopt(tE, TaE, dT_qs, dT_avg, base)
    e1 = _e1_report(tE, dT_qs, P_avg, dT_avg, T_CAP_E1, t_a, cmin_a, t_b, cmin_b, E1)
    v3, e2, e3 = _e2_e3_blocks(grid2)
    e4, e5, e6, dR, quant = _e4_e5_e6_blocks(RES, v6)
    _write_outputs(sens, base, e1, e2, e3, e4, e5, e6, E1, t_all)
    _print_summary(grid2, v3, v6, E1, dR, quant, sens)
    return 0


if __name__ == "__main__":
    sys.exit(main())
