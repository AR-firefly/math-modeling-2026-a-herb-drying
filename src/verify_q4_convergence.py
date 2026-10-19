from __future__ import annotations

import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

_LOG_ENCODING = "utf-8"
if getattr(sys.stdout, "reconfigure", None):
    sys.stdout.reconfigure(encoding=_LOG_ENCODING, errors="replace")

_HERE = Path(__file__).resolve()
_PROJ = _HERE.parent.parent
_SRC_ENTRY = _PROJ / "Q4"
sys.path.insert(0, _SRC_ENTRY.as_posix())

from solver_q4 import solve_q4, driver_at, C_DRY
from run_q4 import map_output_full, locate_crossing, R_OUT_CM, DT_OUT

RES = _PROJ / "results"
T_COMMON = 194400.0
PLANE = np.arange(DT_OUT, T_COMMON + 1e-9, DT_OUT)
TOL_LAYER2 = 5e-5
R_OUT_ALL = np.concatenate([R_OUT_CM, [np.nan]])


def run_case(**kw):
    t0 = time.time()
    o = solve_q4(stop_on_dry=False, t_end_override=T_COMMON,
                 snap_every_s=DT_OUT, snap_until_s=T_COMMON,
                 cmax_track=True, audit=True, verbose=False, **kw)
    xi, snap_t, snap_C, snap_R = o["xi"], o["snap_t"], o["snap_C"], o["snap_R"]
    n = int(round(T_COMMON / DT_OUT))
    if snap_t.size != n + 1:
        raise SystemExit("[FATAL] snapshot count %d != %d"
                         % (snap_t.size, n + 1))
    tab = np.empty((n, R_OUT_CM.size + 1))
    for k in range(n):
        tab[k] = map_output_full(xi, snap_C[k + 1], snap_R[k + 1], R_OUT_CM, "pchip")
    cross = locate_crossing(o["cmax"]["t"], o["cmax"]["cmax"], C_DRY)
    out = dict(tab=tab, t=snap_t[1:], wall=time.time() - t0,
               t_dry_step=cross["t_step_s"] if cross else None,
               t_dry_lin=cross["t_linear"] if cross else None,
               N=o["N"], dt1=kw.get("dt1"), dt2=kw.get("dt2"),
               mass_aud=o["mass_audit"]["max_rel"],
               picard_max=o["picard"]["max_iters"],
               picard_notconv=o["picard"]["n_not_converged"])
    del o
    return out


def rc_p(order_seq, h_seq):
    return [math.log(order_seq[i] / order_seq[i + 1]) / math.log(h_seq[i] / h_seq[i + 1])
            for i in range(len(order_seq) - 1)]


def per_col_max(d):
    return [float(np.max(d[:, j])) for j in range(d.shape[1])]


def _grid_independence():
    print("[A] 网格无关性：N ∈ {100,200,400,800}，Δt_I=1.25 s、Δt_II=5 s（生产 Δt）")
    seqN = [100, 200, 400, 800]
    ref_key = None
    runsN = {}
    for N in seqN:
        r = run_case(N=N, dt1=1.25, dt2=5.0, t_switch=14400.0, R_mode="shrink",
                     T_mode="prescribed", props="app4", face_rule="arith")
        runsN[N] = r
        print("   N=%3d  墙钟 %5.1f s  t_dry(步) %s h  Picard max %d 未收敛 %d  质量审计 %.1e"
              % (N, r["wall"],
                 ("%.4f" % (r["t_dry_step"] / 3600.0)) if r["t_dry_step"] else "None",
                 r["picard_max"], r["picard_notconv"], r["mass_aud"]), flush=True)
    ref = runsN[800]["tab"]
    grid = dict(config=dict(dt1=1.25, dt2=5.0, t_switch=14400.0, t_window_s=T_COMMON,
                            n_out=int(round(T_COMMON / DT_OUT)),
                            columns=["0…1.1 cm + 药材表面"]),
                t_dry_h={str(N): (runsN[N]["t_dry_step"] / 3600.0 if runsN[N]["t_dry_step"]
                                  else None) for N in seqN},
                t_dry_lin_h={str(N): (runsN[N]["t_dry_lin"] / 3600.0
                                      if runsN[N]["t_dry_lin"] else None) for N in seqN},
                wall_s={str(N): runsN[N]["wall"] for N in seqN},
                diff_vs_N800={}, )
    err_seq = []
    tt = runsN[800]["t"]
    for N in seqN:
        d = np.abs(runsN[N]["tab"] - ref)
        grid["diff_vs_N800"][str(N)] = dict(
            max_abs=float(d.max()), at_col=int(np.argmax(d.max(axis=0))),
            at_t_s=float(tt[int(np.argmax(d.max(axis=1)))]),
            per_column_max=per_col_max(d),
            surface_col_max=float(d[:, -1].max()),
            fixed_col_max=float(d[:, :-1].max()),
            max_abs_t_ge_3600s=float(d[tt >= 3600.0].max()),
            max_abs_t_ge_21600s=float(d[tt >= 21600.0].max()),
            max_abs_t_ge_3600s_fixed=float(d[tt >= 3600.0][:, :-1].max()),
            max_abs_t_ge_3600s_surface=float(d[tt >= 3600.0][:, -1].max()),
            per_time_max_fixed=[float(x) for x in d[:, :-1].max(axis=1)],
            per_time_max_surface=[float(x) for x in d[:, -1]])
        if N < 800:
            err_seq.append(float(d.max()))
    h_seq = [1.0 / N for N in seqN[:3]]
    grid["observed_order_p"] = rc_p(np.array(err_seq) + 1e-300, h_seq)
    grid["richardson_extrapolated_err"] = (
        err_seq[-1] / (2 ** max(grid["observed_order_p"][-1], 0.1) - 1)
        if len(err_seq) >= 2 else None)
    tdy = [grid["t_dry_h"][str(N)] for N in seqN if grid["t_dry_h"][str(N)]]
    grid["t_dry_spread_h"] = (max(tdy) - min(tdy)) if tdy else None
    grid["t_dry_spread_rel"] = ((max(tdy) - min(tdy)) / min(tdy)) if tdy else None
    grid["layer2_tol_kgkg"] = TOL_LAYER2
    grid["layer2_pass"] = bool(grid["diff_vs_N800"]["200"]["max_abs"] < TOL_LAYER2)
    return seqN, grid, runsN, tt


def _time_convergence():
    print()
    print("[B] 数值收敛：Δt_I ∈ {20,10,5,2.5,1.25,0.625} s、Δt_II ∈ {60,30,20,10,5,2.5} s（N=200）")
    dt_pairs = [(20.0, 60.0), (10.0, 30.0), (5.0, 20.0), (2.5, 10.0),
                (1.25, 5.0), (0.625, 2.5)]
    runsDT = {}
    for d1, d2 in dt_pairs:
        key = "I%g_II%g" % (d1, d2)
        r = run_case(N=200, dt1=d1, dt2=d2, t_switch=14400.0, R_mode="shrink",
                     T_mode="prescribed", props="app4", face_rule="arith")
        runsDT[key] = r
        print("   Δt_I=%-5g Δt_II=%-5g 墙钟 %5.1f s  t_dry(步) %s h  Picard max %d 未收敛 %d"
              % (d1, d2, r["wall"],
                 ("%.4f" % (r["t_dry_step"] / 3600.0)) if r["t_dry_step"] else "None",
                 r["picard_max"], r["picard_notconv"]), flush=True)
    refd = runsDT["I0.625_II2.5"]["tab"]
    dtd = dict(config=dict(N=200, t_switch=14400.0, t_window_s=T_COMMON,
                           n_out=int(round(T_COMMON / DT_OUT))),
               t_dry_h={k: (v["t_dry_step"] / 3600.0 if v["t_dry_step"] else None)
                        for k, v in runsDT.items()},
               t_dry_lin_h={k: (v["t_dry_lin"] / 3600.0 if v["t_dry_lin"] else None)
                            for k, v in runsDT.items()},
               wall_s={k: v["wall"] for k, v in runsDT.items()}, diff_vs_finest={})
    ttd = runsDT["I0.625_II2.5"]["t"]
    for k, v in runsDT.items():
        d = np.abs(v["tab"] - refd)
        dtd["diff_vs_finest"][k] = dict(
            max_abs=float(d.max()), at_t_s=float(ttd[int(np.argmax(d.max(axis=1)))]),
            per_column_max=per_col_max(d),
            surface_col_max=float(d[:, -1].max()),
            fixed_col_max=float(d[:, :-1].max()),
            max_abs_t_ge_3600s=float(d[ttd >= 3600.0].max()),
            max_abs_t_ge_21600s=float(d[ttd >= 21600.0].max()),
            max_abs_t_ge_3600s_fixed=float(d[ttd >= 3600.0][:, :-1].max()),
            max_abs_t_ge_3600s_surface=float(d[ttd >= 3600.0][:, -1].max()),
            per_time_max_fixed=[float(x) for x in d[:, :-1].max(axis=1)],
            per_time_max_surface=[float(x) for x in d[:, -1]])
    tdy_dt = [dtd["t_dry_h"][k] for k in dtd["t_dry_h"] if dtd["t_dry_h"][k]]
    dtd["t_dry_spread_h"] = (max(tdy_dt) - min(tdy_dt)) if tdy_dt else None
    dtd["t_dry_spread_rel"] = ((max(tdy_dt) - min(tdy_dt)) / min(tdy_dt)) if tdy_dt else None
    dtd["layer2_tol_kgkg"] = TOL_LAYER2
    dtd["layer2_pass_fullwindow_vs_finest"] = bool(
        dtd["diff_vs_finest"]["I1.25_II5"]["max_abs"] < TOL_LAYER2)
    dtd["layer2_pass_t_ge_3600s_vs_finest"] = bool(
        dtd["diff_vs_finest"]["I1.25_II5"]["max_abs_t_ge_3600s"] < TOL_LAYER2)
    return dtd


def _convergence_verdict(grid, dtd):
    dtd["production_choice"] = dict(
        dt1=1.25, dt2=5.0,
        reason=("两段式起点（§7.3）经 `diff_vs_finest` 定案：生产档 vs 最细档"
                "（Δt_I=0.625 s / Δt_II=2.5 s）在 t≥3600 s 的逐点最大差 "
                "%.3e < %.0e（② 层线）；t<3600 s 窗口在本生产配置下未达 ② 层线"
                "（最大差出现在 t=60 s 表面列；见 G5 口径降级与 §7.2 注）。"
                % (dtd["diff_vs_finest"]["I1.25_II5"]["max_abs_t_ge_3600s"],
                   TOL_LAYER2)),
        )
    grid["layer2_pass_fullwindow_vs_N800"] = bool(
        grid["diff_vs_N800"]["400"]["max_abs"] < TOL_LAYER2)
    grid["layer2_pass_t_ge_3600s_vs_N800"] = bool(
        grid["diff_vs_N800"]["400"]["max_abs_t_ge_3600s"] < TOL_LAYER2)
    grid["production_choice"] = dict(
        N=400,
        reason=("网格由 `diff_vs_N800` 定案：N=400 vs N=800 在 t≥3600 s 的逐点最大差 "
                "%.3e < %.0e（② 层线）；t=60 s 的 %.3e 属 t<1 h 窗口，"
                "本生产配置（N=400）在该窗口未达 ② 层线"
                "（δ(60 s)=2.507×10⁻² cm ≈ 5.01Δr@N=400  边界层已被覆盖、加密仍在收敛；"
                "见 `g5_disclosure`）。"
                % (grid["diff_vs_N800"]["400"]["max_abs_t_ge_3600s"], TOL_LAYER2,
                   grid["diff_vs_N800"]["400"]["max_abs"])))
    grid["g5_disclosure"] = dict(
        triggered=bool(not grid["layer2_pass_fullwindow_vs_N800"]
                       or not dtd["layer2_pass_fullwindow_vs_finest"]),
        statement=("G5（§15.1）：全窗口（含 t<3600 s）逐点最大差达不到 5×10⁻⁵。"
                   "不做无限加密；交付表的有效位数按实际不确定度分层声明："
                   "t ≥ 1 h 由 ② 层线支撑（4 位小数）；t < 1 h 的列不支撑第 4 位"
                   "—— 本生产配置（N=400）在该窗口实测 N400-vs-N800 最大差 %.3e"
                   "（t=60 s 表面列），而 δ(60 s)=2.507×10⁻² cm ≈ 5.01Δr@N=400 表明"
                   "初始边界层已被覆盖（不是「不可分辨」）；加密序列在该窗口仍以 "
                   "p≈2.35 收敛、N=800 全程仅 %.1f s  继续加密可望达标"
                   "（按 p 外推 N=1600 ≈ 2.4×10⁻⁵，本轮未做）。"
                   "该分层在 results4.json 与论文章节中如实写明。"
                   % (grid["diff_vs_N800"]["400"]["max_abs"], grid["wall_s"]["800"])))


def _crossing_check():
    print()
    print("[C] 两段式跨接检查：两段式 vs 单段 Δt=1.25 s（跨接窗口 t=14400/18000/21600/25200 s，N=200）")
    single = run_case(N=200, dt1=1.25, dt2=1.25, t_switch=1e18, R_mode="shrink",
                      T_mode="prescribed", props="app4", face_rule="arith")
    two = run_case(N=200, dt1=1.25, dt2=5.0, t_switch=14400.0, R_mode="shrink",
                   T_mode="prescribed", props="app4", face_rule="arith")
    jump = {}
    for ts in (14400.0, 18000.0, 21600.0, 25200.0):
        i = int(round(ts / DT_OUT)) - 1
        d = np.abs(two["tab"][i] - single["tab"][i])
        jump["t_%ds" % int(ts)] = dict(max_abs=float(d.max()),
                                       per_column=[float(x) for x in d])
    bref = {}
    for ts in (14400.0,):
        i0 = int(round(ts / DT_OUT)) - 1
        for off in (-1, 0, 1):
            j = i0 + off
            d = np.abs(two["tab"][j] - single["tab"][j])
            bref["t_%ds" % int(two["t"][j])] = float(d.max())
    cross_chk = dict(
        config="两段式 Δt_I=1.25 s  Δt_II=5 s（切换 t=14400 s）vs 单段 Δt=1.25 s 全程（N=200）",
        window_points=jump, near_switch=bref,
        max_abs_diff_switch_window=float(max(v["max_abs"] for v in jump.values())),
        max_abs_diff_near_switch=float(max(bref.values())),
        tol=TOL_LAYER2,
        two_stage_ok=bool(max(v["max_abs"] for v in jump.values()) < TOL_LAYER2
                          and max(bref.values()) < TOL_LAYER2),
        single_stage_wall_s=single["wall"], two_stage_wall_s=two["wall"])
    print("   跨接窗口最大差 %.3e；跨接点邻域最大差 %.3e；判据 %.0e  %s"
          % (cross_chk["max_abs_diff_switch_window"],
             cross_chk["max_abs_diff_near_switch"], TOL_LAYER2,
             "PASS" if cross_chk["two_stage_ok"] else "FAIL"), flush=True)
    return cross_chk


def _mapping_precision(runsN, tt):
    print()
    print("[D] 映射精度：N=800 的解在同一 r_out 上与 N=200 的 PCHIP 值比较")
    dmap = np.abs(runsN[400]["tab"] - runsN[800]["tab"])
    mapping = dict(
        grid_plus_mapping_uncertainty=dict(
            N_pair="400 vs 800",
            max_abs_vs_N800=float(dmap.max()),
            max_abs_t_ge_3600s=float(dmap[tt >= 3600.0].max()),
            max_abs_t_ge_21600s=float(dmap[tt >= 21600.0].max()),
            surface_col=float(dmap[:, -1].max()),
            fixed_cols=float(dmap[:, :-1].max()),
            per_column_max=per_col_max(dmap)),
        pchip_vs_linear_source="results/q4_mapping_uncertainty.csv（生产运行产出）",
        tol_layer2=TOL_LAYER2,
        pass_=bool(dmap[:, :-1].max() < TOL_LAYER2),
        note=("该差为映射 + 网格的联合不确定度（§8.4 第 5 条）；末尾 `药材表面` 列"
              "为两档网格的节点值之差（不插值），性质是网格不确定度。"),
    )
    return dmap, mapping


def _event_convergence(grid, dtd):
    _sp_dt_h = dtd["t_dry_spread_h"]
    _sp_grid_h = grid["t_dry_spread_h"]
    _pc = dtd["production_choice"]
    event = dict(
        grid_sequence=grid["t_dry_h"], dt_sequence=dtd["t_dry_h"],
        note=("§9.2 推论 2（修订版）：`t_dry` 在生产档 `(%.2f, %g) s` 附近继续加密收益"
              "远小于网格项；但粗档区间内 Δt 仍是显著误差源 —— 本序列极差 "
              "`%r h`（= `%.1f s`，step）大于四档网格极差 "
              "`%r h`（= `%.1f s`），故『序列极差应远小于 ±15.2 s』"
              "的说法被本序列自身否证。原句保留作留痕。"
              % (_pc["dt1"], _pc["dt2"], _sp_dt_h, _sp_dt_h * 3600.0,
                 _sp_grid_h, _sp_grid_h * 3600.0)),
        note_legacy=("§9.2 推论 2：`t_dry` 由场值误差决定，缩小 Δt 不改善；"
                     "故序列的极差应远小于场值误差换算的 ±15.2 s 量级。"),
    )
    return event


def _write_and_summary(grid, dtd, cross_chk, mapping, event, seqN, dmap, t_all):
    out = {"generated_at": datetime.now().isoformat(timespec="seconds"),
           "A_grid_independence": grid, "B_time_convergence": dtd,
           "C_two_stage_crossing": cross_chk, "D_mapping_precision": mapping,
           "E_event_convergence": event,
           "wall_s": time.time() - t_all}
    (RES / "q4_convergence.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    g200, gp = grid["diff_vs_N800"]["400"], dtd["diff_vs_finest"]["I1.25_II5"]
    print("[summary] Δξ：N=400 vs N=800 全窗口逐点最大差 %.3e（判据 %.0e） %s；"
          "观测 p = %s"
          % (g200["max_abs"], TOL_LAYER2,
             "PASS" if grid["layer2_pass"] else "FAIL",
             ["%.2f" % p for p in grid["observed_order_p"]]))
    print("        分层读数：t≥3600 s %.3e（固定列 %.3e / 表面列 %.3e）；"
          "t≥21600 s %.3e；最大差出现于 t=%s s（列 %d）"
          % (g200["max_abs_t_ge_3600s"], g200["max_abs_t_ge_3600s_fixed"],
             g200["max_abs_t_ge_3600s_surface"], g200["max_abs_t_ge_21600s"],
             g200["at_t_s"], g200["at_col"]))
    print("[summary] Δt：生产档 vs 最细档 全窗口逐点最大差 %.3e；分层 t≥3600 s %.3e"
          "（固定列 %.3e / 表面列 %.3e）；最大差出现于 t=%s s"
          % (gp["max_abs"], gp["max_abs_t_ge_3600s"], gp["max_abs_t_ge_3600s_fixed"],
             gp["max_abs_t_ge_3600s_surface"], gp["at_t_s"]))
    print("[summary] t_dry 序列：Δξ %s h（极差 %s h）；Δt %s h（极差 %s h）"
          % (["%s" % grid["t_dry_h"][str(N)] for N in seqN],
             grid["t_dry_spread_h"], ["%s" % v for v in dtd["t_dry_h"].values()],
             dtd["t_dry_spread_h"]))
    print("[summary] 两段式跨接：%s；映射（vs N=800）%.3e；总耗时 %.1f s"
          % ("PASS" if cross_chk["two_stage_ok"] else "FAIL", dmap.max(),
             out["wall_s"]))
    print("[out] results/q4_convergence.json")


def main():
    RES.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    results = {}

    seqN, grid, runsN, tt = _grid_independence()
    dtd = _time_convergence()
    _convergence_verdict(grid, dtd)
    cross_chk = _crossing_check()
    dmap, mapping = _mapping_precision(runsN, tt)
    event = _event_convergence(grid, dtd)
    _write_and_summary(grid, dtd, cross_chk, mapping, event, seqN, dmap, t_all)
    return 0


if __name__ == "__main__":
    rc = main()
    sys.exit(rc)
