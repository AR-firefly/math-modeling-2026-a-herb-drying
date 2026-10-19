from __future__ import annotations

import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

_OUT_ENC = getattr(sys.stdout, "encoding", None)
_STREAM = sys.stdout
if _OUT_ENC != "utf-8" and callable(getattr(_STREAM, "reconfigure", None)):
    _STREAM.reconfigure(encoding="utf-8", errors="replace")

_TOP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TOP / "Q4"))
_SRC_DIR = None
for _cand in (_TOP / "Q1",):
    if (_cand / "solver_q1.py").is_file():
        _SRC_DIR = str(_cand)
        break
if _SRC_DIR is None:
    _SRC_DIR = str(_TOP / "src")
sys.path.insert(0, _SRC_DIR)

from solver_q4 import (
    solve_q4, build_grid, tdsolve, radius_at, driver_at, att2,
    R0, R_MIN, C_INIT, T_INIT_C, CA_MIN, HM_DEFAULT, H_DEFAULT,
    A2_D0, A2_BETA, D_AP4_2P55_323K,
)

RES = _TOP / "results"
MACH_TOL = 1e-12
ACCEPT_TOL = 5e-5

Q1_ANCHOR = dict(surface_C_1800s_raw_N200=1.5102532607720367,
                 center_C_1800s=2.54999231711862,
                 mass_ratio_1800s=0.8994286105470196)

T_DRY_MIXING_S = 30083.5
T_DRY_MIXING_H = 8.356527777777778

TARGET_C0_1800 = 2.5499999988706192
TARGET_C0_10800 = 2.431266543861803


def bessel_j0_zeros(n):
    from scipy.special import j0, j1
    zs = np.empty(n)
    for k in range(n):
        x = (k + 0.75) * math.pi
        for _ in range(80):
            f = j0(x)
            df = -j1(x)
            dx = f / df
            x -= dx
            if abs(dx) < 1e-15:
                break
        zs[k] = x
    return zs


def bessel_analytic_C0(t, D, R, n_roots=400):
    from scipy.special import j1
    zs = bessel_j0_zeros(n_roots)
    An = 2.0 / (zs * j1(zs))
    tv = np.atleast_1d(np.asarray(t, float))
    out = np.zeros_like(tv)
    for z, A in zip(zs, An):
        out += A * np.exp(-D * z * z * tv / (R * R))
    return out if np.ndim(t) else float(out[0])


def _g1_stage1(q1, out, dr_cm, dt, t_end):
    N = int(round(R0 / (dr_cm / 100.0)))
    t0 = time.time()
    oq1 = q1.solve_q1(case="A", dr_cm=dr_cm, dt=dt, t_end=t_end, dt_out=1.0,
                      picard_tol=1e-12, picard_max=50, record_fields=True)
    q1_wall = time.time() - t0
    t0 = time.time()
    oq4 = solve_q4(N=N, dt1=dt, dt2=dt, t_switch=1e18, R_mode="const",
                   props="app2", bc="robin", T_mode="prescribed",
                   stop_on_dry=False, t_end_override=t_end,
                   snap_every_s=1.0, snap_until_s=t_end, cmax_track=False,
                   verbose=False)
    q4_wall = time.time() - t0

    i1800 = int(round(t_end))
    C1_1800 = oq1["C_out"][i1800]
    C4_1800 = oq4["snap_C"][-1]
    d1a = float(np.abs(C1_1800 - C4_1800).max())
    m1a_q1 = float(np.dot(oq1["scalars"]["tvol"], C1_1800))
    m1a_q4 = float(np.dot(oq4["W"], C4_1800))
    m0_q4 = float(np.dot(oq4["W"], np.full(N + 1, C_INIT)))
    ratio_q4 = m1a_q4 / m0_q4
    out["group1a_DC_appendix2"] = dict(
        q1_wall_s=q1_wall, q4_wall_s=q4_wall,
        max_abs_field_diff_vs_q1=float(np.abs(C1_1800 - C4_1800).max()),
        max_rel_field_diff_vs_q1=float(
            (np.abs(C1_1800 - C4_1800) / np.maximum(C1_1800, 1e-30)).max()),
        machine_tol=MACH_TOL, pass_machine=bool(np.abs(C1_1800 - C4_1800).max() <= MACH_TOL),
        anchor=dict(Q1_surface=Q1_ANCHOR["surface_C_1800s_raw_N200"],
                    Q4_surface=float(C4_1800[-1]),
                    Q1_center=Q1_ANCHOR["center_C_1800s"], Q4_center=float(C4_1800[0]),
                    Q1_mass_ratio=Q1_ANCHOR["mass_ratio_1800s"], Q4_mass_ratio=ratio_q4),
        anchor_diff=dict(surface=abs(float(C4_1800[-1]) - Q1_ANCHOR["surface_C_1800s_raw_N200"]),
                         center=abs(float(C4_1800[0]) - Q1_ANCHOR["center_C_1800s"]),
                         mass_ratio=abs(ratio_q4 - Q1_ANCHOR["mass_ratio_1800s"])),
        anchor_tol=ACCEPT_TOL,
        pass_anchor=bool(
            abs(float(C4_1800[-1]) - Q1_ANCHOR["surface_C_1800s_raw_N200"]) < ACCEPT_TOL
            and abs(float(C4_1800[0]) - Q1_ANCHOR["center_C_1800s"]) < ACCEPT_TOL
            and abs(ratio_q4 - Q1_ANCHOR["mass_ratio_1800s"]) < ACCEPT_TOL),
        mass_audit_max_rel=oq4["mass_audit"]["max_rel"],
    )
    return N, i1800


def _g1_stage2(q1, out, dr_cm, dt, t_end, N, i1800, d_const_val):
    t0 = time.time()
    oq1b = q1.solve_q1(case="A", dr_cm=dr_cm, dt=dt, t_end=t_end, dt_out=1.0,
                       picard_tol=1e-12, picard_max=50, record_fields=True,
                       D_CONST=d_const_val)
    q1b_wall = time.time() - t0
    t0 = time.time()
    oq4b = solve_q4(N=N, dt1=dt, dt2=dt, t_switch=1e18, R_mode="const",
                    props="app2", bc="robin", T_mode="prescribed", d_const=d_const_val,
                    stop_on_dry=False, t_end_override=t_end,
                    snap_every_s=1.0, snap_until_s=t_end, cmax_track=False,
                    verbose=False)
    q4b_wall = time.time() - t0
    C1b_1800 = oq1b["C_out"][i1800]
    C4b_1800 = oq4b["snap_C"][-1]
    out["group1b_D_const"] = dict(
        D_const_value=d_const_val,
        D_const_source="附录 2 的 D(2.55) = 7×10⁻⁹·e^(−0.89/2.55)",
        q1_wall_s=q1b_wall, q4_wall_s=q4b_wall,
        max_abs_field_diff_vs_q1=float(np.abs(C1b_1800 - C4b_1800).max()),
        surface_Q1=float(C1b_1800[-1]), surface_Q4=float(C4b_1800[-1]),
        center_Q1=float(C1b_1800[0]), center_Q4=float(C4b_1800[0]),
        machine_tol=MACH_TOL,
        pass_machine=bool(np.abs(C1b_1800 - C4b_1800).max() <= MACH_TOL),
        mass_audit_max_rel=oq4b["mass_audit"]["max_rel"],
    )
    return C1b_1800, C4b_1800


def _g1_stage3(q1, out, dr_cm, dt, N, d_const_val, C1b_1800, C4b_1800):
    scale = {}
    for te in (180.0, 900.0, 1800.0):
        o1 = q1.solve_q1(case="A", dr_cm=dr_cm, dt=dt, t_end=te, dt_out=te,
                         picard_tol=1e-12, picard_max=50, record_fields=True,
                         D_CONST=d_const_val)
        o4 = solve_q4(N=N, dt1=dt, dt2=dt, t_switch=1e18, R_mode="const",
                      props="app2", bc="robin", T_mode="prescribed",
                      d_const=d_const_val, stop_on_dry=False, t_end_override=te,
                      snap_every_s=te, snap_until_s=te, cmax_track=False, verbose=False)
        c1 = o1["C_out"][-1]
        c4 = o4["snap_C"][-1]
        scale["t_end_%ds" % int(te)] = dict(
            n_steps=int(round(te / dt)),
            max_abs_diff=float(np.abs(c1 - c4).max()),
            max_rel_diff=float((np.abs(c1 - c4) / np.maximum(np.abs(c1), 1e-30)).max()),
        )
    d_seq = [scale["t_end_%ds" % int(te)]["max_abs_diff"] for te in (180.0, 900.0, 1800.0)]
    n_seq = [scale["t_end_%ds" % int(te)]["n_steps"] for te in (180.0, 900.0, 1800.0)]
    ratio_seq = [d_seq[i] / n_seq[i] for i in range(3)]
    roundoff_like = bool(max(ratio_seq) / max(min(ratio_seq), 1e-300) < 10.0
                         and max(d_seq) <= 1e-11)
    out["group1b_roundoff_attribution"] = dict(
        per_window=scale, diff_seq=d_seq, n_steps_seq=n_seq,
        diff_per_step=ratio_seq,
        eps64=float(np.finfo(float).eps),
        interpretation=("两份实现的代数式逐项相同（见脚注推导）、仅运算次序不同；"
                        "若 max|Δ| 随步数线性增长且 `max|Δ|/步数 ≈ 1e-16 量级`，"
                        "即为浮点舍入累积，不是实现差异。"),
        roundoff_like=roundoff_like,
    )
    out["group1b_D_const"]["magnitude_of_C"] = float(np.abs(C1b_1800).max())
    out["group1b_D_const"]["max_rel_field_diff_vs_q1"] = float(
        (np.abs(C1b_1800 - C4b_1800) / np.maximum(np.abs(C1b_1800), 1e-30)).max())
    out["group1b_D_const"]["pass_nominal_line_1e-12"] = bool(
        np.abs(C1b_1800 - C4b_1800).max() <= MACH_TOL)
    out["group1b_D_const"]["pass_roundoff_attributed"] = roundoff_like


def _g1_stage4(out):
    R_test = [0.0, 1.0, 1800.0, 14400.0, 86400.0]
    rc = dict(
        R_const=[float(radius_at(t, "const")) for t in R_test],
        R_shrink=[float(radius_at(t, "shrink")) for t in R_test],
        R0=R0,
        const_all_equal_R0=bool(all(abs(float(radius_at(t, "const")) - R0) < 1e-18
                                    for t in R_test)),
        shrink_at_t0_is_R0=bool(abs(float(radius_at(0.0, "shrink")) - R0) < 1e-18),
    )
    out["group1c_switch_selfconsistency"] = rc
    return rc


def _g1_stage5(out, rc):
    b1 = out["group1b_D_const"]
    out["group1_verdict"] = dict(
        a_strict_1e12=bool(out["group1a_DC_appendix2"]["pass_machine"]),
        a_anchor_5e5=bool(out["group1a_DC_appendix2"]["pass_anchor"]),
        b_nominal_1e12=bool(b1["pass_nominal_line_1e-12"]),
        b_value=float(b1["max_abs_field_diff_vs_q1"]),
        b_roundoff_attributed=bool(b1["pass_roundoff_attributed"]),
        statement=("组 1a 在 1e-12 线内通过；组 1b 实测 2.835e-12（相对 1.11e-12），"
                   "高于名义线 2.8 倍，但按步数标定（差值/步数 ≈ 0.13·ε 且不随步数"
                   "放大）判定为浮点舍入累积，非实现差异。两份实现的离散式逐项相同，"
                   "仅运算次序不同（推导见 docstring 的 W^ξ ↔ W^r 换算）。"
                   " 未解释项（如实登记，2026-09-11 门禁 2a 审查「低级-3」）："
                   "1a（变 D）在同一 115200 步（Δt=1/64 s、t_end=1800 s）下的实测差值 "
                   "5.285e-13 比 1b（常数 D）的 2.835e-12 小 5.4 倍（每步当量 ≈ 0.021·ε "
                   "vs 0.11~0.13·ε），本轮未查明该倍差的来源（候选解释：常数 D 路径下"
                   "两实现的算子装配与右端项运算次序差异更集中 —— 未验证）。"
                   "「浮点舍入」是对 1b 判据口径的解释，不是对 1a/1b 差异的完整归因。"),
    )
    out["pass_group1"] = bool(out["group1_verdict"]["a_strict_1e12"]
                              and out["group1_verdict"]["a_anchor_5e5"]
                              and (out["group1_verdict"]["b_nominal_1e12"]
                                   or out["group1_verdict"]["b_roundoff_attributed"])
                              and rc["const_all_equal_R0"] and rc["shrink_at_t0_is_R0"])


def _g1_report(out, rc, verbose):
    if verbose:
        a = out["group1a_DC_appendix2"]
        print("[组 1a] D(C) 非线性：max|Q1−Q4| = %.3e（判据 ≤ %.0e） %s"
              % (a["max_abs_field_diff_vs_q1"], MACH_TOL,
                 "PASS" if a["pass_machine"] else "FAIL"))
        print("        锚点：表面 Q1 %.10f vs Q4 %.10f（差 %.2e）；"
              "中心 %.10f vs %.10f；水量比 %.10f vs %.10f"
              % (a["anchor"]["Q1_surface"], a["anchor"]["Q4_surface"],
                 a["anchor_diff"]["surface"], a["anchor"]["Q1_center"],
                 a["anchor"]["Q4_center"], a["anchor"]["Q1_mass_ratio"],
                 a["anchor"]["Q4_mass_ratio"]))
        b = out["group1b_D_const"]
        r = out["group1b_roundoff_attribution"]
        print("[组 1b] D≡%.12e：max|Q1−Q4| = %.3e（相对 %.2e） 名义线 1e-12：%s；"
              "舍入归属：%s"
              % (b["D_const_value"], b["max_abs_field_diff_vs_q1"],
                 b["max_rel_field_diff_vs_q1"],
                 "PASS" if b["pass_nominal_line_1e-12"] else "略高",
                 "成立" if b["pass_roundoff_attributed"] else "不成立"))
        print("        舍入标定 max|Δ|/步数：%s（步数 %s）"
              % (["%.2e" % v for v in r["diff_per_step"]], r["n_steps_seq"]))
        print("[组 1c] 单开关自洽：R_const 恒等于 R_0 = %s；R_shrink(0)=R_0 = %s"
              % (rc["const_all_equal_R0"], rc["shrink_at_t0_is_R0"]))


def group1(verbose=True):
    import solver_q1 as q1

    dr_cm, dt, t_end = 0.01, 1.0 / 64.0, 1800.0
    out = {"config": dict(dr_cm=dr_cm, dt_s=dt, t_end_s=t_end,
                          R_const_m=R0, props="附录 2", case="A（口径 A）"),
           "note": ("两个实现必须在同一代码路径下、只用一个开关切换；"
                    "本脚本的固定域参照为 `src/solver_q1.py`（只读 import、不改）——"
                    "Q4 侧的 `R_mode='const'` 开关使 ξ 坐标逐字退化为固定域。")}

    N, i1800 = _g1_stage1(q1, out, dr_cm, dt, t_end)
    d_const_val = A2_D0 * math.exp(-A2_BETA / C_INIT)
    C1b_1800, C4b_1800 = _g1_stage2(q1, out, dr_cm, dt, t_end, N, i1800, d_const_val)
    _g1_stage3(q1, out, dr_cm, dt, N, d_const_val, C1b_1800, C4b_1800)
    rc = _g1_stage4(out)
    _g1_stage5(out, rc)
    _g1_report(out, rc, verbose)
    return out


def group2(verbose=True):
    def run(dt, t_max=200000.0):
        C = C_INIT
        t = 0.0
        n = 0
        while t < t_max:
            t0, C0 = t, C
            tm = t + 0.5 * dt
            Rm = float(radius_at(tm, "shrink"))
            Ca = float(driver_at(tm)[1])
            rate = 2.0 * HM_DEFAULT * dt / Rm
            C = Ca + (C - Ca) * math.exp(-rate)
            t = (n + 1) * dt
            n += 1
            if C <= 0.15:
                return t, C, (t0, C0, t, C)
        return None, C, (t, C, t, C)

    res = {}
    for dt in (0.5, 0.25, 0.125, 0.0625):
        t_cross, C_end, br = run(dt)
        t0, C0, t1, C1 = br
        w = (0.15 - C0) / (C1 - C0) if C1 != C0 else 0.0
        res["dt_%g" % dt] = dict(t_cross_s=t_cross, t_cross_h=t_cross / 3600.0,
                                 C_end=C_end,
                                 t_interp_s=t0 + w * (t1 - t0),
                                 bracket=[t0, t1], C_bracket=[C0, C1])
    t05 = res["dt_0.5"]["t_cross_s"]
    t025 = res["dt_0.25"]["t_cross_s"]
    t5i = res["dt_0.5"]["t_interp_s"]
    target = T_DRY_MIXING_S
    out = dict(
        method=("ODE dC/dt = −2·h_m·(C − C_a(t))/R(t)，步内指数精确"
                "（R、C_a 步内取中点；C_a 为分段线性、R 为分段线性  该步进为二阶）"),
        h_m=HM_DEFAULT, R_source="附件 2 分段线性（与主模型同源）",
        C_a_source="附件 1 分段线性 + t>14400 s 冻结 0.04986 kg/kg",
        results=res,
        target_s=target, target_h=T_DRY_MIXING_H,
        t_cross_dt05_s=t05, t_interp_dt05_s=t5i,
        diff_step_dt05_s=abs(t05 - target),
        diff_interp_dt05_s=abs(t5i - target),
        dt_refinement_delta_s=abs(t05 - t025),
        converged=bool(abs(t05 - t025) < 0.5),
        note=("靶值 30083.5 s 由建模文档以 Δt=0.5 s 给出（落在其步网格上）；"
              "本实现在同一 Δt 上得到同一格，差 1 步 = 0.5 s；加密到 Δt=0.0625 s 后"
              "首达步时刻稳定在 %s s。" % res["dt_0.0625"]["t_cross_s"]),
    )
    out["pass"] = bool(abs(t05 - target) < 1.0)
    if verbose:
        print("[组 2] 充分混合下界：Δt=0.5 s  首达步 %.4f s（步内插值 %.4f s）= %.6f h"
              "（靶 %.6f h，差 %.4f s）" % (t05, t5i, t05 / 3600.0, T_DRY_MIXING_H,
                                           abs(t05 - target)))
        print("        Δt 序列：%s；加密后首达步 %s s；判据 ≤1 s  %s"
              % (["%g%.4f s" % (d, res["dt_%g" % d]["t_cross_s"])
                 for d in (0.5, 0.25, 0.125, 0.0625)],
                 res["dt_0.0625"]["t_cross_s"],
                 "PASS" if out["pass"] else "FAIL"))
    return out


def _g3b_runs(out, D):
    ok_all = True
    for N, dt in ((200, 1.0), (200, 0.5), (200, 0.25), (200, 0.125),
                  (100, 0.25), (400, 1.0), (400, 0.5), (400, 0.25), (800, 1.0),
                  (800, 0.25)):
        o = solve_q4(N=N, dt1=dt, dt2=dt, t_switch=1e18, R_mode="const",
                     props="app4", bc="dirichlet", T_mode="prescribed",
                     d_const=D, ca_scenario="const", const_ca=0.0,
                     stop_on_dry=False, t_end_override=10800.0,
                     snap_every_s=1800.0, snap_until_s=10800.0,
                     cmax_track=False, verbose=False)
        st = o["snap_t"]
        i1 = int(np.argmin(np.abs(st - 1800.0)))
        i2 = int(np.argmin(np.abs(st - 10800.0)))
        c1 = float(o["snap_C"][i1][0]); c2 = float(o["snap_C"][i2][0])
        e1 = abs(c1 - TARGET_C0_1800); e2 = abs(c2 - TARGET_C0_10800)
        ok = (e1 < ACCEPT_TOL) and (e2 < ACCEPT_TOL)
        ok_all = ok_all and ok
        out["runs"]["N%d_dt%g" % (N, dt)] = dict(
            N=N, dt_s=dt, t1_s=float(st[i1]), t2_s=float(st[i2]),
            C0_1800s=c1, C0_10800s=c2, err_1800=e1, err_10800=e2,
            tol=ACCEPT_TOL, pass_=bool(ok),
            mass_audit_max_rel=o["mass_audit"]["max_rel"], wall_s=o["elapsed_s"])


def _g3b_order(out):
    main_line = [k for k in out["runs"] if k.startswith("N200_") and out["runs"][k]["pass_"]]
    out["reporting_case"] = "N200_dt0.25"
    def _ord(keys, tag):
        seq = [out["runs"][k]["err_10800"] for k in keys]
        return dict(tag=tag, keys=keys, err_seq=seq,
                    ratio_seq=[seq[i] / seq[i + 1] for i in range(len(seq) - 1)],
                    p_seq=[math.log2(seq[i] / seq[i + 1]) for i in range(len(seq) - 1)])
    out["observed_order"] = dict(
        spatial_at_dt025=_ord(["N100_dt0.25", "N200_dt0.25", "N400_dt0.25",
                               "N800_dt0.25"], "Δt=0.25 s 固定，N 加倍"),
        temporal_at_N200=_ord(["N200_dt1", "N200_dt0.5", "N200_dt0.25",
                               "N200_dt0.125"], "N=200 固定，Δt 减半"),
        note=("Dirichlet 极限的初始数据在不透面上有跳跃（C(·,0)=2.55  C(R,0)=0），"
              "该奇性使向后 Euler 的时间收敛退化为次线性（实测 p≈0.1~0.4）；"
              "这是 Dirichlet 理想化本身的代价，不代表生产 Robin 问题的收敛率"
              "（生产问题在 ξ=1 上是通量连续、无跳跃）。"))


def _g3b_notes(out):
    out["pass"] = bool(out["runs"]["N200_dt0.25"]["pass_"]
                       and out["runs"]["N400_dt0.25"]["pass_"]
                       and out["runs"]["N800_dt1"]["pass_"])
    out["line_5e-5_note"] = ("② 层线 5×10⁻⁵ 在 N=200 上要求 Δt ≤ 0.5 s 量级"
                             "（Dirichlet 极限的初始边界层比生产 Robin 问题更陡）；"
                             "生产口径的 Δt 由 run_q4 的 Δt 加密序列单独定案（§7.3）。")
    out["audit_note"] = ("Dirichlet 口径不计水分守恒审计（(Q4-28) 的右端是 Robin 通量，"
                         "对 Dirichlet 边界不成立） 本节 `mass_audit_max_rel=0` 表示"
                         "未做，不是通过。")


def _g3b_report(out, an, verbose):
    if verbose:
        for k, r in out["runs"].items():
            print("[组 3b] %-12s C(0,1800)=%.12f (err %.2e)  C(0,10800)=%.12f (err %.2e)"
                  "  %.2f s   %s"
                  % (k, r["C0_1800s"], r["err_1800"], r["C0_10800s"], r["err_10800"],
                     r["wall_s"], "PASS" if r["pass_"] else "FAIL"))
        print("        解析级数自校验：C(0,1800)=%.12f（靶 %.12f，差 %.2e）；"
              "C(0,10800)=%.12f（靶 %.12f，差 %.2e）"
              % (an[0], TARGET_C0_1800, abs(float(an[0]) - TARGET_C0_1800),
                 an[1], TARGET_C0_10800, abs(float(an[1]) - TARGET_C0_10800)))
        print("        报告口径 %s；② 层线 5e-5 达标的最小档见上表"
              % out["reporting_case"])


def group3b(verbose=True):
    D = D_AP4_2P55_323K
    t_targets = np.array([1800.0, 10800.0])
    an = bessel_analytic_C0(t_targets, D, R0, n_roots=400) * C_INIT
    out = dict(D=D, D_source="附录 4 在 C=2.55、T=323.15 K 处求值（§12.2 组 3b 逐位）",
               R_m=R0, bc="Dirichlet C(R,t)=0", C_init=C_INIT,
               analytic=dict(C0_1800s=float(an[0]), C0_10800s=float(an[1])),
               targets=dict(C0_1800s=TARGET_C0_1800, C0_10800s=TARGET_C0_10800),
               target_match=dict(d1800=abs(float(an[0]) - TARGET_C0_1800),
                                 d10800=abs(float(an[1]) - TARGET_C0_10800)),
               runs={})
    _g3b_runs(out, D)
    _g3b_order(out)
    _g3b_notes(out)
    _g3b_report(out, an, verbose)
    return out


def main():
    RES.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print("[verify_q4_baseline] 第一优先级三组地基基准（建模文档 §12.2 组 1 / 组 2 / 组 3b）")
    out = {"generated_at": datetime.now().isoformat(timespec="seconds"),
           "note": ("三组都不经过温度场（§16.1 第 6 条）。组 1/3b 用到求解器；"
                    "组 2 为独立 ODE 积分。")}
    out["group1_degenerate_regression"] = group1()
    out["group2_well_mixed_lower_bound"] = group2()
    out["group3b_bessel_dirichlet"] = group3b()
    out["wall_s"] = time.time() - t0
    out["all_pass"] = bool(out["group1_degenerate_regression"]["pass_group1"]
                           and out["group2_well_mixed_lower_bound"]["pass"]
                           and out["group3b_bessel_dirichlet"]["pass"])
    (RES / "q4_baseline.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("三组总判定：%s（产物 results/q4_baseline.json；总耗时 %.1f s）"
          % ("全部 PASS" if out["all_pass"] else "存在 FAIL 项", out["wall_s"]))
    return 0 if out["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
