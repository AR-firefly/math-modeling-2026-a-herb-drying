from __future__ import annotations
import json
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "Q1"))

from solver_q1 import solve_q1
from independent_check_q1_temp import (
    load_oven_temperature, duhamel_temperature,
)

T_TAB = [100, 300, 600, 900, 1200, 1500, 1800]
R_TAB_CM = [0.0, 0.5, 1.0, 1.5, 2.0]
R_ALL_CM = list(np.arange(0.0, 2.0 + 1e-9, 0.1))
DR_CM = 0.01
T_ACC = 5.0e-5
M_ACC = 5.0e-5
T_END = 1800.0


def sweep_time_steps(i_tab, i_all, an_tab, an_1800, an_win):
    out = {"criterion_C": T_ACC, "dr_cm": DR_CM,
           "note": "验收线按窗口内逐点最大误差核对", "rows": []}
    print(f"{'Δt (s)':>12}{'表1(7×5)最大':>16}  {'1800s 21点最大':>16}{'窗口合计最大':>16} 判定")
    c_win = {}
    dt_list = [1.0, 0.5, 0.25, 0.125, 0.0625, 1.0 / 32, 1.0 / 64, 1.0 / 128]
    for dt in dt_list:
        o = solve_q1(case="A", dr_cm=DR_CM, dt=dt, t_end=T_END, dt_out=1.0)
        num_tab = np.array([[o["T_out"][int(t)][i] for i in i_tab] for t in T_TAB])
        e_tab = np.abs(num_tab - an_tab)
        it, ir = np.unravel_index(np.argmax(e_tab), e_tab.shape)
        e_1800 = float(np.max(np.abs(o["T_out"][int(T_END)][i_all] - an_1800)))
        dev_win = np.abs(o["T_out"][1:, i_all] - an_win)
        e_win_all = float(dev_win.max())
        iw_, jw_ = np.unravel_index(np.argmax(dev_win), dev_win.shape)
        e_tab_max = float(e_tab.max())
        ok = max(e_tab_max, e_1800, e_win_all) < T_ACC
        c_win[dt] = o["C_out"][1:][:, i_all].copy()
        out["rows"].append(dict(dt=dt, err_tab7x5_max=e_tab_max,
                                err_tab7x5_argmax={"t_s": T_TAB[it], "r_cm": R_TAB_CM[ir]},
                                err_1800s_21pt=e_1800, err_window_all_cells=e_win_all,
                                err_window_all_argmax={"t_s": int(iw_) + 1, "r_cm": float(R_ALL_CM[jw_])},
                                meets=T_ACC if ok else None, pass_=bool(ok),
                                wall_s=o["elapsed_s"]))
        print(f"{dt:>12.6f}{e_tab_max:>16.3e}@{T_TAB[it]:<5d}{e_1800:>16.3e}{e_win_all:>16.3e}"
              f"  {'PASS' if ok else '超线'}")
    ok_rows = [r for r in out["rows"] if r["pass_"]]
    out["min_dt_meeting_full_window"] = (max(ok_rows, key=lambda r: r["dt"])["dt"]
                                         if ok_rows else None)
    return out, c_win, dt_list


def moisture_vs_reference(c_win, dt_list, out):
    ref_dt = dt_list[-1]
    ref = c_win[ref_dt]
    t_axis = np.arange(1, 1801, dtype=float)
    m_rows = []
    for dt in dt_list:
        d = np.abs(c_win[dt] - ref)
        iw, jw = np.unravel_index(np.argmax(d), d.shape)
        m_rows.append(dict(dt=dt, max_abs_diff_vs_ref=float(d.max()),
                           argmax={"t_s": float(t_axis[iw]), "r_cm": float(R_ALL_CM[jw])}))
        out["rows"][dt_list.index(dt)]["err_window_moisture_vs_1_128"] = float(d.max())
    return ref_dt, ref, t_axis, m_rows


def moisture_error_stats(c_win, ref, ref_dt, t_axis, m_rows, rich_err, out):
    d_64 = c_win[1.0 / 64] - ref
    d_32 = c_win[1.0 / 32] - c_win[1.0 / 64]
    a_64 = np.abs(d_64)
    err64_p1, err64_p2 = rich_err(float(a_64.max()), 1.0), rich_err(float(a_64.max()), 2.0)
    per_t64 = np.abs(d_64).max(axis=1) * 2.0
    n_over = int((np.abs(d_64) * 2.0 > M_ACC).sum())
    iw, jw = np.unravel_index(np.argmax(a_64), a_64.shape)
    out["moisture_window_error"] = {
        "reference_dt_s": ref_dt,
        "criterion_C_analog": M_ACC,
        "criterion_note": "水分无解析解  以更细步长解为参考的时间步自收敛；判据取 5e-5 kg/kg"
                          "，与温度同口径",
        "scope": "仅时间离散",
        "rows": m_rows,
        "delivered_dt_s": 1.0 / 64,
        "raw_gap_1_64_vs_1_128_max": float(a_64.max()),
        "raw_gap_1_64_vs_1_128_argmax": {"t_s": float(t_axis[iw]), "r_cm": float(R_ALL_CM[jw])},
        "delivered_dt_error_estimate_p1": err64_p1,
        "delivered_dt_error_estimate_p2": err64_p2,
        "delivered_dt_meets_analog_criterion_p1": bool(err64_p1 < M_ACC),
        "delivered_dt_meets_analog_criterion_p2": bool(err64_p2 < M_ACC),
        "n_cells_over_criterion_p1": n_over,
        "n_cells_total": int(a_64.size),
        "per_time_max_p1": {"t_max_s": float(t_axis[int(np.argmax(per_t64))]),
                            "max": float(per_t64.max()),
                            "max_at_1800s": float(per_t64[-1]),
                            "max_in_first_60s": float(per_t64[:60].max())},
        "dt_1_32_vs_1_64_max": float(np.abs(d_32).max()),
        "dt_1_32_error_estimate_p1": rich_err(float(np.abs(d_32).max()), 1.0),
        "order_p_note": "后向 Euler 一阶  取 p=1 为主估计；同时给 p=2 的乐观界",
        "implied_1_128_error_estimate_p1": float(a_64.max() / (2.0 ** 1.0 - 1.0)),
    }
    return a_64, d_32, err64_p1, err64_p2, n_over, iw, jw


def main():
    t0 = time.time()
    t_air, T_oven = load_oven_temperature()
    i_tab = [int(round(rc / DR_CM)) for rc in R_TAB_CM]
    i_all = [int(round(rc / DR_CM)) for rc in R_ALL_CM]
    an_tab = duhamel_temperature(T_TAB, np.array(R_TAB_CM) / 100.0, t_air, T_oven, n_roots=320)
    an_1800 = duhamel_temperature([T_END], np.array(R_ALL_CM) / 100.0, t_air, T_oven, n_roots=320)[0]
    an_win = duhamel_temperature(np.arange(1, 1801, dtype=float), np.array(R_ALL_CM) / 100.0, t_air, T_oven, n_roots=320)

    out, c_win, dt_list = sweep_time_steps(i_tab, i_all, an_tab, an_1800, an_win)

    ref_dt, ref, t_axis, m_rows = moisture_vs_reference(c_win, dt_list, out)

    def rich_err(diff_max, p):
        return float(diff_max * (2.0 ** p) / (2.0 ** p - 1.0))

    a_64, d_32, err64_p1, err64_p2, n_over, iw, jw = moisture_error_stats(c_win, ref, ref_dt, t_axis, m_rows, rich_err, out)
    out["wall_s"] = time.time() - t0
    p = ROOT / "results" / "q1_window_error.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结论：满足「全窗口逐点 < 5e-5 ℃」的最大 Δt = "
          f"{out['min_dt_meeting_full_window']} s（本生产取 1/64 s）")
    print(f"水分场（固定 Δr=0.01 cm，原始节点值）：1/64 vs 1/128 实测最大差 = {a_64.max():.3e} kg/kg "
          f"@t={t_axis[iw]:g}s, r={R_ALL_CM[jw]:g}cm")
    print(f"   Δt=1/64 s 自身误差估计：p=1 {err64_p1:.3e}（判据 {M_ACC:.0e}  "
          f"{'PASS' if err64_p1 < M_ACC else '超线'}）/ p=2 {err64_p2:.3e}"
          f"（{'PASS' if err64_p2 < M_ACC else '超线'}）；超判据单元 {n_over}/{a_64.size}")
    print(f"   1/32 -> 1/64 实测最大差 {np.abs(d_32).max():.3e}（其 p=1 误差估计 "
          f"{rich_err(float(np.abs(d_32).max()), 1.0):.3e}）")
    print(f"总耗时 {out['wall_s']:.1f} s；已写出 {p}")


if __name__ == "__main__":
    sys.exit(main())
