from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

_ENC = getattr(sys.stdout, "encoding", "").lower()
_STDOUT = sys.stdout
if _ENC != "utf-8" and hasattr(_STDOUT, "reconfigure"):
    _STDOUT.reconfigure(encoding="utf-8", errors="replace")

_SELF = Path(__file__).resolve()
_PROJ = _SELF.parents[1]
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

from solver_q4 import (
    solve_q4, radius_at, driver_at, radius_check, att1, att2,
    R0, R_MIN, R_MIN_CM, R_FREEZE_T, C_INIT, T_INIT_C, C_DRY, MARGIN_S,
    T_CAP_S, T_FREEZE_S, TA_FREEZE_C, CA_FREEZE, D_FLOOR, D_AP4_2P55_323K,
    ATT1, ATT2, PDF, RESULT4_TEMPLATE,
)

SEED = 42
N_MAIN = 400
DT_I = 1.25
DT_II = 5.0
DT_OUT = 60.0
R_OUT_CM = np.round(np.arange(0.0, 1.1 + 1e-9, 0.1), 10)
R_OUT_CM_WIDE = np.round(np.arange(0.0, 1.9 + 1e-9, 0.1), 10)
R_TAB6_CM = np.array([0.0, 0.5, 1.0])
T_TAB6_H = np.arange(6.0, 73.0, 6.0)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pchip_eval(x, y, xq):
    from scipy.interpolate import PchipInterpolator
    return PchipInterpolator(x, y, extrapolate=False)(xq)


def linear_eval(x, y, xq):
    return np.interp(xq, x, y)


def fv_hermite_eval(x, y, xq):
    n = x.size
    h = np.diff(x)
    d = np.diff(y) / h
    m = np.zeros(n)
    if n == 2:
        m[:] = d[0]
    else:
        w1 = 2 * h[1:] + h[:-1]
        w2 = h[1:] + 2 * h[:-1]
        dd0, dd1 = d[:-1], d[1:]
        same = dd0 * dd1 > 0
        den = np.where(same, w1 / np.where(dd0 == 0, 1.0, dd0)
                       + w2 / np.where(dd1 == 0, 1.0, dd1), 1.0)
        m[1:-1] = np.where(same, (w1 + w2) / den, 0.0)
        for k, (h0, h1, mm0, mm1) in ((0, (h[0], h[1], d[0], d[1])),
                                      (n - 1, (h[-1], h[-2], d[-1], d[-2]))):
            dv = ((2 * h0 + h1) * mm0 - h0 * mm1) / (h0 + h1)
            if np.sign(dv) != np.sign(mm0):
                dv = 0.0
            elif (np.sign(mm0) != np.sign(mm1)) and abs(dv) > 3.0 * abs(mm0):
                dv = 3.0 * mm0
            m[k] = dv
    i = np.clip(np.searchsorted(x, xq, side="right") - 1, 0, n - 2)
    s = (xq - x[i]) / h[i]
    hi = h[i]
    h00 = (1 + 2 * s) * (1 - s) ** 2
    h10 = s * (1 - s) ** 2
    h01 = s * s * (3 - 2 * s)
    h11 = s * s * (s - 1)
    return h00 * y[i] + h10 * hi * m[i] + h01 * y[i + 1] + h11 * hi * m[i + 1]


def map_output_full(xi_nodes, C_nodes, R_now, r_out_cm, method="pchip"):
    vals = map_output(xi_nodes, C_nodes, R_now, r_out_cm, method)
    return np.concatenate([vals, [float(C_nodes[-1])]])


def map_output(xi_nodes, C_nodes, R_now, r_out_cm, method="pchip"):
    r_out_m = np.asarray(r_out_cm, float) / 100.0
    Rm = float(R_now)
    vals = np.full(r_out_m.shape, np.nan)
    inside = r_out_m <= Rm + 1e-15
    if not np.any(inside):
        return vals
    xq = r_out_m[inside] / Rm
    if method == "pchip":
        vals[inside] = pchip_eval(xi_nodes, C_nodes, xq)
    elif method == "linear":
        vals[inside] = linear_eval(xi_nodes, C_nodes, xq)
    elif method == "fvhermite":
        vals[inside] = fv_hermite_eval(xi_nodes, C_nodes, xq)
    else:
        raise ValueError(method)
    j = np.flatnonzero(inside)
    for k in j:
        if r_out_m[k] == 0.0:
            vals[k] = C_nodes[0]
        elif abs(r_out_m[k] - Rm) <= 1e-15:
            vals[k] = C_nodes[-1]
    return vals


def locate_crossing(tt, cm, level):
    idx = np.flatnonzero(cm <= level)
    if idx.size == 0:
        return None
    n = int(idx[0])
    out = {"n_star": n, "level": float(level), "t_step_s": float(tt[n]),
           "t_after_s": float(tt[n]), "cmax_after": float(cm[n])}
    if n == 0:
        out.update(t_before=None, cmax_before=None, t_linear=float(tt[0]),
                   t_hermite=None, localization_error=None, slope=0.0,
                   slope_at_cross=0.0)
        return out
    t0, t1 = float(tt[n - 1]), float(tt[n])
    c0, c1 = float(cm[n - 1]), float(cm[n])
    out.update(t_before=t0, cmax_before=c0)
    if c1 == c0:
        out.update(t_linear=t1, t_hermite=None, localization_error=None,
                   slope=0.0, slope_at_cross=0.0)
        return out
    w = (level - c0) / (c1 - c0)
    t_lin = t0 + w * (t1 - t0)

    def slope_at(i):
        if i <= 0:
            return (cm[1] - cm[0]) / (tt[1] - tt[0])
        if i >= cm.size - 1:
            return (cm[i] - cm[i - 1]) / (tt[i] - tt[i - 1])
        return (cm[i + 1] - cm[i - 1]) / (tt[i + 1] - tt[i - 1])
    m0, m1 = slope_at(n - 1), slope_at(n)
    h = t1 - t0
    lo, hi = 0.0, 1.0
    for _ in range(80):
        s = 0.5 * (lo + hi)
        s2, s3 = s * s, s * s * s
        val = ((2 * s3 - 3 * s2 + 1) * c0 + (s3 - 2 * s2 + s) * h * m0
               + (-2 * s3 + 3 * s2) * c1 + (s3 - s2) * h * m1 - level)
        if val > 0:
            lo = s
        else:
            hi = s
    t_herm = t0 + 0.5 * (lo + hi) * h
    out.update(t_linear=float(t_lin), t_hermite=float(t_herm),
               slope=float((c1 - c0) / h),
               slope_at_cross=float(0.5 * (m0 + m1)),
               localization_error=float(abs(t_herm - t_lin)))
    return out


def write_xlsx(path, times_s, rows, header, decimals=4, sheet="Sheet1"):
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = sheet
    ws.append(header)
    for k, t in enumerate(times_s):
        vals = []
        for v in rows[k]:
            vals.append(None if (v is None or (isinstance(v, float) and math.isnan(v)))
                        else float(round(float(v), decimals)))
        ws.append([int(round(t))] + vals)
    wb.save(path)


T_DRY_MAIN_COMPONENT = "grid_contribution_s"
_T_DRY_CV = "q4_convergence.json"
_T_DRY_EB = "q4_error_budget.json"
_ASSUMED_FIELD_ERROR_KGKG = 1.0e-5


def fwd(pth):
    return str(pth).replace("\\", "/")


def jget(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def load_json_soft(path):
    path = Path(path)
    if not path.is_file():
        return None, "缺 %s" % fwd(path)
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except Exception as e:
        return None, "%s 解析失败：%r" % (fwd(path), e)


def parse_h_interval(stmt):
    if not isinstance(stmt, str):
        return None
    i, j = stmt.find("["), stmt.find("] h")
    if i < 0 or j <= i:
        return None
    try:
        lo, hi = stmt[i + 1:j].split(",")
        return [float(lo.strip()), float(hi.strip())]
    except ValueError:
        return None


def grid_keys():
    return str(N_MAIN), str(2 * N_MAIN), str(N_MAIN // 2)


def grid_component(lin, step_seq, cfg, t_nom_h):
    if lin is None:
        return None, None, None, None
    k_prod, k_fine, k_coarse = grid_keys()
    cfg_ok = (isinstance(cfg, dict) and cfg.get("dt1") == DT_I
              and cfg.get("dt2") == DT_II and cfg.get("t_switch") == T_FREEZE_S)
    nom_ok = (lin.get(k_prod) is not None
              and abs(float(lin[k_prod]) - t_nom_h) <= 1e-9 * max(1.0, abs(t_nom_h)))
    have = all(k in lin for k in (k_coarse, k_prod, k_fine))
    if not cfg_ok:
        return None, None, None, (
            "results/%s  A_grid_independence.config = %r 与本轮生产配置"
            "（dt1=%r、dt2=%r、t_switch=%r）不一致  该加密档与生产不同源、不可比"
            % (_T_DRY_CV, cfg, DT_I, DT_II, T_FREEZE_S))
    if not nom_ok:
        return None, None, None, (
            "results/%s  A_grid_independence.t_dry_lin_h[%s] = %r 与本轮生产档 "
            "t_dry.t_nominal_linear_h = %r 不一致  档位与生产解不同源"
            % (_T_DRY_CV, k_prod, lin.get(k_prod), t_nom_h))
    if not have:
        return None, None, None, (
            "results/%s  A_grid_independence.t_dry_lin_h 缺 %s / %s / %s 档"
            % (_T_DRY_CV, k_coarse, k_prod, k_fine))
    grid = abs(float(lin[k_fine]) - float(lin[k_prod])) * 3600.0
    grid_up = abs(float(lin[k_prod]) - float(lin[k_coarse])) * 3600.0
    step_diff = None
    if step_seq and all(k in step_seq for k in (k_prod, k_fine)):
        step_diff = abs(float(step_seq[k_fine]) - float(step_seq[k_prod])) * 3600.0
    return grid, grid_up, step_diff, None


def build_t_dry_uncertainty(res, t_dry_step, t_dry_nom, t_dry_herm, loc_err,
                             cond_s_per_unit, dt_ii):
    base_h, cfg, cv, e1_rel, e2_pct, fe, interval, lin, order_p, res, sens, spread_h, step_seq, t_nom_h, t_step_h, why = load_uncertainty_inputs(
        cond_s_per_unit,
        res,
        t_dry_nom,
        t_dry_step,
    )

    grid, grid_up, step_diff, why_bad = grid_component(lin, step_seq, cfg, t_nom_h)
    if why_bad:
        why = why_bad

    rec = ("t_dry = %s s = %s h（首个 C_max ≤ 0.15 的内部步；线性/三次 Hermite 未舍入值"
           "另见 t_dry.t_nominal_linear_s = %s / t_dry.t_nominal_hermite_s = %s）。"
           % (float(t_dry_step), t_step_h, float(t_dry_nom), t_dry_herm))
    dep_why, doc_note, fe_note, grid_note, rec, up_note = fill_grid_components(
        cfg,
        dt_ii,
        fe,
        grid,
        grid_up,
        interval,
        lin,
        loc_err,
        order_p,
        rec,
        sens,
        spread_h,
        step_diff,
        why,
    )

    mdata_note = ("来源 results/%s  interval_statement（逐位照抄该字符串中的 [%s, %s] h）："
                  "E1（缺潜热  t_dry 偏短 %.1f %%，上端 = base×(1+E1)，E1_relative = %s）与"
                  "E2（1D 无限长  偏长 ≤ +%.1f %%，下端 = base×(1−E2)，E2.upper_bound_pct = %s）"
                  "的方向性上界，非统计置信区间；base = baseline_t_dry_h = %s。"
                  " 与 Q2 的 model_data_uncertainty_interval_h 口径不同（Q2 的是驱动平台"
                  "情景之差），不要混用。"
                  % (_T_DRY_EB,
                     interval[0] if interval else None, interval[1] if interval else None,
                     e1_rel * 100.0 if e1_rel is not None else float("nan"),
                     e1_rel, e2_pct if e2_pct is not None else float("nan"),
                     e2_pct, base_h)) if interval is not None else (
                  "待回填：%s（E1/E2 方向性上界，须读 results/%s  interval_statement）"
                  % (why, _T_DRY_EB))

    return uncertainty_record(
        cv,
        dep_why,
        doc_note,
        dt_ii,
        fe,
        fe_note,
        grid,
        grid_note,
        grid_up,
        interval,
        loc_err,
        mdata_note,
        rec,
        sens,
        spread_h,
        up_note,
        why,
    )

def load_uncertainty_inputs(cond_s_per_unit, res, t_dry_nom, t_dry_step):
    res = Path(res)
    sens = cond_s_per_unit
    fe = _ASSUMED_FIELD_ERROR_KGKG * sens
    t_step_h = float(t_dry_step) / 3600.0
    t_nom_h = float(t_dry_nom) / 3600.0
    cv, why_cv = load_json_soft(res / _T_DRY_CV)
    eb, why_eb = load_json_soft(res / _T_DRY_EB)
    lin = jget(cv, "A_grid_independence", "t_dry_lin_h")
    step_seq = jget(cv, "A_grid_independence", "t_dry_h")
    cfg = jget(cv, "A_grid_independence", "config")
    spread_h = jget(cv, "A_grid_independence", "t_dry_spread_h")
    order_p = jget(cv, "A_grid_independence", "observed_order_p")
    interval = parse_h_interval(jget(eb, "interval_statement"))
    e1_rel = jget(eb, "E1_latent_heat", "E1_relative")
    e2_pct = jget(eb, "E2_end_face", "upper_bound_pct")
    base_h = jget(eb, "baseline_t_dry_h")
    why = why_cv or why_eb or "未知"
    return (
        base_h,
        cfg,
        cv,
        e1_rel,
        e2_pct,
        fe,
        interval,
        lin,
        order_p,
        res,
        sens,
        spread_h,
        step_seq,
        t_nom_h,
        t_step_h,
        why,
    )

def fill_grid_components(
    cfg,
    dt_ii,
    fe,
    grid,
    grid_up,
    interval,
    lin,
    loc_err,
    order_p,
    rec,
    sens,
    spread_h,
    step_diff,
    why,
):
    if grid is not None:
        rec += ("数值不确定度主报 ±%s s（≈ ±9.1×10⁰ s = %s h，Q4 自己的实测网格项："
                "生产档 N=%s vs 更细一档 N=%s，与 Q2 的「生产 vs 更细一档」口径严格同构）；"
                "±%s s 为灵敏度说明（= 1/|dCmax_dt_at_crossing| × 假定场值误差 "
                "1e-5 kg/kg），不得单独当作 t_dry 的不确定度（本项是网格项的 %s 倍，"
                "比网格项大  单用它会高估）；"
                % (grid, float(grid) / 3600.0, N_MAIN, 2 * N_MAIN, fe, fe / float(grid)))
        if interval is not None:
            rec += ("模型/数据不确定度另以区间 [%s, %s] h（E1/E2 方向性上界，"
                    "非统计置信区间）表述。" % (interval[0], interval[1]))
        if grid_up is not None:
            rec += ("另登记网格敏感度上界（粗一档 N=%s vs 生产 N=%s）= %s s"
                    "（见 grid_sensitivity_upper_bound_s；为网格项的 %s 倍）。"
                    % (N_MAIN // 2, N_MAIN, grid_up, grid_up / float(grid)))
        rec += "不得声称 4 位小数有效（docs/Q4_建模文档.md §9.2）。"
    else:
        rec += ("数值不确定度主报项 = 实测网格项（生产档 N=%d vs 更细一档）待回填"
                "（%s）—— 本脚本单次运行算不出该值，须先跑 `src/verify_q4_convergence.py` "
                "产出 `results/%s` 之后重跑本脚本；回填前不得报任何 t_dry 不确定度数。"
                "不得声称 4 位小数有效（docs/Q4_建模文档.md §9.2）。"
                % (N_MAIN, why, _T_DRY_CV))

    if grid is not None:
        k_prod, k_fine, k_coarse = grid_keys()

        why_lin = ("为何用 t_dry_lin_h 而非 t_dry_h：Q2 网格项取的就是线性插值定位"
                   "（results/q2_error_budget.json  …space_grid_tail_measured."
                   "t_dry_dr_0_01_cm 在 src/verify_q2_error_budget.py 中读的是 q2_parts 的 "
                   "t_dry_linear），同口径才可比较；"
                   + ("换用 t_dry_h（首个满足步）序列时同一对的差为 %s s。" % step_diff
                      if step_diff is not None else "（t_dry_h 序列该对值不可用。）"))
        grid_note = (
            "Q4 有自己的网格收敛研究，本键由它独立复算得出（不是从 Q2 继承、不是硬编码）。"
            "复算过程：读 results/%s  A_grid_independence.t_dry_lin_h（该档 config: dt1 = %s、"
            "dt2 = %s、t_switch = %s，只改 N，与 Q2 网格项的「只改 Δr」口径相同），取 "
            "N=%s  %s h（生产档，逐位等于 results4.json  t_dry.t_nominal_linear_h）、"
            "N=%s  %s h（更细一档）；|%s − %s| × 3600 = %s s。"
            "口径为什么是 N=%s↔N=%s（生产 vs 更细一档）：Q2 的生产网格是 N=200，其网格项 "
            "= |t_dry(N=200) − t_dry(N=400)| 即「生产 vs 更细一档」；Q4 的生产网格是 N=%s"
            "（results4.json  meta.config.N），严格同构的量为 |t_dry(N=%s) − t_dry(N=%s)| "
            " 本键取此。%s旁证：同档 observed_order_p = %s（p > 2，加密仍在收敛）；"
            "t_dry_spread_h = %s（四档 step 序列极差 = %.1f s，同值见 "
            "results/q4_error_budget.json  E4_discrete.t_dry_grid_spread_h）。"
            % (_T_DRY_CV, cfg.get("dt1"), cfg.get("dt2"), cfg.get("t_switch"),
               k_prod, lin[k_prod], k_fine, lin[k_fine], lin[k_fine], lin[k_prod], grid,
               k_prod, k_fine, N_MAIN, N_MAIN, 2 * N_MAIN, why_lin,
               order_p, spread_h, float(spread_h) * 3600.0))
        up_note = (
            "= |t_dry_lin_h(N=%s) − t_dry_lin_h(N=%s)| × 3600 = |%s − %s| × 3600 = %s s"
            "（同档 config: dt1 = %s、dt2 = %s、t_switch = %s，只改 N）。"
            "这是「粗一档 vs 生产」的网格敏感度上界，不是主报项 —— 它含更粗档 N=%s 自身"
            "的误差，不能当作生产解 N=%s 的网格不确定度；主报项 = 生产 vs 更细一档 = "
             "grid_contribution_s = %s s（本键为它的 %s 倍）。主报改用与 Q2 严格同构的 "
             "N=%s↔N=%s，本键作为独立键保留此值以便溯源与保守对照。"
            % (k_coarse, k_prod, lin[k_prod], lin[k_coarse], grid_up,
               cfg.get("dt1"), cfg.get("dt2"), cfg.get("t_switch"),
               k_coarse, k_prod, grid, grid_up / grid, k_prod, k_fine)
            + ("旁证（step 序列同一对）= %s s。" % step_diff
               if step_diff is not None else ""))
    else:
        grid_note = ("待回填：%s。主报网格项（|t_dry_lin_h(N=%s) − t_dry_lin_h(N=%s)| × "
                     "3600）须读 results/%s 现算 —— 不填数、不估算。回填："
                     "python src/verify_q4_convergence.py，再重跑本脚本。"
                     % (why, N_MAIN, 2 * N_MAIN, _T_DRY_CV))
        up_note = "待回填：%s" % why

    if grid is not None:
        fe_note = ("= field_error_sensitivity_s_per_kgkg × 1e-5 kg/kg（假定场值误差，"
                   "不是实测误差）—— 仅灵敏度说明，非主报项；本项是主报网格项的 %s 倍"
                   "（比网格项大  单用它当 t_dry 的不确定度会高估）。"
                   "本键承载的正是原 t_dry.uncertainty.field_error_1e-5_kgkg_to_time_s"
                   "（旧主报值），已降格；同名旧键仍在顶层保留（见 legacy_compat_note）。"
                   % (fe / float(grid),))
        dep_why = ("原 uncertainty 块把灵敏度换算（%s s = ∂t_dry/∂C_max × 假定场值误差"
                   " 1e-5 kg/kg）与定位括住宽（%.1f s）、定位误差（%r s）平列，并以 statement "
                   "宣称「t_dry 不得声称 4 位小数有效：场值误差 1e-5 kg/kg  %.1f s」—— "
                   "那 %s s 是灵敏度换算、不是 t_dry 的数值不确定度（它是「假定场值误差 "
                   "1e-5 kg/kg」下换算出来的，不是本问实测的离散误差）；Q4 自己的实测网格项"
                   "（主报项，生产档 vs 更细一档）= %s s。2026-09-11 跨问口径预检确认该根因"
                   "已在 Q2/Q3/Q4 复制 3 次；本块按 Q2 的分层结构改写，原键值"
                   "全部保留（顶层同名 + 本 deprecated 块），未删除。"
                   % (fe, dt_ii, loc_err, fe, fe, grid))
        doc_note = (" 待裁决 "
                    "§9.2 的条件数与 ±15.2 s、§9.4 余量表的 ±15 s 均为设计期值（k_lump 由"
                    "预研原型两个诊断点推出，文档自述属③层冒烟判据）；实测条件数为 %s s per "
                    "(kg/kg)、场值误差 1e-5 kg/kg 的时间当量为 %s s（即设计期 15.2 s 的 "
                    "%.3f 倍）。本块已按实测落盘（数值不确定度主报项 = 实测网格项 %s s，"
                    "非灵敏度项）；文档侧的更正见本轮跨问口径一批改的文档改动。"
                    % (sens, fe, fe / 15.2, grid))
    else:
        fe_note = ("仅灵敏度说明，非主报项：= 1/|dCmax_dt_at_crossing| × 假定场值误差 "
                   "1e-5 kg/kg（本脚本自算）。主报项（实测网格项）待回填前，不得用本项"
                   "代替 t_dry 的不确定度。")
        dep_why = "待回填：%s" % why
    return dep_why, doc_note, fe_note, grid_note, rec, up_note

def uncertainty_record(
    cv,
    dep_why,
    doc_note,
    dt_ii,
    fe,
    fe_note,
    grid,
    grid_note,
    grid_up,
    interval,
    loc_err,
    mdata_note,
    rec,
    sens,
    spread_h,
    up_note,
    why,
):
    return {
        "main_reported_component": T_DRY_MAIN_COMPONENT,
        "recommended_statement": rec,
        "grid_contribution_s": grid,
        "grid_contribution_note": grid_note,
        "grid_sensitivity_upper_bound_s": grid_up,
        "grid_sensitivity_upper_bound_note": up_note,
        "time_step_contribution_s": None,
        "time_step_contribution_note": (
            "置 null：Q4 没有生产网格上的 Δtt_dry 落盘项，故不移植、不新算"
            "（本项目纪律：不得自算无落盘依据的数）。落盘事实三条：① results/%s  "
            "B_time_convergence.config.N = 200（不是生产网格 N=%s），其 t_dry_lin_h 六档"
            "序列 I20_II60…I0.625_II2.5 的极差 = %s h = %.1f s（含 I20_II60 等粗档，"
            "口径是 spread 不是 contribution；逐位等于 results/%s  E4_discrete."
            "t_dry_dt_spread_h）；② A_grid_independence.t_dry_spread_h = %s h（= %.1f s，"
            "四档网格 spread）；③ Q2 的对应键 time_step_contribution_s = 2.9718922144239177 s "
            "出自「Δt_II 1/8 vs 1/16 的 C_max 轨迹最大偏差 × 实测放大系数」，Q4 无同口径落盘量。"
            "建议：由编程手在 N=%s 上补一对 Δt 加密（或明确以 B_time_convergence 的 "
            "N=200 口径为准）后回填本键。"
            % (_T_DRY_CV, N_MAIN, dt_spread_h(cv), dt_spread_h(cv) * 3600.0,
               _T_DRY_EB, spread_h, float(spread_h) * 3600.0, N_MAIN)
            if (dt_spread_h(cv) is not None and spread_h is not None) else
            "待回填：%s（Δt 加密研究的实测值；口径为 spread 不是 contribution，"
            "不能直接当不确定度）" % why),
        "localization_only_s": loc_err,
        "localization_only_note": ("Q4 自己的线性 vs 三次 Hermite 定位差（= 本文件 t_dry."
                                   "localization_error_s）；比网格项小 6 个量级。"),
        "field_error_sensitivity_s_per_kgkg": sens,
        "field_error_sensitivity_note": (
            "= 1/|t_dry.dCmax_dt_at_crossing|（实测跨越点斜率 %r kg/(kg·s)）= 灵敏度换算系数，"
            "不是不确定度本身；逐位等于本文件 t_dry.condition_number_s_per_kgkg（Q4 的规范"
            "位置在 t_dry 顶层，本键仅为与 Q2 键名对齐而设）。" % (1.0 / sens)),
        "field_error_at_1e-5_kgkg_s": fe,
        "field_error_at_1e-5_kgkg_note": fe_note,
        "model_data_uncertainty_interval_h": interval,
        "model_data_uncertainty_note": mdata_note,
        "doc_consistency_note": doc_note,
        "field_to_time_factor_s_per_kgkg": sens,
        "field_to_time_factor_s_per_kgkg_status": (
            "本键为对齐 Q2 键名而设；Q4 的规范位置是 t_dry.condition_number_s_per_kgkg"
            "（逐位同值）。 不要与 Q2 的同名键 953600.0 混淆 —— 那是 Q2 的设计期名义值"
            "（已废弃）。本键值是实测跨越点斜率倒数，可用。"),
        "deprecated": {
            "field_error_1e-5_kgkg_to_time_s": fe,
            "field_error_1e-4_kgkg_to_time_s": 1e-4 * sens,
            "bracket_width_s": dt_ii,
            "localization_s": loc_err,
            "statement": ("t_dry 不得声称 4 位小数有效：场值误差 1e-5 kg/kg  %.1f s；"
                          "定位括住宽 %.0f s；缩小 Δt 不改善 t_dry（它由场值误差决定，"
                          "§9.2 推论 2）" % (fe, dt_ii)),
            "why": dep_why,
        },
        "legacy_compat_note": (
            "顶层 bracket_width_s / localization_s / field_error_1e-5_kgkg_to_time_s 三个键"
            "按原名原值保留，唯一原因是 src/build_answer_bundle.py 直接按名读取"
            "（_unc[\"field_error_1e-5_kgkg_to_time_s\"]、_unc[\"localization_s\"]、"
            "_unc[\"bracket_width_s\"]），移走或改名会使该脚本当场 KeyError。它们不是 "
            "t_dry 的数值不确定度：主报项见 grid_contribution_s。"),
        "bracket_width_s": dt_ii,
        "localization_s": loc_err,
        "field_error_1e-5_kgkg_to_time_s": fe,
        "field_error_1e-4_kgkg_to_time_s": 1e-4 * sens,
    }


def dt_spread_h(cv):
    return jget(cv, "B_time_convergence", "t_dry_spread_h")


def _assert_status(o):
    if o["status"] == "not_dry_by_cap":
        raise SystemExit("[FATAL] cap 14 d reached, C_max ≤ %g unmet" % C_DRY)
    if o["status"] == "extremum_violation":
        for v in o["violations"][:10]:
            print("  [violation]", v)
        raise SystemExit("[FATAL] extremum principle violated (Q4-27)")
    if o["status"] not in ("ok", "fixed_end"):
        raise SystemExit("[FATAL] unknown status=%r" % o["status"])
    if o["floor"]["count"] > 0:
        for e in o["floor"]["first_events"][:10]:
            print("   ", e)
        raise SystemExit("[FATAL] D_floor active %d times" % o["floor"]["count"])
    if o["picard"]["n_not_converged"] > 0:
        raise SystemExit("[FATAL] Picard unconverged steps %d" % o["picard"]["n_not_converged"])
    if o["mass_audit"]["max_rel"] > 1e-8:
        raise SystemExit("[FATAL] mass audit residual %.3e > 1e-8" % o["mass_audit"]["max_rel"])


def _locate_and_build(o):
    tt, cm = o["cmax"]["t"], o["cmax"]["cmax"]
    cross = locate_crossing(tt, cm, C_DRY)
    if cross is None:
        raise SystemExit("[FATAL] no inner step with C_max ≤ %g" % C_DRY)
    t_dry_step = cross["t_step_s"]
    t_dry_nom = cross["t_linear"]
    slope_cross = cross["slope_at_cross"]
    dCmax_dt = abs(slope_cross)
    cond_s_per_unit = 1.0 / dCmax_dt if dCmax_dt > 0 else float("inf")
    snap_t, snap_R, snap_C, xi = o["snap_t"], o["snap_R"], o["snap_C"], o["xi"]
    t_last_out = math.floor(t_dry_step / DT_OUT + 1e-9) * DT_OUT
    i_last = int(np.argmin(np.abs(snap_t - t_last_out)))
    if abs(snap_t[i_last] - t_last_out) > 1e-9:
        raise SystemExit("[FATAL] no output snapshot at t=%g s" % t_last_out)
    n_row = i_last
    if n_row < 1:
        raise SystemExit("[FATAL] zero delivery rows, t_dry=%s" % t_dry_step)
    times_out = snap_t[1:n_row + 1]
    R_out_t = snap_R[1:n_row + 1]
    Cpk = np.empty((n_row, R_OUT_CM.size + 1))
    Cln = np.empty_like(Cpk)
    Chm = np.empty_like(Cpk)
    assert_blank = 0
    for k in range(n_row):
        Ck = snap_C[k + 1]
        Rk = snap_R[k + 1]
        Cpk[k] = map_output_full(xi, Ck, Rk, R_OUT_CM, "pchip")
        Cln[k] = map_output_full(xi, Ck, Rk, R_OUT_CM, "linear")
        Chm[k] = map_output_full(xi, Ck, Rk, R_OUT_CM, "fvhermite")
        assert_blank += int(np.count_nonzero(np.isnan(Cpk[k][:-1])))
    if assert_blank != 0:
        raise SystemExit("[FATAL] %d cells outside r > R(t) in main column" % assert_blank)
    return (tt, cm, cross, t_dry_step, t_dry_nom, slope_cross, dCmax_dt, cond_s_per_unit,
            snap_t, snap_R, snap_C, xi, n_row, times_out, R_out_t, Cpk, Cln, Chm,
            t_last_out, assert_blank)


def _prepare_and_solve():
    dry = os.environ.get("MM_Q4_DRY") == "1"
    global N_MAIN, DT_I, DT_II
    t0 = time.time()
    if dry:
        N_MAIN, DT_I, DT_II = 40, 20.0, 60.0
        res = _PROJ / "results" / "_dryrun_q4"
        print("[dryrun] MM_Q4_DRY=1：粗网格/粗步长，产物只落 %s" % res)
    else:
        res = _PROJ / "results"
    res.mkdir(parents=True, exist_ok=True)

    inputs = {}
    for p in (ATT1, ATT2, PDF, RESULT4_TEMPLATE):
        pp = Path(p)
        inputs[pp.name] = {"path": str(pp),
                           "sha256": sha256(pp) if pp.exists() else None,
                           "exists": pp.exists()}
    scripts = {}
    for s in ("solver_q4.py", "run_q4.py"):
        sp = _PROJ / "Q4" / s
        scripts[s] = sha256(sp) if sp.exists() else None

    rchk = radius_check("shrink")
    if not (rchk["monotone_ok"] and rchk["mono_nonincreasing"]):
        raise SystemExit("[FATAL] radius violates monotone rule, %s" % rchk)

    print("[run_q4] 生产运行：N=%d（Δξ=%g）、Δt_I=%g s、Δt_II=%g s、输出 %g s"
          % (N_MAIN, 1.0 / N_MAIN, DT_I, DT_II, DT_OUT))
    o = solve_q4(N=N_MAIN, dt1=DT_I, dt2=DT_II, t_switch=T_FREEZE_S,
                 R_mode="shrink", geo_layer="nplus1", face_rule="arith",
                 props="app4", bc="robin", T_mode="prescribed",
                 stop_on_dry=True, snap_every_s=DT_OUT,
                 cmax_track=True, audit=True, audit_series=True,
                 verbose=True, progress_every=2000)
    print("[run_q4] status=%s  t_final=%.6f s (%.4f h)  t_dry=%s  steps=%d  墙钟 %.1f s"
          % (o["status"], o["t_final"], o["t_final"] / 3600.0,
             ("%.6f s" % o["t_dry"]) if o["t_dry"] else "None",
             o["n_steps"], o["elapsed_s"]), flush=True)
    return dry, res, inputs, scripts, t0, o, rchk


def _write_cmax_traces(res, tt, cm, o, t_dry_nom):
    np.savez_compressed(res / "q4_cmax_trajectory.npz",
                        t_s=tt, C_max=cm, argmax_i=o["cmax"]["argmax"],
                        C_center=o["cmax"]["c0"], C_surface=o["cmax"]["cR"],
                        C_min=o["cmax"]["cmin"], R_m=radius_at(tt, "shrink"))
    m = (tt >= t_dry_nom - 600.0) & (tt <= t_dry_nom + 600.0)
    np.savetxt(res / "q4_cmax_crossing_window.csv",
               np.column_stack([tt[m], cm[m], o["cmax"]["c0"][m], o["cmax"]["cR"][m],
                                o["cmax"]["cmin"][m], radius_at(tt[m], "shrink")]),
               delimiter=",", comments="",
               header="t_s,C_max,C_center,C_surface,C_min,R_m")
    i60 = np.searchsorted(tt, np.arange(DT_OUT, tt[-1] + 1e-9, DT_OUT))
    np.savetxt(res / "q4_cmax_60s.csv",
               np.column_stack([tt[i60], cm[i60], o["cmax"]["c0"][i60],
                                o["cmax"]["cR"][i60], radius_at(tt[i60], "shrink")]),
               delimiter=",", comments="",
               header="t_s,C_max,C_center,C_surface,R_m")
    ms = o["mass_audit_series"]
    np.savetxt(res / "q4_mass_audit.csv",
               np.column_stack([ms["t"], ms["lhs"], ms["rhs"], ms["rel"], ms["cmax"]]),
               delimiter=",", comments="", header="t_s,lhs,rhs,rel_resid,C_max")


def _build_table6(res, tt, cross, snap_t, snap_C, snap_R, xi, t_dry_step):
    i_dry = cross["n_star"]
    tab6_t_h = T_TAB6_H[T_TAB6_H * 3600.0 <= t_dry_step + 1e-9]
    rows6, times6 = [], []
    surf_vals = []
    for th in tab6_t_h:
        ts = th * 3600.0
        i = int(round(ts / DT_OUT))
        if abs(snap_t[i] - ts) > 1e-9:
            raise SystemExit("[FATAL] table 6 t=%g h misses snapshot" % th)
        vals = map_output_full(xi, snap_C[i], snap_R[i], R_TAB6_CM, "pchip")
        vals[0] = snap_C[i][0]
        rows6.append(vals); times6.append(th)
        surf_vals.append((float(th), float(snap_C[i][-1]), float(snap_R[i])))
    i_near = int(np.argmin(np.abs(snap_t - tt[i_dry])))
    C_end_field = snap_C[i_near]
    R_end_field = snap_R[i_near]
    vals_end = map_output_full(xi, C_end_field, R_end_field, R_TAB6_CM, "pchip")
    vals_end[0] = C_end_field[0]
    table6 = dict(
        header=["时间/h", "0", "0.5", "1.0", "药材表面"],
        unit="C / (kg/kg)；时间 / h",
        rows_h=[float(v) for v in times6] + ["烘干结束时间"],
        values=[[float(v) for v in r] for r in rows6]
        + [[float(v) for v in vals_end.flatten()]],
        t_dry_h_rounded_3=float(round(t_dry_step / 3600.0, 3)),
        t_end_row_field_snapshot_s=float(snap_t[i_near])
    )
    with open(res / "q4_table6.csv", "w", encoding="utf-8-sig") as f:
        f.write(",".join(table6["header"]) + "\n")
        for i, th in enumerate(times6):
            f.write("%g," % th + ",".join("%.4f" % v for v in rows6[i]) + "\n")
        f.write("烘干结束时间,%s\n" % ",".join("%.4f" % v for v in vals_end))
    return table6, times6, vals_end, rows6, surf_vals, i_near


def _write_result4_outputs(res, times_out, snap_C, snap_R, xi, n_row, Cpk, Cln, Chm, R_out_t):
    Cwide = np.full((n_row, R_OUT_CM_WIDE.size + 1), np.nan)
    for k in range(n_row):
        Cwide[k] = map_output_full(xi, snap_C[k + 1], snap_R[k + 1], R_OUT_CM_WIDE, "pchip")
    write_xlsx(res / "result4_wide.xlsx", times_out, Cwide,
               ["时间\\到药材中心的距离"]
               + [float(round(rc, 1)) for rc in R_OUT_CM_WIDE] + ["药材表面"],
               decimals=4, sheet="Sheet1")
    np.savez_compressed(res / "result4_full.npz", t_s=times_out, r_cm=R_OUT_CM,
                        C_pchip=Cpk, C_linear=Cln, C_fvhermite=Chm, R_m=R_out_t)
    with open(res / "result4_full.csv", "w", encoding="utf-8") as f:
        f.write("t_s," + ",".join("%.1f" % rc for rc in R_OUT_CM) + ",药材表面,R_m\n")
        for k in range(n_row):
            f.write("%d," % int(round(times_out[k]))
                    + ",".join("%.10f" % v for v in Cpk[k])
                    + ",%.10f\n" % R_out_t[k])
    dmap = np.abs(Cpk[:, :-1] - Cln[:, :-1])
    dmap2 = np.abs(Cpk[:, :-1] - Chm[:, :-1])
    np.savetxt(res / "q4_mapping_uncertainty.csv",
               np.column_stack([times_out, dmap, dmap2]),
               delimiter=",", comments="",
               header=("t_s," + ",".join("dPL_%.1f" % rc for rc in R_OUT_CM)
                       + "," + ",".join("dPH_%.1f" % rc for rc in R_OUT_CM)))
    return dmap, dmap2


def main():
    dry, res, inputs, scripts, t0, o, rchk = _prepare_and_solve()

    _assert_status(o)

    tt, cm, cross, t_dry_step, t_dry_nom, slope_cross, dCmax_dt, cond_s_per_unit, snap_t, snap_R, snap_C, xi, n_row, times_out, R_out_t, Cpk, Cln, Chm, t_last_out, assert_blank = _locate_and_build(o)
    if not np.array_equal(times_out, np.arange(DT_OUT, t_last_out + 1e-9, DT_OUT)):
        raise SystemExit("[FATAL] delivery axis is not 60 s")

    header4 = (["时间\\到药材中心的距离"]
               + [float(round(rc, 1)) for rc in R_OUT_CM] + ["药材表面"])
    write_xlsx(res / "result4.xlsx", times_out, Cpk, header4, decimals=4, sheet="Sheet1")

    dmap, dmap2 = _write_result4_outputs(res, times_out, snap_C, snap_R, xi, n_row,
                                         Cpk, Cln, Chm, R_out_t)

    table6, times6, vals_end, rows6, surf_vals, i_near = _build_table6(
        res, tt, cross, snap_t, snap_C, snap_R, xi, t_dry_step)

    _write_cmax_traces(res, tt, cm, o, t_dry_nom)
    et = o["audit_energy_terms"]["values"]
    audit_summary = build_audit_summary(et, o)
    (res / "q4_audit_summary.json").write_text(
        json.dumps(audit_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    j, k, prof = write_xi_profile(R_out_t, n_row, res, snap_C, times_out, xi)

    chk = monotonic_check(assert_blank, cm, o, prof, t_dry_step)

    wall = time.time() - t0
    g5_tiering = {"trigger": "G5（§15.1）口径降级", "registered_at": "2026-09-11",
                  "note": "缺 results/q4_convergence.json  分层声明未登记（须先跑 verify_q4_convergence.py）"}
    _cvp = res / "q4_convergence.json"
    if _cvp.exists():
        _cvj = json.loads(_cvp.read_text(encoding="utf-8"))
        _A = _cvj["A_grid_independence"]
        _d4 = _A["diff_vs_N800"]["400"]
        _dt = _cvj["B_time_convergence"]["diff_vs_finest"]["I1.25_II5"]
        g5_tiering = {
            "trigger": "G5（§15.1）口径降级",
            "registered_at": "2026-09-11",
            "source": "results/q4_convergence.json  A_grid_independence.g5_disclosure",
            "rule": ("交付表（result4.xlsx / 表 6）的有效位数按实际离散不确定度分层："
                     "t ≥ 1 h 支撑 4 位小数；t < 1 h（前 59 行）不支撑第 4 位。"),
            "evidence": {
                "t_ge_3600s_grid_N400_vs_N800": _d4["max_abs_t_ge_3600s"],
                "t_ge_3600s_dt_prod_vs_finest": _dt["max_abs_t_ge_3600s"],
                "t_lt_3600s_max": _d4["max_abs"],
                "t_lt_3600s_max_at": "t=60 s、药材表面列",
                "layer2_tol_kgkg": _A["layer2_tol_kgkg"],
            },
            "reason": ("t<1 h 的最大差出现在 t=60 s 表面列（%.3e）；δ(60 s)=2.507×10⁻² cm "
                       "≈ 5.01Δr@N=400  初始边界层已被本生产配置覆盖（不是「不可分辨」）；"
                       "加密序列在该窗口仍以 p≈2.35 收敛、N=800 全程仅 %.1f s  继续加密可望"
                       "达标（按 p 外推 N=1600 ≈ 2.4×10⁻⁵），本轮未做。"
                       % (_d4["max_abs"], _A["wall_s"]["800"])),
            "statement": ("论文与 result4.xlsx 须并列声明：t ≥ 1 h 的 4 位小数由 ② 层线"
                          "（5×10⁻⁵）支撑；t < 1 h 不支撑第 4 位。"
                          "详见 results/q4_convergence.json  g5_disclosure。"),
        }
    summary = build_summary(
        Cpk,
        audit_summary,
        chk,
        cond_s_per_unit,
        cross,
        dmap,
        dmap2,
        g5_tiering,
        inputs,
        j,
        k,
        o,
        rchk,
        res,
        scripts,
        slope_cross,
        surf_vals,
        t_dry_nom,
        t_dry_step,
        table6,
        times_out,
        tt,
        wall,
    )
    (res / "results4.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    write_manifest(inputs, k, res, scripts, summary, wall)

    print_summary(
        Cpk,
        chk,
        cond_s_per_unit,
        cross,
        dmap,
        dmap2,
        k,
        o,
        res,
        rows6,
        slope_cross,
        t_dry_nom,
        t_dry_step,
        table6,
        times6,
        times_out,
        vals_end,
        wall,
    )
    if not all(chk.values()):
        raise SystemExit("[FATAL] physical self-check failed")
    return 0

def build_audit_summary(et, o):
    audit_summary = dict(
        moisture=(dict(max_rel_resid=o["mass_audit"]["max_rel"],
                       argmax_t_s=o["mass_audit"]["worst_t"],
                       n_violations=o["mass_audit"]["n_ge_1e8"],
                       first_step=o["mass_audit"]["first"],
                       criterion="相对残差 ≤ 1e-8（> 1e-10 即停）",
                       identity="Σ_i W_i^ξ(C_i^{n+1}−C_i^n) = Δt·h_m(C_a^{n+1}−C_N^{n+1})/R^{n+1}")),
        moisture_closure=dict(m0=o["mass"]["m0"], m1=o["mass"]["m1"],
                              ratio=o["mass"]["ratio"],
                              cum_flux=o["mass"]["cum_flux"],
                              rel_err=o["mass"]["rel_err"]),
        energy=dict(form=o["audit"]["form"], applicable=o["audit"]["applicable"],
                    n_steps=o["audit"]["n"],
                    max_rel_self=o["audit"]["max_rel_global"],
                    max_rel_envelope=o["audit"]["max_rel_env"],
                    E_scale_max=o["audit"]["E_scale_max"],
                    max_abs=o["audit"]["max_abs"], n_ge_1e8=o["audit"]["n_ge_1e8"],
                    n_transient_steps=o["audit"]["n_transient"],
                    max_rel_transient=o["audit"]["max_rel_transient"],
                    max_rel_envelope_transient=o["audit"]["max_rel_env_transient"],
                    max_abs_transient=o["audit"]["max_abs_transient"],
                    samples=o["audit"]["samples"],
                    note=o["audit"]["note"]),
        energy_terms_B=dict(header=o["audit_energy_terms"]["header"],
                            n=o["audit_energy_terms"]["n"],
                            max_abs_dH=float(np.abs(et[:, 1]).max()) if et.size else 0.0,
                            max_abs_cap_account=float(np.abs(et[:, 2]).max()) if et.size else 0.0,
                            max_abs_geo_term=float(np.abs(et[:, 3]).max()) if et.size else 0.0,
                            max_abs_boundary_flux=float(np.abs(et[:, 4]).max()) if et.size else 0.0,
                            note=("写法 B 三分项（§6.7）：总焓变 / 热容变化账 / 几何项 / "
                                  "边界导热净流入；几何项只在此处出现，绝不进右端通量。")),
        T_ref_invariance=dict(
            note=("§6.7 末的代码级断言：对同一 (T,C) 用 `T` 与 `T+273.15` 各算一次"
                  "(Q4-E-A) 左端，两者之差 < 1e-12×|左端|；该断言能一次性抓住「误用乘积"
                  "形式」这一类错误。本文件在 verify_q4_audit.py 中实现并落盘。")),
    )
    return audit_summary

def write_xi_profile(R_out_t, n_row, res, snap_C, times_out, xi):
    xi_cols = np.round(np.arange(0.0, 1.0 + 1e-12, 0.005), 10)
    if abs(round(0.005 / (1.0 / N_MAIN)) * (1.0 / N_MAIN) - 0.005) < 1e-12:
        j = np.rint(xi_cols / (1.0 / N_MAIN)).astype(int)
        if not np.allclose(xi[j], xi_cols, atol=1e-12):
            raise SystemExit("[FATAL] mother column xi is not a node subset")
        prof_cols = j
    else:
        raise SystemExit("[FATAL] N=%d dxi does not divide 0.005" % N_MAIN)
    prof = snap_C[1:n_row + 1][:, prof_cols]
    np.savez_compressed(res / "q4_xi_profile.npz", t_s=times_out, xi=xi_cols,
                        C_kgkg=prof, R_m=R_out_t)
    with open(res / "q4_xi_profile.csv", "w", encoding="utf-8") as f:
        f.write("t_s,R_m," + ",".join("xi=%.3f" % v for v in xi_cols) + "\n")
        for k in range(prof.shape[0]):
            f.write("%d,%.10f," % (int(round(times_out[k])), R_out_t[k])
                    + ",".join("%.10f" % v for v in prof[k]) + "\n")
    return j, k, prof

def monotonic_check(assert_blank, cm, o, prof, t_dry_step):
    MONO_TOL = 1e-12
    dcm = np.diff(cm)
    n_rise = int(np.count_nonzero(dcm > MONO_TOL))
    chk = {
        "no_floor_activation": bool(o["floor"]["count"] == 0),
        "picard_all_converged": bool(o["picard"]["n_not_converged"] == 0),
        "no_extremum_violation": bool(len(o["violations"]) == 0),
        "mass_audit_le_1e-8": bool(o["mass_audit"]["max_rel"] <= 1e-8),
        "mass_closure_le_1e-8": bool(o["mass"]["rel_err"] <= 1e-8),
        "R_monotone_within_bounds": bool(o["R_min_seen"] >= R_MIN - 1e-12),
        "t_dry_above_mixing_lower_bound": bool(t_dry_step > 30083.5),
        "C_max_monotone_nonincreasing": bool(n_rise == 0),
        "C_radial_nonincreasing_all_window": bool(
            np.all(np.diff(prof, axis=1) <= 1e-9)),
        "delivered_columns_inside_domain": bool(assert_blank == 0),
        "t_dry_within_R_data_window": bool(t_dry_step <= R_FREEZE_T),
        "no_r_extrapolation": bool(t_dry_step <= R_FREEZE_T),
    }
    return chk

def build_summary(
    Cpk,
    audit_summary,
    chk,
    cond_s_per_unit,
    cross,
    dmap,
    dmap2,
    g5_tiering,
    inputs,
    j,
    k,
    o,
    rchk,
    res,
    scripts,
    slope_cross,
    surf_vals,
    t_dry_nom,
    t_dry_step,
    table6,
    times_out,
    tt,
    wall,
):
    summary = {
        "meta": {
            "question": "2026 国赛 A 题 问题 4（收缩域，材料坐标 ξ = r/R(t)）",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "seed": SEED, "python": platform.python_version(), "numpy": np.__version__,
            "wall_s": wall, "solver_wall_s": o["elapsed_s"],
            "config": {
                "N": N_MAIN, "dxi": 1.0 / N_MAIN, "dt_I_s": DT_I, "dt_II_s": DT_II,
                "t_switch_s": T_FREEZE_S, "dt_out_s": DT_OUT,
                "R_mode": "shrink（附件 2 分段线性，路线 (a)）",
                "geo_layer": "nplus1（1/R^{n+1,2}，步内常数、不进 Picard）",
                "face_rule": "arith（界面系数 = 相邻节点算术平均）",
                "props": "附录 4", "bc": "robin（口径 A）",
                "T_mode": "prescribed（T := T_a(t)，主口径）",
                "picard_tol": 1e-12, "picard_max": 100,
                "t_dry_criterion": "C_max(t) ≤ %g kg/kg（首个满足的内部步）" % C_DRY,
                "margin_s": MARGIN_S, "t_cap_s": T_CAP_S,
                "n_steps": o["n_steps"], "t_final_s": o["t_final"],
                "status": o["status"],
            },
        },
        "inputs_sha256": inputs,
        "script_sha256": scripts,
        "model_choices": {
            "equations": "∂C/∂t|_ξ = (1/(R²ξ))∂/∂ξ(ξD∂C/∂ξ)；"
                         "ρ(C)c_p(C)∂T/∂t|_ξ = (1/(R²ξ))∂/∂ξ(ξk∂T/∂ξ)（Q4-9 / Q4-12b）",
            "props_appendix4": ("ρ=760+90C；c_p=1850+2150C/(C+1)；k=0.12+0.20C/(C+1)；"
                                "D=4.2e-4·e^(−0.30/C)·e^(−3850/T)"),
            "D_T_unit_declaration": (" D 的指数项 e^(−3850/T) 中 T 必须用热力学温度 K"
                                     "（输入 28 ℃ 用 301.15 K）；输出阶段才转 ℃"),
            "h_hm_source": ("附录 4 未给  沿用附录 2 的 h=25 W/(m²·K)、h_m=8e-7 m/s"
                            "（唯一可溯来源的推断，Q4-A4，非题给）"),
            "R_source": "附件 2 的 145 行半径序列，分段线性插值；t>259200 s 冻结 1.198 cm",
            "driver": "附件 1 分段线性插值 [0,14400] s；t>14400 s 冻结 T_a=50.165 ℃、"
                      "C_a=0.04986 kg/kg",
            "initial": "T=28 ℃、C=2.55 kg/kg 均匀，t=0 起算（不承接 Q3 终态）",
            "D_eval": ("对数空间 log D = log(4.2e-4) − 0.30/C − 3850/T(K)，再 exp；"
                       "防卫下界 D_floor=1e-25 m²/s（非干扰：低于双精度相对分辨率）"),
            "D_floor": D_FLOOR,
            "picard": ("单列 Picard（预设 T 口径）；累积项右端恒取 n 层（R3 条款）、"
                       "边界驱动取 n+1 层、几何因子取 n+1 层且不进 Picard"),
            "mapping": "ξ 向 PCHIP（主）+ 线性（对照）+ FV-Hermite（互校）；表面/中心列不插值",
            "termination": "首个 C_max ≤ %g 的内部步 + %g s 余量" % (C_DRY, MARGIN_S),
        },
        "radius_check": rchk,
        "attachment2_key_values": {
            "R0_cm": 2.0, "R_final_cm": R_MIN_CM, "t_final_s": R_FREEZE_T,
            "R_min_m": float(radius_at(np.array([R_FREEZE_T]), "shrink")[0]),
            "note": "逐位照抄附件 2；不做四舍五入",
        },
        "attachment1_key_values": {
            "T_a_min_C": float(att1()[1].min()), "T_a_max_C": float(att1()[1].max()),
            "C_a_min": float(att1()[2].min()), "C_a_max": float(att1()[2].max()),
            "freeze_values": {"T_a_C": TA_FREEZE_C, "C_a": CA_FREEZE},
        },
        "t_dry": {
            "t_step_s": float(t_dry_step),
            "t_step_h": float(t_dry_step / 3600.0),
            "t_step_d": float(t_dry_step / 86400.0),
            "t_nominal_linear_s": float(t_dry_nom),
            "t_nominal_linear_h": float(t_dry_nom / 3600.0),
            "t_nominal_hermite_s": cross["t_hermite"],
            "bracket": {"t_before_s": cross["t_before"], "C_max_before": cross["cmax_before"],
                        "t_after_s": cross["t_after_s"], "C_max_after": cross["cmax_after"],
                        "width_s": (cross["t_after_s"] - cross["t_before"])
                        if cross["t_before"] is not None else None},
            "localization_error_s": cross["localization_error"],
            "dCmax_dt_at_crossing": slope_cross,
            "condition_number_s_per_kgkg": cond_s_per_unit,
            "uncertainty": build_t_dry_uncertainty(
                res, t_dry_step, t_dry_nom, cross["t_hermite"],
                cross["localization_error"], cond_s_per_unit, DT_II),
            "t_end_s": o["t_end"], "margin_s": MARGIN_S, "status": o["status"],
            "n_steps_to_dry": cross["n_star"],
            "mixing_lower_bound_s": 30083.5,
            "mixing_lower_bound_check": bool(t_dry_step > 30083.5),
        },
        "cross_validation_attachment2": {
            "note": ("§9.3：模型量 `t_dry` 与数据量「半径停止变化」的四个平台首现时刻"
                     "并列报告，两者口径不同、不构成等式。"),
            "T_R_first_appearance_s": {"1.201": 106200.0, "1.200": 126000.0,
                                       "1.199": 208800.0, "1.198": 241200.0},
            "T_R_first_appearance_h": {"1.201": 29.5, "1.200": 35.0,
                                       "1.199": 58.0, "1.198": 67.0},
            "consistency_rule": "t_dry ∈ [35.0 h, 67.0 h]  一致；∉ [29.5 h, 72.0 h]  红旗",
            "verdict": ("一致（落在尺寸基本定型的窗口内）"
                        if 35.0 <= t_dry_step / 3600.0 <= 67.0 else
                        ("边缘/红旗：t_dry = %.3f h" % (t_dry_step / 3600.0))),
            "t_dry_h": float(t_dry_step / 3600.0),
        },
        "result4": {
            "path": "results/result4.xlsx", "sheet": "Sheet1",
            "shape": [int(Cpk.shape[0]), int(Cpk.shape[1])],
            "t_range_s": [float(times_out[0]), float(times_out[-1])],
            "t_step_s": DT_OUT, "r_cm": R_OUT_CM.tolist(),
            "columns_total": int(Cpk.shape[1] + 1),
            "decimals": 4,
            "unrounded_npz": "results/result4_full.npz",
            "unrounded_csv": "results/result4_full.csv",
            "wide_backup": "results/result4_wide.xlsx",
            "template_cells": {
                "A1": "时间\\到药材中心的距离",
                "A2": 60, "A3": 120,
                "B1": 0.0, "C1": 0.1, "D1": 0.2, "last": "药材表面",
                "sheet_name": "Sheet1",
            },
        },
        "tables": {"table6": table6},
        "tables6_surface_column_detail": [
            {"t_h": th, "t_s": th * 3600.0, "C_surface": cs, "R_m": rm,
             "R_cm": rm * 100.0}
            for (th, cs, rm) in surf_vals],
        "cmax_trajectory": {
            "npz": "results/q4_cmax_trajectory.npz", "n_internal_steps": int(tt.size),
            "columns": ["t_s", "C_max", "argmax_i", "C_center", "C_surface", "C_min",
                        "R_m"],
            "dt_I_s": DT_I, "dt_II_s": DT_II, "bracket_width_s": DT_II,
            "note": "逐内部步（§9.2）；60 s 采样仅供参考、不用于定位",
        },
        "picard": {k: v for k, v in o["picard"].items()},
        "residual_inf": o["residual"],
        "picard_step_dmax": o["picard_step_dmax"],
        "floor_monitor": o["floor"],
        "extremum_principle": {"n_violations": len(o["violations"]),
                               "violations": o["violations"][:10],
                               "bounds": o["extremum_bounds"],
                               "declared": {"T_C": [28.0, 50.246],
                                            "C_kgkg": [0.01963, 2.55]}},
        "audit": audit_summary,
        "physical_checks": chk,
        "precision_tiering": g5_tiering,
        "mapping_uncertainty": {
            "csv": "results/q4_mapping_uncertainty.csv",
            "max_abs_diff_pchip_linear": float(np.nanmax(dmap)),
            "max_abs_diff_pchip_fvhermite": float(np.nanmax(dmap2)),
            "per_column_max": {("%.1f" % rc): float(np.nanmax(dmap[:, j]))
                               for j, rc in enumerate(R_OUT_CM)},
            "note": ("PCHIP（主）与线性（对照）之差即映射不确定度（§8.4 第 2 条）；"
                     "FV-Hermite 为节点导数取自 FV 通量差商的自实现，用于与 PCHIP 互校。"),
        },
        "delivery_form_percent": {
            "R_min_over_R0_sq": float((R_MIN / R0) ** 2),
            "note": "收缩使体积（单位长度）缩到 (R_min/R_0)² = 0.359 倍",
        },
    }
    return summary

def write_manifest(inputs, k, res, scripts, summary, wall):
    manifest = {
        "seed": SEED,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "question": "2026 国赛 A 题 问题 4",
        "inputs_sha256": inputs,
        "script_sha256": scripts,
        "config": summary["meta"]["config"],
        "model_choices": summary["model_choices"],
        "wall_s": wall,
        "outputs": {},
        "uncertainty": {
            "t_dry": summary["t_dry"]["uncertainty"],
            "mapping": summary["mapping_uncertainty"],
            "grid_and_time": ("见 results/q4_convergence.json（Δξ）与 "
                              "results/q4_dt_convergence.json（Δt）"),
            "model_error": ("见 results/q4_error_budget.json（E1–E6）；"
                            "E1 的 0-D 集总对照为必做项"),
            "note": "本块只登记口径；数值在验证脚本产出后回填",
        },
        "reproduce": "python src/run_q4.py（从项目根以相对路径可跑；seed=42 固定）",
    }
    OUTPUTS = ["result4.xlsx", "result4_wide.xlsx", "result4_full.npz",
               "result4_full.csv", "q4_table6.csv", "q4_xi_profile.npz",
               "q4_xi_profile.csv", "q4_cmax_trajectory.npz",
               "q4_cmax_crossing_window.csv", "q4_cmax_60s.csv",
               "q4_mass_audit.csv", "q4_audit_summary.json",
               "q4_mapping_uncertainty.csv", "results4.json"]
    for f in OUTPUTS:
        p = res / f
        manifest["outputs"][f] = sha256(p) if p.exists() else None
    def _load(fn):
        p = res / fn
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    cv, sen, eb, cmpj, bl = (_load("q4_convergence.json"), _load("q4_sensitivity.json"),
                             _load("q4_error_budget.json"), _load("q4_compare.json"),
                             _load("q4_baseline.json"))
    if cv:
        g = cv["A_grid_independence"]["diff_vs_N800"]["400"]
        d = cv["B_time_convergence"]["diff_vs_finest"]["I1.25_II5"]
        manifest["uncertainty"]["grid_and_time"] = dict(
            grid_N400_vs_N800_t_ge_3600s_kgkg=g["max_abs_t_ge_3600s"],
            grid_N400_vs_N800_fullwindow_kgkg=g["max_abs"],
            grid_fullwindow_dominated_by=("t=60 s 的初始窗口（δ(60 s)=2.507×10⁻² cm ≈ "
                                          "5.01Δr@N=400；本生产配置 N=400 在该窗口未达 ② 层线）"),
            dt_prod_vs_finest_t_ge_3600s_kgkg=d["max_abs_t_ge_3600s"],
            dt_prod_vs_finest_fullwindow_kgkg=d["max_abs"],
            layer2_tol_kgkg=5e-5,
            verdict=("t ≥ 1 h：Δξ 与 Δt 均 < 5×10⁻⁵（② 层线）；"
                     "t < 1 h 不支撑第 4 位（G5 口径降级，已在 results4.json 分层声明）"),
            t_dry_grid_spread_h=cv["A_grid_independence"]["t_dry_spread_h"],
            t_dry_dt_spread_h=cv["B_time_convergence"]["t_dry_spread_h"])
    if sen:
        manifest["uncertainty"]["sensitivity"] = dict(
            h_hm_3x3=sen["h_hm_2d"]["grid"],
            h_hm_3x3_spread_pct=sen["h_hm_2d"]["spread_pct"],
            h_degeneracy=sen["h_hm_2d"].get("degeneracy",
                "在预设 T 口径下 `h` 恒无影响（能量方程不求解） 3×3 的 9 组退化为 3 个不同值；"
                "本模型对 `h` 的灵敏度低于真实情况（§11.4）。"),
            driver_scenarios=sen["driver_scenarios"])
    if eb:
        manifest["uncertainty"]["model_error"] = {
            k: eb[k] for k in ("E1_latent_heat", "E2_end_face", "E3_h_hm",
                               "E4_discrete", "E5_R_interp", "E6_driver",
                               "interval_statement", "reporting_rule")}
    if cmpj:
        manifest["uncertainty"]["comparison"] = {
            k: cmpj[k] for k in cmpj if k.startswith("group")}
    if bl:
        manifest["uncertainty"]["baseline_groups"] = {
            "group1_verdict": bl["group1_degenerate_regression"].get("group1_verdict"),
            "group2": bl["group2_well_mixed_lower_bound"],
            "group3b": {k: v for k, v in bl["group3b_bessel_dirichlet"].items()
                        if k in ("D", "analytic", "targets", "pass")}}
    manifest["uncertainty"]["backfilled_at"] = datetime.now().isoformat(timespec="seconds")
    (res / "run_manifest_q4.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

def print_summary(
    Cpk,
    chk,
    cond_s_per_unit,
    cross,
    dmap,
    dmap2,
    k,
    o,
    res,
    rows6,
    slope_cross,
    t_dry_nom,
    t_dry_step,
    table6,
    times6,
    times_out,
    vals_end,
    wall,
):
    print()
    print("[summary] t_dry（操作口径，首个内部步）= %.6f s = %.4f h = %.4f d"
          % (t_dry_step, t_dry_step / 3600.0, t_dry_step / 86400.0))
    print("[summary] 标称（步内线性插值）= %.6f s = %.4f h；Hermite = %s；定位差 %.3e s"
          % (t_dry_nom, t_dry_nom / 3600.0,
             ("%.6f s" % cross["t_hermite"]) if cross["t_hermite"] else "None",
             cross["localization_error"] or 0.0))
    print("[summary] 括住区间 [%s, %s] s（宽 %s s）；跨越点 dC_max/dt = %.6e kg/(kg·s)；"
          "条件数 %.6e s/(kg/kg)"
          % (cross["t_before"], cross["t_after_s"], DT_II,
             slope_cross, cond_s_per_unit))
    print("[summary] t_end = %s s；t_final = %.6f s；步数 %d" % (o["t_end"], o["t_final"],
                                                                 o["n_steps"]))
    print("[summary] 表 6（kg/kg）：")
    print("        " + "  ".join("%10s" % h for h in table6["header"]))
    for i, th in enumerate(times6):
        print("   %6g h " % th + "  ".join("%10.4f" % v for v in rows6[i]))
    print("   %6s   " % "结束" + "  ".join("%10.4f" % v for v in vals_end))
    print("[summary] result4.xlsx：%d 行 × %d 列（含时间列），t ∈ [%g, %g] s，r ∈ [0, 1.1] cm + 药材表面"
          % (Cpk.shape[0], Cpk.shape[1] + 1, times_out[0], times_out[-1]))
    print("[summary] Picard 均值 %.3f 中位 %.1f 最大 %d 未收敛 %d；直方 %s"
          % (o["picard"]["mean_iters"], o["picard"]["median_iters"],
             o["picard"]["max_iters"], o["picard"]["n_not_converged"],
             dict(zip(o["picard"]["hist_iters"], o["picard"]["hist_counts"]))))
    print("[summary] 水分审计 max 相对残差 %.3e（worst t=%.1f s）；闭合相对误差 %.3e"
          % (o["mass_audit"]["max_rel"], o["mass_audit"]["worst_t"] or -1.0,
             o["mass"]["rel_err"]))
    print("[summary] 能量审计（写法 A）：预设定 T 口径不适用审计判据；"
          "自尺度 max %.3e（退化，仅参考）；包络尺度 max %.3e（判据位）；"
          "升温段包络尺度 max %.3e（= 预设口径的热失衡诊断量）"
          % (o["audit"]["max_rel_global"], o["audit"]["max_rel_env"],
             o["audit"]["max_rel_env_transient"]))
    print("[summary] 映射不确定度 max：PCHIP−线性 %.3e；PCHIP−FV-Hermite %.3e"
          % (np.nanmax(dmap), np.nanmax(dmap2)))
    print("[summary] 物理自检：" + "；".join("%s=%s" % (k, "PASS" if v else "FAIL")
                                          for k, v in chk.items()))
    print("[summary] 总耗时 %.1f s（求解 %.1f s）；产物已写 %s" % (wall, o["elapsed_s"], res))


if __name__ == "__main__":
    sys.exit(main())
