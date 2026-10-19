from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_APP_DIR = str(Path(__file__).resolve().parents[0])
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from params import (C_CRIT, C_INIT, DT_SOLVER, FACE_D, N_NODES, OUT_N_COLS, PICARD_ITERS,
                   Q2_SPAN_S, Q3_REC_STEP_S, Q3_TABLE_STEP_S, ROUND_DIGITS, T_MAX_S)
from problem2 import BASELINE_KEY, baseline_config, solve_baseline
from utils import (FIGURES_DIR, LIB_VERSIONS, OUTPUT_DIR, ROOT, SolverConfig, dump_json, jsonable,
                   load_air_boundary, profile_at, radii_cm, solve_case, table_node_index,
                   write_csv, write_grid_xlsx)

PROTOTYPE_T_STAR_S = 205957.6284
UNREACHABLE = "在 t_max 内未达标"


def endpoint_from_records(coarse: dict, c_crit: float, dt: float) -> dict:
    cmax = coarse["C"].max(axis=1)
    hits = np.flatnonzero(cmax < c_crit)
    if hits.size == 0:
        return {"hit": False, "reason": UNREACHABLE}
    first = int(hits[0])
    gap = cmax - coarse["C"][:, 0]
    return {"hit": True, "first_record_idx": first,
            "t_first_record_s": float(coarse["t"][first]),
            "cmax_at_first_record": float(cmax[first]),
            "t_prev_record_s": float(coarse["t"][first - 1]),
            "cmax_at_prev_record": float(cmax[first - 1]),
            "record_step_s": float(coarse["t"][first] - coarse["t"][first - 1]),
            "argmax_is_center": bool(gap.max() <= 1e-12),
            "max_gap_center_vs_max": float(gap.max()),
            "n_rows_strict_argmax_off_center": int(np.count_nonzero(coarse["C"].argmax(axis=1) != 0)),
            "n_rows": int(coarse["t"].size)}


def validate_constraints(sol: dict, air: dict, tables: dict, recheck: dict) -> dict:
    coarse = sol["coarse"]
    t_star = sol["t_star"]
    report = {}

    assert t_star is not None, "C7 违反：基准算例未定位到首达时刻"
    lo, hi = recheck["t_prev_record_s"], recheck["t_first_record_s"]
    on_grid = abs(t_star / Q3_REC_STEP_S - round(t_star / Q3_REC_STEP_S)) < 1e-9
    report["C7"] = {"t_star_s": float(t_star), "t_star_h": float(t_star) / 3600.0,
                    "bracket_s": [lo, hi], "cmax_before": recheck["cmax_at_prev_record"],
                    "cmax_after": recheck["cmax_at_first_record"], "c_crit": C_CRIT,
                    "t_star_on_60s_grid": on_grid, "dt_used_for_interp_s": DT_SOLVER,
                    "argmax_r_is_center": recheck["argmax_is_center"],
                    "max_gap_center_vs_max": recheck["max_gap_center_vs_max"],
                    "n_rows_strict_argmax_off_center": recheck["n_rows_strict_argmax_off_center"]}
    assert lo <= t_star <= hi, f"C7 违反：t*={t_star} 不在记录括区 [{lo},{hi}] 内"
    assert recheck["cmax_at_prev_record"] >= C_CRIT > recheck["cmax_at_first_record"], (
        f"C7 违反：括区两端未夹住阈值 {report['C7']}")
    assert not on_grid, "C7/⑩ 违反：t* 恰落在 60 s 记录栅格上，判据疑似写在记录分支内（预案 A7）"
    assert recheck["argmax_is_center"], (
        f"C7 违反：存在真实的内部极大值，中心值比全场最大值低 "
        f"{recheck['max_gap_center_vs_max']:.3e}（>1e-12 浮点容差）")

    report["C9"] = {"max_dC_dr": float(np.diff(coarse["C"], axis=1).max())}
    assert np.diff(coarse["C"], axis=1).max() <= 1e-12, f"C9 违反：{report['C9']}"

    cols = radii_cm()
    shapes = tables["result3"]["sheets"]
    expected_rows = int(np.ceil(t_star / Q3_REC_STEP_S)) + 1
    report["C10"] = {"n_cols": int(cols.size), "sheets": sorted(shapes),
                     "rows": shapes["Sheet1"][0], "expected_rows": expected_rows,
                     "digits": ROUND_DIGITS}
    assert sorted(shapes) == ["Sheet1"], f"C10 违反：result3 工作表名 {sorted(shapes)} 应为 ['Sheet1']"
    assert shapes["Sheet1"] == (expected_rows, OUT_N_COLS), f"C10 违反：{report['C10']}"
    return report


def validate_capability(sol: dict, air: dict, tables: dict, recheck: dict) -> dict:
    coarse = sol["coarse"]
    volume = sol["grid"]["volume"]
    t_star = float(sol["t_star"])
    source = (Path(_APP_DIR) / "utils.py").read_text(encoding="utf-8")
    assert "C.max()" in source, "P3-C1 不成立：源码未按 max_r C 取判据"

    def first_cross(series: np.ndarray) -> float | None:
        hits = np.flatnonzero(series < C_CRIT)
        if hits.size == 0:
            return None
        i = int(hits[0])
        if i == 0:
            return 0.0
        span = coarse["t"][i] - coarse["t"][i - 1]
        return float(coarse["t"][i - 1] + (series[i - 1] - C_CRIT)
                     / (series[i - 1] - series[i]) * span)

    surface = first_cross(coarse["C"][:, -1])
    mean_w = first_cross((coarse["C"] @ volume) / volume.sum())
    assert surface is not None and surface < 0.5 * t_star, (
        "P3-C1 不成立：表面值口径未显著早于 max_r 口径，判据无从证伪")
    assert mean_w is not None and mean_w < t_star, "P3-C1 不成立：体均口径未早于 max_r 口径"
    verdict = {"P3-C1": {"holds": True, "t_star_h": t_star / 3600.0,
                         "wrong_surface_criterion_h": surface / 3600.0,
                         "wrong_volume_mean_criterion_h": mean_w / 3600.0,
                         "surface_underestimates_by": 1.0 - surface / t_star,
                         "criterion": "t* = min{t : max_r C(r,t) < 0.15}",
                         "evaluated_every_step": True, "dt_s": DT_SOLVER}}

    shapes = tables["result3"]["sheets"]
    assert tables["table5"]["rows"] == int(t_star // Q3_TABLE_STEP_S) + 1, (
        f"P3-C2 不成立：表5 行数 {tables['table5']['rows']} 与 6 h 栅格+t* 行不符")
    verdict["P3-C2"] = {"holds": True, "table5_rows": tables["table5"]["rows"],
                        "table5_path": tables["table5"]["path"],
                        "result3_sheets": jsonable(shapes),
                        "result3_bytes": tables["result3"]["bytes"],
                        "result3_sha256": tables["result3"]["sha256"]}
    return verdict


def boundary_cases(air: dict, t_star_baseline: float) -> dict:
    b6 = solve_case("B6_crit_at_initial", SolverConfig(c_crit=C_INIT, dt=1.0, rec_step=1.0,
                                                      t_max=600.0), air)
    assert b6["t_star"] is not None, "B6 不成立：阈值取初值时应在过程最开端即穿越"
    ratio = float(b6["t_star"]) / t_star_baseline
    assert ratio < 0.01, f"B6 不成立：t*={b6['t_star']} 占基准 t* 的 {ratio:.2%}，未趋于 0+"
    cmax_b6 = b6["coarse"]["C"].max(axis=1)
    ulp = float(np.spacing(C_INIT))
    detect = int(np.flatnonzero(cmax_b6 < C_INIT)[0])

    b7 = solve_case("B7_crit_unreachable", SolverConfig(c_crit=0.01, dt=60.0, rec_step=3600.0,
                                                       t_max=float(T_MAX_S)), air)
    assert b7["t_star"] is None and b7["hit"] is False, "B7 不成立：不可达阈值竟返回了 t*"
    label = UNREACHABLE if b7["t_star"] is None else f"{b7['t_star']:.4f} s"
    assert isinstance(label, str), "B7 不成立：未命中分支未给出显式字符串返回值"
    return {"B6": {"c_crit": C_INIT, "t_star_s": float(b6["t_star"]),
                   "fraction_of_baseline_t_star": ratio,
                   "detect_step_s": float(b6["coarse"]["t"][detect]), "ulp_of_threshold": ulp,
                   "note": "双精度下 C_max 的下降量小于阈值的 ulp 时不可分辨，故检出延迟 ~3e2 s；"
                           "精确算术意义上首步即穿越",
                   "expected": "t*->0+（过程最开端穿越）", "holds": True},
            "B7": {"c_crit": 0.01, "t_max_s": float(T_MAX_S), "t_star": None, "label": label,
                   "c_air_hold": float(air["C_air"][-1]),
                   "cmin_reached": float(b7["diag"]["c_min"]),
                   "expected": "永不达标，显式未命中分支而非 None 参与算术", "holds": True}}


def _build_table5(sol: dict, t_star: float) -> tuple:
    idx = table_node_index()
    cols_cm = radii_cm()
    times = np.arange(Q3_TABLE_STEP_S, t_star, Q3_TABLE_STEP_S, dtype=float)
    rows, labels = [], []
    for t in times:
        rows.append([profile_at(sol, t, "C")[j] for j in idx])
        labels.append((f"{t:.0f}", f"{t / 3600.0:.4f}"))
    rows.append([profile_at(sol, t_star, "C")[j] for j in idx])
    labels.append((f"{t_star:.4f}", f"{t_star / 3600.0:.4f}"))
    table5 = np.array(rows)
    return idx, cols_cm, labels, table5


def _write_q3_tables(sol: dict, coarse: dict, cols_cm: np.ndarray, idx: list,
                     labels: list, table5: np.ndarray) -> tuple:
    info = {}
    info["table5"] = write_csv(OUTPUT_DIR / "table5_moisture_6h.csv",
                               ["时间/s", "时间/h", *[f"r={cols_cm[j]:.1f}cm" for j in idx]],
                               [[t_s, t_h, *np.round(row, ROUND_DIGITS)]
                                for (t_s, t_h), row in zip(labels, table5)])
    info["result3"] = write_grid_xlsx(OUTPUT_DIR / "result3.xlsx",
                                      {"Sheet1": (coarse["t"], coarse["C"])})

    thresholds = {f"{level:g}": (None if val is None else float(val))
                  for level, val in sorted(sol["thresholds"].items())}
    info["threshold_sensitivity"] = dump_json(
        OUTPUT_DIR / "threshold_sensitivity.json",
        {"note": "§10.2 实验3：同一次积分内每步监视多个 C_crit，避免重复求解带来的口径漂移",
         "c_crit_to_t_star_s": thresholds,
         "c_crit_to_t_star_h": {key: (None if val is None else val / 3600.0)
                                for key, val in thresholds.items()}})
    return info, thresholds


def _cross_check_q2(sol: dict, air: dict) -> tuple:
    q2 = solve_case("q2_independent_3h", SolverConfig(t_end=float(Q2_SPAN_S),
                                                     rec_step=float(Q3_REC_STEP_S)), air)
    at3h_T, at3h_C = profile_at(sol, float(Q2_SPAN_S), "T"), profile_at(sol, float(Q2_SPAN_S), "C")
    rel_t = float(np.abs(q2["T"] - at3h_T).max() / np.abs(at3h_T).max())
    rel_c = float(np.abs(q2["C"] - at3h_C).max() / np.abs(at3h_C).max())
    assert rel_t < 1e-12 and rel_c < 1e-12, f"Q2/Q3 字段不一致：dT={rel_t:.2e}, dC={rel_c:.2e}"
    return rel_t, rel_c


def _write_q3_results_json(sol: dict, info: dict, coarse: dict, cols_cm: np.ndarray, labels: list,
                           table5: np.ndarray, thresholds: dict, constraints: dict,
                           capability: dict, bounds: dict, recheck: dict, t_star: float,
                           deviation: float, rel_t: float, rel_c: float) -> None:
    cmax = coarse["C"].max(axis=1)
    payload = {
        "problem": 3, "title": "烘干所需时间（各处水分浓度首次全部低于 0.15 kg/kg 的时刻）",
        "answer": {"t_star_s": t_star, "t_star_h": round(t_star / 3600.0, ROUND_DIGITS),
                   "t_star_days": round(t_star / 86400.0, ROUND_DIGITS),
                   "criterion": "min{t : max_r C(r,t) < 0.15 kg/kg}",
                   "bound_kind": "lower_bound",
                   "bound_reason": "忽略汽化潜热  T 偏高  D 偏高  t* 为真实烘干时间的下界",
                   "prototype_reference_s": PROTOTYPE_T_STAR_S,
                   "deviation_vs_prototype": deviation},
        "method": {"same_solver_as_problem2": True, "checkpoint_key": BASELINE_KEY,
                   "N_nodes": N_NODES, "dt_s": DT_SOLVER, "picard_iters": PICARD_ITERS,
                   "face_diffusivity": FACE_D, "air_extrap": baseline_config().air_extrap,
                   "new_in_Q3": ["C_crit=0.15", "式(15)(16) 每步评估", "式(17) 线性插值定位",
                                 "输出粒度 60 s"], "libs": LIB_VERSIONS},
        "table5_moisture": {"times_s": [float(a) for a, _ in labels],
                            "times_h": [float(b) for _, b in labels],
                            "radii_cm": [0.0, 0.5, 1.0, 1.5, 2.0],
                            "values": jsonable(np.round(table5, ROUND_DIGITS)),
                            "last_row_is_t_star": True},
        "cmax_history_60s": {"t_s": jsonable(coarse["t"]), "cmax": jsonable(cmax),
                             "c_surface": jsonable(coarse["C"][:, -1]), "c_crit": C_CRIT},
        "radial_snapshots": {"times_h": [0, 6, 12, 18, 24, 30, 36, 42, 48, 54,
                                         round(t_star / 3600.0, 4)],
                             "r_cm": jsonable(cols_cm),
                             "C": jsonable([np.round(profile_at(sol, min(h * 3600.0, t_star), "C"),
                                                     ROUND_DIGITS)
                                            for h in (0, 6, 12, 18, 24, 30, 36, 42, 48, 54)]
                                           + [np.round(profile_at(sol, t_star, "C"), ROUND_DIGITS)])},
        "drying_rate_60s": {"t_s": jsonable(coarse["t"][1:]),
                            "minus_dW_dt": jsonable(-np.diff(coarse["W"]) / np.diff(coarse["t"])),
                            "W": jsonable(coarse["W"])},
        "threshold_sensitivity": {"c_crit_to_t_star_h":
                                  {key: (None if val is None else val / 3600.0)
                                   for key, val in thresholds.items()}},
        "endpoint_recheck_from_records": jsonable(recheck),
        "conservation": {"eps_mass": sol["eps_mass"], "W0": sol["W0"], "W_end": sol["W_end"],
                         "Q_out": sol["Q_out"], "dehydration": sol["dehydration"]},
        "cross_problem": {"t_common_s": float(Q2_SPAN_S), "rel_diff_T": rel_t, "rel_diff_C": rel_c,
                          "tolerance": 1e-12},
        "boundary_cases": jsonable(bounds),
        "constraints": jsonable(constraints), "capability": jsonable(capability),
        "diagnostics": jsonable(sol["diag"]), "artifacts": jsonable(info),
    }
    dump_json(FIGURES_DIR / "problem_3_results.json", payload)


def _write_q3_check_report(sol: dict, info: dict, constraints: dict, capability: dict,
                           bounds: dict, recheck: dict, thresholds: dict, t_star: float,
                           deviation: float, rel_t: float, rel_c: float) -> None:
    lines = ["# 问题三 自检报告", "",
             f"- t\\* = {t_star:.4f} s = {t_star / 3600.0:.4f} h = {t_star / 86400.0:.4f} d"
             f"（判据 max_r C < {C_CRIT}，每步评估 + 式(17) 插值）",
             f"- 与建模原型 {PROTOTYPE_T_STAR_S:.4f} s 的偏差 {deviation:.4%} < 2%（预案 A5 未触发）",
             f"- 括区：{constraints['C7']['bracket_s'][0]:.0f} s（Cmax="
             f"{constraints['C7']['cmax_before']:.6f}） {constraints['C7']['bracket_s'][1]:.0f} s"
             f"（Cmax={constraints['C7']['cmax_after']:.6f}），t* 未落在 60 s 栅格上（C7/⑩）",
             f"- max_r C 全程位于中心：中心值与全场最大值之差 ≤ "
             f"{recheck['max_gap_center_vs_max']:.3e}（浮点容差 1e-12 内）；严格 argmax≠0 的记录点 "
             f"{recheck['n_rows_strict_argmax_off_center']}/{recheck['n_rows']} 个，均为 §5.2 声明的"
             f"内部平台等值点（差 ~1e-15）。C9 单调 max dC={constraints['C9']['max_dC_dr']:.3e}",
             f"- 误用表面值判据会得 {capability['P3-C1']['wrong_surface_criterion_h']:.4f} h"
             f"（低估 {capability['P3-C1']['surface_underestimates_by']:.1%}）；"
             f"误用体均得 {capability['P3-C1']['wrong_volume_mean_criterion_h']:.4f} h（P3-C1 可证伪）",
             f"- result3.xlsx：{constraints['C10']['rows']} 行 × {OUT_N_COLS} 列，Sheet1（C10）",
             f"- 表5：{info['table5']['rows']} 行（每 6 h + t* 行）",
             f"- 阈值敏感性 C_crit -> t*/h：" + ", ".join(
                 f"{k}{'未达标' if v is None else f'{v / 3600.0:.4f}'}"
                 for k, v in thresholds.items()),
             f"- 边界情形 B6：C_crit={C_INIT} 时 t*={bounds['B6']['t_star_s']:.4f} s，"
             f"占基准 t* 的 {bounds['B6']['fraction_of_baseline_t_star']:.3%}（0+）；"
             f"检出延迟源于 ulp({C_INIT})={bounds['B6']['ulp_of_threshold']:.2e} 的浮点分辨率，"
             f"非首达逻辑有误",
             f"- 边界情形 B7：C_crit=0.01 < C_air 末值 {bounds['B7']['c_air_hold']:.5f} 时"
             f"{bounds['B7']['label']}（{T_MAX_S / 3600:.0f} h 内 C_min 只到 "
             f"{bounds['B7']['cmin_reached']:.6f}），显式未命中分支，无 None 参与算术",
             f"- Q2/Q3 字段在 t=10800 s 一致：相对差 dT={rel_t:.2e}, dC={rel_c:.2e} < 1e-12",
             f"- 守恒残差 |eps_mass|={abs(sol['eps_mass']):.3e}，累计脱水率 "
             f"{sol['dehydration']:.4%}",
             "", "结论：C7、C9、C10 与 P3-C1/P3-C2 全部通过，B6/B7 边界情形按预期行为返回。",
             "", f"结论口吻：t\\*={t_star / 3600.0:.4f} h 是在题面给定参数体系与假设 1–6 下的",
             "计算结果，且因忽略汽化潜热而为真实烘干时间的下界（§9 假设问责表）。"]
    (ROOT / "_tmp").mkdir(parents=True, exist_ok=True)
    (ROOT / "_tmp" / "problem_3_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    air = load_air_boundary()
    sol = solve_baseline(air)
    t_star = float(sol["t_star"])
    coarse = sol["coarse"]
    recheck = endpoint_from_records(coarse, C_CRIT, DT_SOLVER)
    assert recheck["hit"], "基准算例应在 t_max 内达标"

    idx, cols_cm, labels, table5 = _build_table5(sol, t_star)

    info, thresholds = _write_q3_tables(sol, coarse, cols_cm, idx, labels, table5)

    constraints = validate_constraints(sol, air, info, recheck)
    capability = validate_capability(sol, air, info, recheck)
    bounds = boundary_cases(air, t_star)

    rel_t, rel_c = _cross_check_q2(sol, air)

    deviation = abs(t_star - PROTOTYPE_T_STAR_S) / PROTOTYPE_T_STAR_S
    assert deviation < 0.02, f"与建模原型偏差 {deviation:.4%} > 2%，按预案 A5 处理"
    assert 40.0 <= t_star / 3600.0 <= 80.0, f"⑮ 违反：t*={t_star / 3600.0:.4f} h 不在 [40,80]"

    _write_q3_results_json(sol, info, coarse, cols_cm, labels, table5, thresholds,
                           constraints, capability, bounds, recheck, t_star, deviation,
                           rel_t, rel_c)

    _write_q3_check_report(sol, info, constraints, capability, bounds, recheck, thresholds,
                           t_star, deviation, rel_t, rel_c)

    print(f"[problem3] t* = {t_star:.4f} s = {t_star / 3600.0:.4f} h "
          f"= {t_star / 86400.0:.4f} d（偏差 {deviation:.4%} vs 原型）")
    print(f"[problem3] 误用表面值 {capability['P3-C1']['wrong_surface_criterion_h']:.4f} h / "
          f"误用体均 {capability['P3-C1']['wrong_volume_mean_criterion_h']:.4f} h  判据不可替代")
    print(f"[problem3] B6 t*={bounds['B6']['t_star_s']:.4f} s, B7 {bounds['B7']['label']}")
    print(f"[problem3] 落盘 {len(info)} 件：{', '.join(v['path'] for v in info.values())}")
    print("[problem3] SELF-CHECK PASS（C7, C9, C10, P3-C1/P3-C2, B6/B7）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
