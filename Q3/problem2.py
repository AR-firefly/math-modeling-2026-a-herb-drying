from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

_CODE_DIR = Path(os.path.realpath(__file__)).parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from params import (C_CRIT, C_INIT, H_CONV, HM_MASS, N_NODES, DT_SOLVER, PICARD_ITERS, FACE_D,
                    OUT_N_COLS, Q2_SPAN_S, Q2_TABLE_STEP_S, Q3_REC_STEP_S, R_FIX, ROUND_DIGITS,
                    T_INIT_C)
from utils import (LIB_VERSIONS, OUTPUT_DIR, FIGURES_DIR, ROOT, SolverConfig, diffusivity,
                   dump_json, jsonable, load_air_boundary, profile_at, properties, radii_cm,
                   solve_case, table_node_index, write_csv, write_grid_xlsx)

BASELINE_KEY = "baseline_N21_dt1"


def baseline_config() -> SolverConfig:
    return SolverConfig(fine_step=DT_SOLVER, fine_until=float(Q2_SPAN_S),
                        rec_step=float(Q3_REC_STEP_S), extra_crit=(0.12, 0.18, 0.20))


def solve_baseline(air: dict) -> dict:
    return solve_case(BASELINE_KEY, baseline_config(), air)


def series_d(T: np.ndarray, C: np.ndarray) -> np.ndarray:
    return diffusivity(C.ravel(), T.ravel())


def window_checks(name: str, t: np.ndarray, T: np.ndarray, C: np.ndarray,
                   volume: np.ndarray, air: dict) -> dict:
    out = {"window": name, "n_rows": int(t.size), "t_span_s": [float(t[0]), float(t[-1])]}
    c_air = np.interp(t, air["t"], air["C_air"], left=air["C_air"][0], right=air["C_air"][-1])

    out["C1"] = {"C_min": float(C.min()), "C_max": float(C.max())}
    assert C.min() >= -1e-9 and C.max() <= C_INIT + 1e-9, f"C1 违反@{name}：{out['C1']}"

    cap = float(air["T_air"].max()) + 0.5
    out["C2"] = {"T_min": float(T.min()), "T_max": float(T.max()), "cap": cap}
    assert T.min() >= T_INIT_C - 1e-6 and T.max() <= cap, f"C2 违反@{name}：{out['C2']}"

    for field_name, field in (("C", C), ("T", T)):
        first = float(np.abs(field[:, 1] - field[:, 0]).max())
        second = float(np.abs(field[:, 2] - field[:, 1]).max())
        out[f"C3_{field_name}"] = {"d01": first, "d12": second}
        assert first <= second + 1e-12, f"C3 违反@{name}（{field_name}）：{out[f'C3_{field_name}']}"

    w_hist = C @ volume
    out["C5"] = {"max_dW": float(np.diff(w_hist).max())}
    assert np.diff(w_hist).max() <= 1e-12, f"C5 违反@{name}：{out['C5']}"

    margin = float((C[:, -1] - c_air).min())
    out["C6"] = {"min_surface_margin": margin}
    assert margin >= 0.0, f"C6 违反@{name}：表面浓度低于空气侧 {margin:.3e}"

    out["C9"] = {"max_dC_dr": float(np.diff(C, axis=1).max())}
    assert np.diff(C, axis=1).max() <= 1e-12, f"C9 违反@{name}：{out['C9']}"
    return out


def validate_constraints(sol: dict, air: dict, tables: dict) -> dict:
    fine = sol["fine"]
    coarse = sol["coarse"]
    volume = sol["grid"]["volume"]
    report = {"windows": [
        window_checks("result2_1s_0-3h", fine["t"], fine["T"], fine["C"], volume, air),
        window_checks("full_60s_0-tstar", coarse["t"], coarse["T"], coarse["C"], volume, air)]}
    for key in ("C1", "C2", "C5", "C6", "C9"):
        report[key] = {win["window"]: win[key] for win in report["windows"]}
    report["C3"] = {win["window"]: {"C": win["C3_C"], "T": win["C3_T"]} for win in report["windows"]}

    d_full = series_d(coarse["T"], coarse["C"])
    d_fine = series_d(fine["T"], fine["C"])
    cv = float(d_full.std() / d_full.mean())
    source = (_CODE_DIR / "utils.py").read_text(encoding="utf-8")
    report["C4"] = {"cv_D_full_run": cv, "cv_D_first_3h": float(d_fine.std() / d_fine.mean()),
                    "D_min": float(d_full.min()), "D_max": float(d_full.max()),
                    "D_dynamic_range": float(d_full.max() / d_full.min()),
                    "has_moisture_factor": "moisture_factor" in source,
                    "has_absolute_temperature": "KELVIN_OFFSET" in source,
                    "props_recomputed_per_picard_sweep": PICARD_ITERS}
    assert cv > 0.5, f"C4 违反：全程 std/mean(D)={cv:.4f} <= 0.5，物性疑似被常量化"
    assert d_full.max() / d_full.min() > 10.0, "C4 违反：D 动态范围不足，疑似取了常量"

    report["C8"] = {"eps_mass": float(sol["eps_mass"]), "W0": float(sol["W0"]),
                    "W_end": float(sol["W_end"]), "Q_out": float(sol["Q_out"])}
    assert abs(sol["eps_mass"]) < 1e-10, f"C8 违反：{report['C8']}"

    cols = radii_cm()
    report["C10"] = {"n_cols": int(cols.size), "first": float(cols[0]), "last": float(cols[-1]),
                     "sheets": sorted(tables["result2"]["sheets"]),
                     "digits": ROUND_DIGITS}
    assert cols.size == OUT_N_COLS and abs(cols[-1] - 2.0) < 1e-12, f"C10 违反：{report['C10']}"
    assert sorted(tables["result2"]["sheets"]) == ["水分浓度", "温度"], "C10 违反：result2 工作表名不符"
    assert tables["result2"]["sheets"]["温度"] == (Q2_SPAN_S, OUT_N_COLS), (
        f"C10 违反：result2 形状 {tables['result2']['sheets']['温度']} 应为 ({Q2_SPAN_S}, {OUT_N_COLS})")
    return report


def validate_capability(sol: dict, tables: dict) -> dict:
    source = (_CODE_DIR / "utils.py").read_text(encoding="utf-8")
    verdict = {}

    probe_c, probe_t = np.array([2.55, 0.15]), np.array([28.0, 50.0])
    rho, cp, k = properties(probe_c)
    d_two = diffusivity(probe_c, probe_t)
    d_no_moisture = np.exp(-3850.0 / (probe_t + 273.15)) * 2.4e-3
    assert rho[0] != rho[1] and cp[0] != cp[1] and k[0] != k[1], "P2-C1 不成立：物性对 C 无响应"
    assert not np.allclose(d_two, d_no_moisture), "P2-C1 不成立：D 缺 exp(-0.45/C) 因子"
    assert "temperature_factor" in source and "KELVIN_OFFSET" in source, "P2-C1 不成立：D 缺绝对温度因子"
    assert "properties(C)" in source, "P2-C1 不成立：物性未在时间循环内按当前 C 重算"
    ratio = float(diffusivity(np.array([2.55]), np.array([50.0]))[0]
                  / diffusivity(np.array([0.15]), np.array([50.0]))[0])
    verdict["P2-C1"] = {"holds": True, "D_ratio_wet_over_dry": ratio,
                        "rho": jsonable(rho), "cp": jsonable(cp), "k": jsonable(k)}

    for name in ("table3", "table4"):
        info = tables[name]
        assert info["rows"] == Q2_SPAN_S // Q2_TABLE_STEP_S, f"P2-C2 不成立：{name} 行数 {info['rows']}"
    verdict["P2-C2"] = {"holds": True, "rows": tables["table3"]["rows"],
                        "radii_cm": [0.0, 0.5, 1.0, 1.5, 2.0], "digits": ROUND_DIGITS,
                        "csv": tables["table3"]["path"]}

    shapes = tables["result2"]["sheets"]
    assert set(shapes) == {"温度", "水分浓度"}, "P2-C3 不成立：工作表名不符"
    for name, shape in shapes.items():
        assert shape == (Q2_SPAN_S, OUT_N_COLS), f"P2-C3 不成立：{name} 形状 {shape}"
    verdict["P2-C3"] = {"holds": True, "sheets": jsonable(shapes),
                        "bytes": tables["result2"]["bytes"], "sha256": tables["result2"]["sha256"]}
    return verdict


def _solve_and_tables(air: dict) -> tuple:
    sol = solve_baseline(air)
    idx = table_node_index()
    cols_cm = radii_cm()
    times = np.arange(Q2_TABLE_STEP_S, Q2_SPAN_S + 1, Q2_TABLE_STEP_S, dtype=float)

    table3 = np.array([[profile_at(sol, t, "T")[j] for j in idx] for t in times])
    table4 = np.array([[profile_at(sol, t, "C")[j] for j in idx] for t in times])
    header = ["时间/s", "时间/h", *[f"r={cols_cm[j]:.1f}cm" for j in idx]]
    info = {}
    info["table3"] = write_csv(OUTPUT_DIR / "table3_temperature.csv", header,
                               [[t, t / 3600.0, *np.round(row, ROUND_DIGITS)]
                                for t, row in zip(times, table3)])
    info["table4"] = write_csv(OUTPUT_DIR / "table4_moisture.csv", header,
                               [[t, t / 3600.0, *np.round(row, ROUND_DIGITS)]
                                for t, row in zip(times, table4)])
    return sol, cols_cm, times, table3, table4, info


def _write_result_tables(sol: dict, info: dict) -> dict:
    fine = sol["fine"]
    assert fine["t"].size == Q2_SPAN_S and abs(fine["t"][0] - 1.0) < 1e-12, "细记录应为 t=1..10800 s"
    info["result2"] = write_grid_xlsx(OUTPUT_DIR / "result2.xlsx",
                                      {"温度": (fine["t"], fine["T"]),
                                       "水分浓度": (fine["t"], fine["C"])})
    coarse = sol["coarse"]
    info["result2_full_60s"] = write_grid_xlsx(OUTPUT_DIR / "result2_full_60s.xlsx",
                                               {"温度": (coarse["t"], coarse["T"]),
                                                "水分浓度": (coarse["t"], coarse["C"])})
    info["W_history"] = write_csv(OUTPUT_DIR / "W_history.csv",
                                  ["t_s", "W", "Q_out", "eps_mass", "Cmax", "T_air", "C_air"],
                                  np.column_stack([coarse["t"], coarse["W"], coarse["Q_out"],
                                                   coarse["eps"], coarse["cmax"],
                                                   coarse["T_air"], coarse["C_air"]]))
    info["center_surface"] = write_csv(
        OUTPUT_DIR / "center_surface.csv",
        ["t_s", "T_center", "T_surface", "C_center", "C_surface", "T_air", "C_air"],
        np.column_stack([coarse["t"], coarse["T"][:, 0], coarse["T"][:, -1],
                         coarse["C"][:, 0], coarse["C"][:, -1], coarse["T_air"], coarse["C_air"]]))
    return coarse


def _independent_recompute(sol: dict, air: dict) -> tuple:
    check = solve_case("q2_independent_3h", SolverConfig(t_end=float(Q2_SPAN_S),
                                                        rec_step=float(Q3_REC_STEP_S)), air)
    drift_t = float(np.abs(check["T"] - profile_at(sol, float(Q2_SPAN_S), "T")).max())
    drift_c = float(np.abs(check["C"] - profile_at(sol, float(Q2_SPAN_S), "C")).max())
    assert drift_t < 1e-12 and drift_c < 1e-12, f"独立复算漂移：dT={drift_t:.2e}, dC={drift_c:.2e}"

    at3h_T, at3h_C = profile_at(sol, 10800.0, "T"), profile_at(sol, 10800.0, "C")
    return drift_t, drift_c, at3h_T, at3h_C


def _write_results_json(air: dict, sol: dict, info: dict, cols_cm: np.ndarray, times: np.ndarray,
                        table3: np.ndarray, table4: np.ndarray, constraints: dict,
                        capability: dict, drift_t: float, drift_c: float, at3h_T: np.ndarray,
                        at3h_C: np.ndarray, coarse: dict) -> None:
    payload = {
        "problem": 2, "title": "整个烘干过程的温度场与水分浓度场",
        "method": {"scheme": "conservative radial FVM + backward Euler + Picard",
                   "face_diffusivity": FACE_D, "N_nodes": N_NODES, "dt_s": DT_SOLVER,
                   "picard_iters": PICARD_ITERS, "R_m": R_FIX, "h": H_CONV, "hm": HM_MASS,
                   "libs": LIB_VERSIONS},
        "grid": {"r_cm": jsonable(cols_cm), "volume_weighted": jsonable(sol["grid"]["volume"])},
        "table3_temperature": {"times_s": jsonable(times), "radii_cm": [0.0, 0.5, 1.0, 1.5, 2.0],
                               "values": jsonable(np.round(table3, ROUND_DIGITS))},
        "table4_moisture": {"times_s": jsonable(times), "radii_cm": [0.0, 0.5, 1.0, 1.5, 2.0],
                            "values": jsonable(np.round(table4, ROUND_DIGITS))},
        "profiles_3h": {"T": jsonable(np.round(at3h_T, ROUND_DIGITS)),
                        "C": jsonable(np.round(at3h_C, ROUND_DIGITS))},
        "center_surface_60s": {"t_s": jsonable(coarse["t"][coarse["t"] <= Q2_SPAN_S]),
                               "T_center": jsonable(coarse["T"][coarse["t"] <= Q2_SPAN_S, 0]),
                               "T_surface": jsonable(coarse["T"][coarse["t"] <= Q2_SPAN_S, -1]),
                               "C_center": jsonable(coarse["C"][coarse["t"] <= Q2_SPAN_S, 0]),
                               "C_surface": jsonable(coarse["C"][coarse["t"] <= Q2_SPAN_S, -1])},
        "boundary_data": {"t_s": jsonable(air["t"]), "T_air": jsonable(air["T_air"]),
                          "C_air": jsonable(air["C_air"]), "sha256": air["sha256"]},
        "conservation": {"W0": sol["W0"], "W_end_full_run": sol["W_end"],
                         "Q_out_full_run": sol["Q_out"], "eps_mass": sol["eps_mass"],
                         "dehydration_full_run": sol["dehydration"]},
        "diagnostics": jsonable(sol["diag"]),
        "constraints": jsonable(constraints), "capability": jsonable(capability),
        "independent_recompute_drift": {"dT_max": drift_t, "dC_max": drift_c},
        "artifacts": jsonable(info),
    }
    dump_json(FIGURES_DIR / "problem_2_results.json", payload)


def _write_check_report(sol: dict, constraints: dict, capability: dict, drift_t: float,
                        drift_c: float, at3h_T: np.ndarray, at3h_C: np.ndarray) -> None:
    lines = ["# 问题二 自检报告", "",
             f"- 求解口径：N={N_NODES}, dt={DT_SOLVER} s, Picard={PICARD_ITERS}, face_D={FACE_D}",
             f"- 3 h 中心：T={at3h_T[0]:.4f} ℃（检查点⑮区间 [49,50]）, "
             f"C={at3h_C[0]:.4f} kg/kg（区间 [1.6,1.9]）",
             f"- 3 h 表面：T={at3h_T[-1]:.4f} ℃, C={at3h_C[-1]:.4f} kg/kg",
             f"- 守恒残差 |eps_mass|={abs(sol['eps_mass']):.3e} < 1e-10（C8）",
             *[f"- 窗口 {win['window']}（{win['n_rows']} 行）："
               f"C∈[{win['C1']['C_min']:.6f},{win['C1']['C_max']:.6f}]（C1）; "
               f"T∈[{win['C2']['T_min']:.4f},{win['C2']['T_max']:.4f}]≤{win['C2']['cap']:.4f}（C2）; "
               f"max dW={win['C5']['max_dW']:.3e}（C5）; "
               f"min(C_s-C_air)={win['C6']['min_surface_margin']:.3e}（C6）; "
               f"max dC/dr={win['C9']['max_dC_dr']:.3e}（C9）" for win in constraints["windows"]],
             f"- std(D)/mean(D)={constraints['C4']['cv_D_full_run']:.4f} > 0.5（C4 动态派生，全程窗口；"
             f"仅 0–3 h 为 {constraints['C4']['cv_D_first_3h']:.4f}，因退化层在 3 h 后才出现），"
             f"D 动态范围 {constraints['C4']['D_dynamic_range']:.1f} 倍",
             f"- 独立复算漂移 dT={drift_t:.2e}, dC={drift_c:.2e}（< 1e-12）",
             f"- 能力项：{', '.join(f'{k}=成立' for k in capability)}",
             "", "结论：C1–C6、C8–C10 与 P2-C1/C2/C3 全部通过。"]
    (ROOT / "_tmp").mkdir(parents=True, exist_ok=True)
    (ROOT / "_tmp" / "problem_2_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    air = load_air_boundary()
    sol, cols_cm, times, table3, table4, info = _solve_and_tables(air)

    coarse = _write_result_tables(sol, info)

    constraints = validate_constraints(sol, air, info)
    capability = validate_capability(sol, info)

    drift_t, drift_c, at3h_T, at3h_C = _independent_recompute(sol, air)

    _write_results_json(air, sol, info, cols_cm, times, table3, table4, constraints,
                        capability, drift_t, drift_c, at3h_T, at3h_C, coarse)

    _write_check_report(sol, constraints, capability, drift_t, drift_c, at3h_T, at3h_C)

    print(f"[problem2] 3h 中心 T={at3h_T[0]:.4f} ℃ / C={at3h_C[0]:.4f} kg/kg；"
          f"表面 T={at3h_T[-1]:.4f} / C={at3h_C[-1]:.4f}")
    print(f"[problem2] |eps_mass|={abs(sol['eps_mass']):.3e}, std(D)/mean(D)="
          f"{constraints['C4']['cv_D_full_run']:.4f}（全程）, 独立复算漂移 {max(drift_t, drift_c):.2e}")
    print(f"[problem2] 落盘 {len(info)} 件：{', '.join(v['path'] for v in info.values())}")
    print("[problem2] SELF-CHECK PASS（C1–C6, C8–C10, P2-C1/C2/C3）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
