from __future__ import annotations

import json
import re
from pathlib import Path

_PROJ_ROOT = Path(__file__).resolve().parents[1]
_FACTS_PATH = _PROJ_ROOT / "数据台账" / "DATA_FACTS.json"
_REPORT_PATH = _PROJ_ROOT / "数据台账" / "MODELING_REPORT.md"
_LEDGER_PATH = _PROJ_ROOT / "数据台账" / "CROSS_PROBLEM_LEDGER.json"

_FACTS = json.loads(_FACTS_PATH.read_text(encoding="utf-8"))
_GIVEN = {item["name"]: item for item in _FACTS["given"]}

_NUM = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def const(name: str) -> float:
    item = _GIVEN[name]
    assert item["kind"] == "constant", f"{name} 不是 constant 条目"
    return float(item["value"])


def formula_coeffs(name: str, count: int) -> list[float]:
    item = _GIVEN[name]
    assert item["kind"] == "formula", f"{name} 不是 formula 条目"
    rhs = item["form"].split("=", 1)[1]
    found = [float(x) for x in _NUM.findall(rhs)]
    assert len(found) >= count, f"{name} 解析到的系数不足 {count} 个：{found}"
    return found[:count]


LENGTH_M = const("药材几何-长")
R_FIX = const("药材几何-半径")
T_INIT_C = const("初始温度")
C_INIT = const("初始水分浓度")
H_CONV = const("对流换热系数 h")
HM_MASS = const("对流传质系数 h_m")
C_CRIT = const("问题三烘干终点阈值")

_RHO_A, _RHO_B = formula_coeffs("密度经验公式(Q2/Q3)", 2)
_CP_A, _CP_B = formula_coeffs("比热容经验公式(Q2/Q3)", 2)
_K_A, _K_B = formula_coeffs("热传导系数经验公式(Q2/Q3)", 2)
_D_PRE, _D_EC, _D_ET = formula_coeffs("水分扩散系数经验公式(Q2/Q3)", 3)

RHO_COEF = (_RHO_A, _RHO_B)
CP_COEF = (_CP_A, _CP_B)
K_COEF = (_K_A, _K_B)
D_COEF = (_D_PRE, _D_EC, _D_ET)
RHO_DRY_REF = _RHO_A

_REPORT = _REPORT_PATH.read_text(encoding="utf-8")
_ROW = re.compile(r"^\|[^|]+\|\s*(" + _NUM.pattern + r")\s*\|[^|]*\|\s*`([A-Za-z_]+)`\s*\|", re.M)
_CAL = {label: float(value) for value, label in _ROW.findall(_REPORT)}
assert _CAL, "未能从 MODELING_REPORT.md §7.3 解析参数口径表"

N_NODES = int(_CAL["N_nodes"])
DT_SOLVER = _CAL["dt_solver"]
T_AIR_HOLD = _CAL["T_air_hold"]
C_AIR_HOLD = _CAL["C_air_hold"]

PICARD_ITERS = int(re.search(r"PICARD_ITERS\s*[=:]\s*(\d+)", _REPORT).group(1))
FACE_D = "kirchhoff"
FACE_K = "arith"
AIR_EXTRAP = "hold_last"
LATENT_HEAT_COUPLING = False
SHRINKAGE = False

_UNIT_TEMP = json.loads(_LEDGER_PATH.read_text(encoding="utf-8"))["unit_ledger"]["temperature_in_properties"]
KELVIN_OFFSET = float(re.search(r"\+\s*(" + _NUM.pattern + r")", _UNIT_TEMP).group(1))

OUT_DR_CM = 0.1
OUT_R_MAX_CM = R_FIX * 100.0
OUT_N_COLS = int(round(OUT_R_MAX_CM / OUT_DR_CM)) + 1
Q2_SPAN_S = 3 * 3600
Q2_TABLE_STEP_S = 1800
Q3_TABLE_STEP_S = 6 * 3600
Q3_REC_STEP_S = 60
ROUND_DIGITS = 4
T_MAX_S = 150 * 3600
D_SAFE_FLOOR = 1e-9

assert abs(_CAL["radius"] - R_FIX) < 1e-12, "半径口径冲突：口径表 vs DATA_FACTS"
assert abs(_CAL["length"] - LENGTH_M) < 1e-12, "长度口径冲突"
assert abs(_CAL["T_init"] - T_INIT_C) < 1e-12, "初温口径冲突"
assert abs(_CAL["C_init"] - C_INIT) < 1e-12, "初始水分口径冲突"
assert abs(_CAL["h_conv"] - H_CONV) < 1e-12, "对流换热系数口径冲突"
assert abs(_CAL["hm_mass"] - HM_MASS) / HM_MASS < 1e-9, "对流传质系数口径冲突"
assert abs(_CAL["C_crit"] - C_CRIT) < 1e-12, "终点阈值口径冲突"
assert abs(_CAL["rho_dry_ref"] - RHO_DRY_REF) < 1e-12, "干固体参考密度口径冲突"

_DR_MM = R_FIX / (N_NODES - 1) * 1000.0
assert abs(_DR_MM - OUT_DR_CM * 10.0) < 1e-12, (
    f"网格与输出栅格不一致：dr={_DR_MM} mm 与输出步长 {OUT_DR_CM} cm 不重合，"
    "N=21 的目的就是让 r=0/0.5/1/1.5/2 cm 直接落节点（§13 S11）")
assert N_NODES == OUT_N_COLS, "主算节点数应与输出列数一致（各 0.1 cm 一列）"
assert 200 < KELVIN_OFFSET < 300, "绝对温度换算量异常"
assert LATENT_HEAT_COUPLING is False and SHRINKAGE is False, "假设 3/6 开关被改动"

if __name__ == "__main__":
    print(f"[params] R={R_FIX} m, T0={T_INIT_C} C, C0={C_INIT} kg/kg, h={H_CONV}, hm={HM_MASS}")
    print(f"[params] rho={RHO_COEF}, cp={CP_COEF}, k={K_COEF}, D={D_COEF}, K_offset={KELVIN_OFFSET}")
    print(f"[params] N={N_NODES}, dt={DT_SOLVER}, Picard={PICARD_ITERS}, C_crit={C_CRIT}")
    print(f"[params] hold-last: T_air={T_AIR_HOLD} C, C_air={C_AIR_HOLD} kg/kg")
    print("[params] 口径一致性 OK")
