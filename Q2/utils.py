from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import openpyxl
import scipy
from scipy.linalg import solve_banded
from scipy.optimize import brentq
from scipy.special import exp1, j0, j1, jn_zeros

_HERE = Path(__file__).parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from params import (
    R_FIX, T_INIT_C, C_INIT, H_CONV, HM_MASS, C_CRIT, KELVIN_OFFSET,
    RHO_COEF, CP_COEF, K_COEF, D_COEF, N_NODES, DT_SOLVER, PICARD_ITERS,
    FACE_D, FACE_K, AIR_EXTRAP, T_AIR_HOLD, C_AIR_HOLD, T_MAX_S, D_SAFE_FLOOR,
    Q3_REC_STEP_S, OUT_N_COLS, OUT_DR_CM, ROUND_DIGITS,
)

ROOT = _HERE
OUTPUT_DIR = ROOT / "output"
FIGURES_DIR = ROOT / "figures"
AIR_FILE = ROOT.parent / "附件" / "附件1.xlsx"
AIR_SHEET = "Sheet1"

LIB_VERSIONS = {
    "python": platform.python_version(),
    "numpy": np.__version__,
    "scipy": scipy.__version__,
    "openpyxl": openpyxl.__version__,
}


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_air_boundary() -> dict:
    profile = json.loads((ROOT.parent / "数据台账" / "DATA_PROFILE.json").read_text(encoding="utf-8"))
    entry = profile["files"].get(AIR_FILE.name, {})
    assert "total_rows" in entry, (
        f"[未建档] {AIR_FILE.name} 不在 DATA_PROFILE.json 或建档失败（{entry.get('error', '缺记录')}）")

    book = openpyxl.load_workbook(AIR_FILE, data_only=True, read_only=True)
    assert AIR_SHEET in book.sheetnames, f"附件1 缺工作表 {AIR_SHEET}：{book.sheetnames}"
    sheet = book[AIR_SHEET]
    rows = list(sheet.iter_rows(values_only=True))
    book.close()

    header = [str(x).strip() if x is not None else "" for x in rows[0]]
    idx = {name: header.index(name) for name in ("时间", "温度", "水分浓度")}
    records = [r for r in rows[1:] if r[idx["时间"]] is not None]

    t = np.array([float(r[idx["时间"]]) for r in records])
    T = np.array([float(r[idx["温度"]]) for r in records])
    C = np.array([float(r[idx["水分浓度"]]) for r in records])

    assert len(records) == entry["total_rows"] - 1 or len(records) == entry["total_rows"], (
        f"[摄入不全] 附件1 实读 {len(records)} 条数据行，建档 total_rows={entry['total_rows']}"
        f"（{entry['n_sheets']} 张 sheet）——检查是否被截断或只读了首表")
    assert np.all(np.diff(t) > 0), "附件1 时间列非严格递增"
    steps = np.unique(np.diff(t))
    assert steps.size == 1 and abs(steps[0] - 60.0) < 1e-9, f"附件1 时间步长非 60 s：{steps}"
    assert np.all(np.isfinite(T)) and np.all(np.isfinite(C)), "附件1 含非有限值"
    print(f"[data_ingest] 附件1 全量 {len(records)} 行 / {entry['n_sheets']} sheet 已核对通过；"
          f"t∈[{t.min():.0f},{t.max():.0f}] s, T_air∈[{T.min():.3f},{T.max():.3f}] ℃, "
          f"C_air∈[{C.min():.5f},{C.max():.5f}] kg/kg")
    return {"t": t, "T_air": T, "C_air": C, "sha256": sha256_of(AIR_FILE)}


PHI_TABLE_N = 131072
PHI_ERR_TOL = 1e-6


def phi_exact(C: np.ndarray) -> np.ndarray:
    C = np.maximum(np.asarray(C, dtype=float), 1e-300)
    a = -D_COEF[1]
    return C * np.exp(-a / C) - a * exp1(a / C)


def build_kirchhoff_table(n_points: int = PHI_TABLE_N) -> tuple[np.ndarray, np.ndarray]:
    grid = np.concatenate(([0.0], np.logspace(-4.0, np.log10(3.0), n_points)))
    integrand = np.zeros_like(grid)
    positive = grid > 0
    integrand[positive] = np.exp(D_COEF[1] / grid[positive])
    phi = np.concatenate(([0.0], np.cumsum(np.diff(grid) * 0.5 * (integrand[1:] + integrand[:-1]))))
    assert np.all(np.diff(phi) >= 0.0), "Kirchhoff 势必须单调不减"
    return grid, phi


_PHI_GRID, _PHI_VALS = build_kirchhoff_table()
_PHI_PROBE = np.linspace(0.02, 3.0, 2000)
PHI_TABLE_ERR = float(np.max(np.abs(np.interp(_PHI_PROBE, _PHI_GRID, _PHI_VALS)
                                    - phi_exact(_PHI_PROBE)) / phi_exact(_PHI_PROBE)))
assert PHI_TABLE_ERR < PHI_ERR_TOL, (
    f"Kirchhoff 查表精度 {PHI_TABLE_ERR:.3e} 未达 §7.2 口径 {PHI_ERR_TOL}")


def phi_of(C: np.ndarray) -> np.ndarray:
    return np.interp(np.asarray(C, dtype=float), _PHI_GRID, _PHI_VALS)


def moisture_factor(C: np.ndarray) -> np.ndarray:
    return np.exp(D_COEF[1] / np.maximum(np.asarray(C, dtype=float), D_SAFE_FLOOR))


def temperature_factor(T_c: np.ndarray) -> np.ndarray:
    return np.exp(D_COEF[2] / (np.asarray(T_c, dtype=float) + KELVIN_OFFSET))


def diffusivity(C: np.ndarray, T_c: np.ndarray) -> np.ndarray:
    return D_COEF[0] * moisture_factor(C) * temperature_factor(T_c)


def properties(C: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    C = np.asarray(C, dtype=float)
    ratio = C / (C + 1.0)
    rho = RHO_COEF[0] + RHO_COEF[1] * C
    cp = CP_COEF[0] + CP_COEF[1] * ratio
    k = K_COEF[0] + K_COEF[1] * ratio
    return rho, cp, k


def face_diffusivity(C: np.ndarray, T_c: np.ndarray, mode: str = FACE_D) -> np.ndarray:
    T_face = 0.5 * (T_c[:-1] + T_c[1:])
    arr_t = temperature_factor(T_face)
    if mode == "kirchhoff":
        dC = C[1:] - C[:-1]
        small = np.abs(dC) < 1e-12
        C_bar = 0.5 * (C[:-1] + C[1:])
        slope = np.where(small, moisture_factor(C_bar),
                         (phi_of(C[1:]) - phi_of(C[:-1])) / np.where(small, 1.0, dC))
        return D_COEF[0] * arr_t * slope
    if mode == "midpoint":
        return D_COEF[0] * arr_t * moisture_factor(0.5 * (C[:-1] + C[1:]))
    node_d = diffusivity(C, T_c)
    if mode == "arith":
        return 0.5 * (node_d[:-1] + node_d[1:])
    if mode == "harmonic":
        lo, hi = node_d[:-1], node_d[1:]
        return np.where(lo + hi > 0.0, 2.0 * lo * hi / np.where(lo + hi > 0.0, lo + hi, 1.0), 0.0)
    raise ValueError(f"未知界面格式：{mode}")


def face_conductivity(k_node: np.ndarray, mode: str = FACE_K) -> np.ndarray:
    if mode == "arith":
        return 0.5 * (k_node[:-1] + k_node[1:])
    if mode == "harmonic":
        lo, hi = k_node[:-1], k_node[1:]
        return 2.0 * lo * hi / (lo + hi)
    raise ValueError(f"未知界面格式：{mode}")


def radial_grid(n_nodes: int, radius: float = R_FIX, stretch: float = 1.0) -> dict:
    assert n_nodes >= 3, "至少 3 个节点"
    assert stretch >= 1.0, "stretch<1 会把网格挤向中心，与退化层加密的目的相反"
    if stretch == 1.0:
        r = np.linspace(0.0, radius, n_nodes)
    else:
        widths = stretch ** np.arange(n_nodes - 1)
        r = np.concatenate(([0.0], np.cumsum(widths) / widths.sum() * radius))
    r[-1] = radius
    r_face = 0.5 * (r[:-1] + r[1:])
    edges = np.concatenate(([0.0], r_face, [radius]))
    volume = 0.5 * (edges[1:] ** 2 - edges[:-1] ** 2)
    assert abs(volume.sum() - 0.5 * radius ** 2) < 1e-15 * radius ** 2, "加权体积未闭合到 R^2/2"
    return {"dr": float(r[1] - r[0]), "r": r, "volume": volume, "r_face": r_face,
            "dr_face": np.diff(r), "radius": radius, "stretch": stretch}


def solve_tridiag(volume: np.ndarray, dt: float, time_coef: np.ndarray,
                   conductance: np.ndarray, robin: float, robin_source: float,
                   old: np.ndarray) -> tuple[np.ndarray, float]:
    n = volume.size
    ab = np.zeros((3, n))
    rhs = time_coef * old
    diag = time_coef.copy()
    diag[:-1] += conductance
    diag[1:] += conductance
    diag[-1] += robin
    rhs[-1] += robin_source
    ab[0, 1:] = -conductance
    ab[1, :] = diag
    ab[2, :-1] = -conductance
    off_sum = np.zeros(n)
    off_sum[:-1] += conductance
    off_sum[1:] += conductance
    margin = float(np.min(diag - off_sum))
    return solve_banded((1, 1), ab, rhs), margin


@dataclass
class SolverConfig:
    n_nodes: int = N_NODES
    dt: float = DT_SOLVER
    picard_iters: int = PICARD_ITERS
    face_d: str = FACE_D
    face_k: str = FACE_K
    air_extrap: str = AIR_EXTRAP
    radius: float = R_FIX
    stretch: float = 1.0
    t_init: float = T_INIT_C
    c_init: float = C_INIT
    s_d: float = 1.0
    s_hm: float = 1.0
    s_h: float = 1.0
    t_air_shift: float = 0.0
    const_d: float | None = None
    freeze_props: bool = False
    air_const: tuple | None = None
    c_crit: float = C_CRIT
    extra_crit: tuple = ()
    t_end: float | None = None
    t_max: float = T_MAX_S
    rec_step: float = Q3_REC_STEP_S
    fine_step: float | None = None
    fine_until: float = 0.0


def air_series(air: dict, cfg: SolverConfig):
    if cfg.air_const is not None:
        t_const, c_const = float(cfg.air_const[0]), float(cfg.air_const[1])
        return (lambda _t: t_const), (lambda _t: c_const)
    grid = air["t"]
    temp = air["T_air"] + cfg.t_air_shift
    conc = air["C_air"]
    if cfg.air_extrap == "hold_last":
        right_t, right_c = temp[-1], conc[-1]
        assert abs(right_t - cfg.t_air_shift - T_AIR_HOLD) < 5e-4, "hold-last 末值温度与口径表不符"
        assert abs(right_c - C_AIR_HOLD) < 5e-6, "hold-last 末值浓度与口径表不符"
    elif cfg.air_extrap == "mean_tail":
        tail = grid >= grid[-1] - 3600.0
        right_t, right_c = float(temp[tail].mean()), float(conc[tail].mean())
    else:
        raise ValueError(f"未知外推口径：{cfg.air_extrap}")
    return (lambda tt: float(np.interp(tt, grid, temp, left=temp[0], right=right_t)),
            lambda tt: float(np.interp(tt, grid, conc, left=conc[0], right=right_c)))


def _advance_in_time(cfg: SolverConfig, T: np.ndarray, C: np.ndarray, q_out, w_prev, w0,
                     cmax_prev, tail_target, diag: dict, t_hard_stop, dr, volume, r_face,
                     radius: float, dt: float, robin_m, robin_h, hm_eff, n: int, t_air_of,
                     c_air_of, props_frozen, rec_every: int, fine_every: int, rec: dict,
                     fine: dict, thresholds: dict):
    while True:
        if t_hard_stop is not None and diag["steps"] * dt >= t_hard_stop - 1e-9:
            break
        if tail_target is not None and diag["steps"] * dt >= tail_target - 1e-9:
            break
        step = diag["steps"] + 1
        t = step * dt
        t_air, c_air = t_air_of(t), c_air_of(t)
        T_old, C_old = T, C
        for _ in range(cfg.picard_iters):
            rho, cp, k_node = props_frozen if props_frozen is not None else properties(C)
            k_face = face_conductivity(k_node, cfg.face_k)
            T, margin_h = solve_tridiag(volume, dt, volume * rho * cp / dt,
                                         r_face * k_face / dr, robin_h, robin_h * t_air, T_old)
            if cfg.const_d is not None:
                d_face = np.full(n - 1, float(cfg.const_d))
            else:
                d_face = face_diffusivity(C, T, cfg.face_d)
            d_face = cfg.s_d * d_face
            C, margin_m = solve_tridiag(volume, dt, volume / dt,
                                         r_face * d_face / dr, robin_m, robin_m * c_air, C_old)
            diag["margin"] = min(diag["margin"], margin_h, margin_m)

        q_out += hm_eff * (C[-1] - c_air) * radius * dt
        w_now = float(C @ volume)
        diag["dw_max"] = max(diag["dw_max"], w_now - w_prev)
        w_prev = w_now
        diag["d_min"] = min(diag["d_min"], float(d_face.min()))
        diag["d_max"] = max(diag["d_max"], float(d_face.max()))
        diag["t_min"] = min(diag["t_min"], float(T.min()))
        diag["t_max"] = max(diag["t_max"], float(T.max()))
        diag["c_min"] = min(diag["c_min"], float(C.min()))
        diag["c_max"] = max(diag["c_max"], float(C.max()))
        diag["surf_margin"] = min(diag["surf_margin"], float(C[-1] - c_air))
        diag["steps"] = step

        cmax = float(C.max())
        for level in thresholds:
            if thresholds[level] is None and cmax < level <= cmax_prev:
                thresholds[level] = t - dt + (cmax_prev - level) / (cmax_prev - cmax) * dt
        cmax_prev = cmax
        if tail_target is None and thresholds[cfg.c_crit] is not None and cfg.t_end is None:
            tail_target = np.ceil(thresholds[cfg.c_crit] / cfg.rec_step) * cfg.rec_step

        if fine_every and step % fine_every == 0 and t <= cfg.fine_until + 1e-9:
            fine["t"].append(t); fine["T"].append(T.copy()); fine["C"].append(C.copy())
        if step % rec_every == 0:
            rec["t"].append(t); rec["T"].append(T.copy()); rec["C"].append(C.copy())
            rec["W"].append(w_now); rec["Q_out"].append(q_out)
            rec["eps"].append((w0 - w_now - q_out) / w0)
            rec["cmax"].append(cmax); rec["T_air"].append(t_air); rec["C_air"].append(c_air)
    return T, C, q_out, w_prev


def run_solver(air: dict, cfg: SolverConfig | None = None) -> dict:
    cfg = cfg or SolverConfig()
    grid = radial_grid(cfg.n_nodes, cfg.radius, cfg.stretch)
    dr, volume, r_face = grid["dr_face"], grid["volume"], grid["r_face"]
    radius, dt = grid["radius"], cfg.dt
    t_air_of, c_air_of = air_series(air, cfg)
    hm_eff, h_eff = HM_MASS * cfg.s_hm, H_CONV * cfg.s_h
    robin_m, robin_h = radius * hm_eff, radius * h_eff

    n = cfg.n_nodes
    T = np.full(n, cfg.t_init, dtype=float)
    C = np.full(n, cfg.c_init, dtype=float)
    props_frozen = properties(np.full(n, cfg.c_init)) if cfg.freeze_props else None

    rec_every = int(round(cfg.rec_step / dt))
    fine_every = int(round(cfg.fine_step / dt)) if cfg.fine_step else 0
    assert abs(rec_every * dt - cfg.rec_step) < 1e-12, "记录粒度必须是 dt 的整数倍"

    rec = {"t": [0.0], "T": [T.copy()], "C": [C.copy()], "W": [], "Q_out": [],
           "eps": [], "cmax": [float(C.max())], "T_air": [t_air_of(0.0)], "C_air": [c_air_of(0.0)]}
    fine = {"t": [], "T": [], "C": []}
    w0 = float(C @ volume)
    rec["W"].append(w0); rec["Q_out"].append(0.0); rec["eps"].append(0.0)

    q_out = 0.0
    cmax_prev = float(C.max())
    thresholds = dict.fromkeys((cfg.c_crit, *cfg.extra_crit))
    tail_target = None
    diag = {"margin": np.inf, "d_min": np.inf, "d_max": -np.inf, "surf_margin": np.inf,
            "dw_max": -np.inf, "t_min": np.inf, "t_max": -np.inf,
            "c_min": np.inf, "c_max": -np.inf, "steps": 0}
    w_prev = w0
    t_hard_stop = cfg.t_end if cfg.t_end is not None else cfg.t_max
    wall0 = time.perf_counter()

    T, C, q_out, w_prev = _advance_in_time(cfg, T, C, q_out, w_prev, w0, cmax_prev,
                                           tail_target, diag, t_hard_stop, dr, volume,
                                           r_face, radius, dt, robin_m, robin_h, hm_eff, n,
                                           t_air_of, c_air_of, props_frozen, rec_every,
                                           fine_every, rec, fine, thresholds)

    t_final = diag["steps"] * dt
    eps_mass = (w0 - w_prev - q_out) / w0
    t_star = thresholds[cfg.c_crit]
    return {"grid": grid, "cfg": cfg, "T": T, "C": C, "t_final": t_final,
            "coarse": {key: (np.array(val) if key != "t" else np.array(val)) for key, val in rec.items()},
            "fine": {key: np.array(val) for key, val in fine.items()},
            "W0": w0, "W_end": w_prev, "Q_out": q_out, "eps_mass": eps_mass,
            "t_star": t_star, "thresholds": dict(thresholds), "diag": diag,
            "dehydration": 1.0 - w_prev / w0, "wall_s": time.perf_counter() - wall0,
            "hit": t_star is not None}


def solve_case(key: str, cfg: SolverConfig, air: dict, *, force: bool = False) -> dict:
    print(f"[solve] {key}: N={cfg.n_nodes}, dt={cfg.dt}, face_d={cfg.face_d} ...", flush=True)
    sol = run_solver(air, cfg)
    print(f"[solve] {key} done in {sol['wall_s']:.1f} s: t*={sol['t_star']}, "
          f"eps={sol['eps_mass']:.3e}, steps={sol['diag']['steps']:.0f}", flush=True)
    return sol


def robin_cylinder_series(r: np.ndarray, t: float, radius: float, alpha: float,
                          biot: float, n_terms: int = 60) -> np.ndarray:
    zeros = np.concatenate(([0.0], jn_zeros(0, n_terms + 1)))
    lam = np.array([brentq(lambda x: x * j1(x) - biot * j0(x),
                           zeros[i] + 1e-12, zeros[i + 1] - 1e-12, xtol=1e-14, rtol=1e-15)
                    for i in range(n_terms)])
    coef = 2.0 * j1(lam) / (lam * (j0(lam) ** 2 + j1(lam) ** 2))
    shape = j0(np.outer(np.asarray(r, dtype=float) / radius, lam))
    decay = np.exp(-lam ** 2 * alpha * t / radius ** 2)
    return shape @ (coef * decay)


def radii_cm() -> np.ndarray:
    return np.round(np.arange(OUT_N_COLS) * OUT_DR_CM, 1)


def table_node_index() -> list[int]:
    step = int(round(0.5 / OUT_DR_CM))
    return list(range(0, OUT_N_COLS, step))


def profile_at(sol: dict, t: float, field: str = "C") -> np.ndarray:
    grid_t = sol["coarse"]["t"]
    data = sol["coarse"][field]
    idx = int(np.searchsorted(grid_t, t))
    if idx < grid_t.size and abs(grid_t[idx] - t) < 1e-9:
        return data[idx]
    assert 0 < idx < grid_t.size, f"t={t} 超出记录区间 [{grid_t[0]}, {grid_t[-1]}]"
    weight = (t - grid_t[idx - 1]) / (grid_t[idx] - grid_t[idx - 1])
    return (1.0 - weight) * data[idx - 1] + weight * data[idx]


def write_grid_xlsx(path: Path, sheets: dict) -> dict:
    book = openpyxl.Workbook(write_only=True)
    header = ["时间\\到药材中心的距离", *[float(x) for x in radii_cm()]]
    shapes = {}
    for name, (times, field) in sheets.items():
        sheet = book.create_sheet(title=name)
        sheet.append(header)
        for row_t, row in zip(times, field):
            sheet.append([float(row_t), *[round(float(v), ROUND_DIGITS) for v in row]])
        shapes[name] = (len(times), field.shape[1])
    path.parent.mkdir(parents=True, exist_ok=True)
    book.save(path)
    book.close()
    return {"path": str(path.relative_to(ROOT).as_posix()), "sheets": shapes,
            "sha256": sha256_of(path), "bytes": path.stat().st_size}


def write_csv(path: Path, header: list[str], rows) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [",".join(header)]
    lines += [",".join(f"{float(v):.10g}" for v in row) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"path": str(path.relative_to(ROOT).as_posix()), "rows": len(lines) - 1,
            "sha256": sha256_of(path)}


def dump_json(path: Path, payload: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False, sort_keys=False)
    path.write_text(text, encoding="utf-8")
    return {"path": str(path.relative_to(ROOT).as_posix()), "bytes": len(text.encode("utf-8"))}


def jsonable(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return [jsonable(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value
