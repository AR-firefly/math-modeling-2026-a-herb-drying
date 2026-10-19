from __future__ import annotations
import json
import os
import sys
from pathlib import Path
import numpy as np
from scipy.special import j0, j1
from scipy.optimize import brentq

RHO = 820.0
CP = 2600.0
K = 0.36
H = 25.0
R = 0.02
T_INIT = 28.0
ALPHA = K / (RHO * CP)
BI = H * R / K

_BASE = Path(__file__).resolve().parents[1]
_DATA_DIR = Path(os.environ.get("MM_DATA_DIR", str(_BASE)))
ATTACH1 = _DATA_DIR / "附件" / "附件1.xlsx"
OUTJSON = _BASE / "results" / "independent_check_q1_temp.json"


def load_oven_temperature():
    import pandas as pd
    df = pd.read_excel(ATTACH1)
    t = df["时间"].to_numpy(dtype=float)
    T = df["温度"].to_numpy(dtype=float)
    return t, T


def bessel_roots(n_roots: int, bi: float) -> np.ndarray:
    def f(lam):
        return lam * j1(lam) - bi * j0(lam)

    roots = []
    lo = 1e-9
    step = 0.05
    lam = lo
    prev = f(lam)
    while len(roots) < n_roots and lam < 4000.0:
        lam_next = lam + step
        cur = f(lam_next)
        if prev == 0.0:
            roots.append(lam)
        elif prev * cur < 0.0:
            roots.append(brentq(f, lam, lam_next, xtol=1e-14, rtol=1e-15))
        lam, prev = lam_next, cur
        if len(roots) > 6:
            step = 0.5
    return np.array(roots[:n_roots])


def duhamel_temperature(t_query, r_query, t_air, T_air, n_roots=80):
    lam = bessel_roots(n_roots, BI)
    Cn = 2.0 * j1(lam) / (lam * (j0(lam) ** 2 + j1(lam) ** 2))
    beta = ALPHA * lam ** 2 / R ** 2

    slopes = np.diff(T_air) / np.diff(t_air)
    seg_starts = t_air[:-1]
    seg_ends = t_air[1:]

    t_query = np.asarray(t_query, dtype=float)
    r_query = np.asarray(r_query, dtype=float)

    I = np.zeros(n_roots)
    T_out = np.empty((t_query.size, r_query.size))
    seg = 0
    last_t = 0.0
    for i, t in enumerate(t_query):
        while seg < slopes.size and seg_ends[seg] <= t + 1e-12:
            dt = seg_ends[seg] - last_t
            if dt > 0:
                decay = np.exp(-beta * dt)
                I = I * decay + slopes[seg] * (1.0 - decay) / beta
            last_t = seg_ends[seg]
            seg += 1
        dt = t - last_t
        if dt > 1e-15 and seg < slopes.size:
            decay = np.exp(-beta * dt)
            I_t = I * decay + slopes[seg] * (1.0 - decay) / beta
        else:
            I_t = I
        T_air_t = np.interp(t, t_air, T_air)
        radial = j0(np.outer(lam, r_query) / R)
        T_out[i] = T_INIT + (T_air_t - T_INIT) - (Cn[:, None] * radial * I_t[:, None]).sum(axis=0)
    return T_out


def main():
    t_air, T_oven = load_oven_temperature()
    print(f"附件1: {t_air.size} 点, t∈[{t_air[0]:.0f},{t_air[-1]:.0f}] s")
    print(f"附件1 末行逐位: t={t_air[-1]!r}, T={T_oven[-1]!r}")
    print(f"alpha = {ALPHA!r} m^2/s,  Bi = {BI!r}")

    t_tab = np.array([100, 300, 600, 900, 1200, 1500, 1800], dtype=float)
    r_tab = np.array([0.0, 0.005, 0.01, 0.015, 0.02])

    T_tab = duhamel_temperature(t_tab, r_tab, t_air, T_oven)

    smoke = {"t": 1800.0, "r0": 33.5765, "rR": 36.7863, "T_air": 41.5130}

    print("\n解析解（表1 格式，单位 ℃）")
    header = "时间/s " + "".join(f"{r*100:>12.4f}" for r in r_tab)
    print(header)
    for i, t in enumerate(t_tab):
        print(f"{t:7.0f} " + "".join(f"{T_tab[i, j]:12.4f}" for j in range(r_tab.size)))

    print("\n与 smoke test 对比")
    a0, aR = T_tab[-1, 0], T_tab[-1, -1]
    print(f"  解析解  中心={a0:.10f}  表面={aR:.10f}")
    print(f"  数值解  中心={smoke['r0']:.10f}  表面={smoke['rR']:.10f}")
    print(f"  偏差    中心={abs(a0-smoke['r0']):.3e}  表面={abs(aR-smoke['rR']):.3e}")

    T0 = duhamel_temperature(np.array([0.0]), r_tab, t_air, T_oven)
    print("\n解析解自检")
    print(f"  t=0 全场 = {T0[0]}  (应全为 {T_INIT})  -> {'PASS' if np.allclose(T0[0], T_INIT, atol=1e-10) else 'FAIL'}")

    out = {
        "note": "独立解析验算，不引用solver_q1.py",
        "params": {"rho": RHO, "cp": CP, "k": K, "h": H, "R": R, "T_init": T_INIT,
                   "alpha": ALPHA, "Bi": BI},
        "attach1_last_row": {"t": float(t_air[-1]), "T": float(T_oven[-1])},
        "table1_like": {"times_s": t_tab.tolist(), "radii_cm": (r_tab * 100).tolist(),
                        "T_C": T_tab.tolist()},
        "vs_smoke_at_1800": {"analytic_center": a0, "analytic_surface": aR,
                             "numeric_center_reported": smoke["r0"],
                             "numeric_surface_reported": smoke["rR"],
                             "abs_diff_center": abs(a0 - smoke["r0"]),
                             "abs_diff_surface": abs(aR - smoke["rR"])},
    }
    OUTJSON.parent.mkdir(parents=True, exist_ok=True)
    OUTJSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写出 {OUTJSON}")


if __name__ == "__main__":
    sys.exit(main())
