from __future__ import annotations
import json
import math
import sys
import time
from pathlib import Path
import numpy as np
from scipy.linalg import solve_banded
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

_PROJ = Path(__file__).resolve()
ROOT = _PROJ.parents[1]
sys.path.insert(0, str(ROOT / "Q1"))

from solver_q1 import (
    solve_q1, build_grid, D_of_C, alpha, H, HM, K, RHO, CP, R, R_CM,
    T_INIT, C_INIT, interp_driver, surface_robin_terms, RHO_D,
    rho_v_sat, rho_v_inf,
)
from independent_check_q1_temp import (
    load_oven_temperature, duhamel_temperature,
)

T_REF_CENTER = 33.575344826514
T_REF_SURF = 36.785560351561
T_END = 1800.0
DR_CM = 0.01
DT_IN = 1.0 / 64

def _mass_step_buggy(C_old, r, dr, dt, t_new, d_bug_iters=30):
    N = len(r) - 1
    C = C_old.copy()
    rp = r + dr / 2.0
    rm = r - dr / 2.0
    geo_in = np.zeros(N + 1)
    geo_out = np.zeros(N + 1)
    geo_in[1:N] = dt * rm[1:N] / (r[1:N] * dr * dr)
    geo_out[1:N] = dt * rp[1:N] / (r[1:N] * dr * dr)
    c0_factor = 4.0 * dt / (dr * dr)
    C_air_new = interp_driver(t_new, "linear")[1]
    for _ in range(d_bug_iters):
        C_it = C.copy()
        Dv = D_of_C(np.maximum(C_it, 1e-12))
        dm = 0.5 * (Dv[:-1] + Dv[1:])
        d = np.zeros(N + 1)
        l = np.zeros(N + 1)
        u = np.zeros(N)
        d[1:N] = 1.0 + geo_in[1:N] * dm[0:N - 1] + geo_out[1:N] * dm[1:N]
        l[1:N] = -geo_in[1:N] * dm[0:N - 1]
        u[1:N] = -geo_out[1:N] * dm[1:N]
        a0 = c0_factor * dm[0]
        d[0], u[0] = 1.0 + a0, -a0
        m, a_in, a_out = surface_robin_terms(dr, dm[N - 1], HM)
        d[N] = m / dt + a_in + a_out
        l[N] = -a_in
        rhs = C_it.copy()
        rhs[N] = (m / dt) * C_it[N] + a_out * C_air_new
        ab = np.zeros((3, N + 1))
        ab[0, 1:] = u
        ab[1, :] = d
        ab[2, :-1] = l[1:]
        C = solve_banded((1, 1), ab, rhs)
    return C


def verify_picard_old_layer():
    r, dr, N = build_grid(DR_CM)
    dt = DT_IN
    few = solve_q1(case="A", dr_cm=DR_CM, dt=dt, t_end=50.0, dt_out=50.0, picard_max=2)
    many = solve_q1(case="A", dr_cm=DR_CM, dt=dt, t_end=50.0, dt_out=50.0, picard_max=50)
    dmax = float(np.max(np.abs(few["C_out"][-1] - many["C_out"][-1])))
    print(f"t=50 s 场：max|C(2 次迭代) − C（50 次迭代）| = {dmax:.3e} kg/kg")
    print(f"生产运行 Picard 统计：{many['picard']}")

    prod = solve_q1(case="A", dr_cm=DR_CM, dt=DT_IN, t_end=T_END, dt_out=T_END)
    C_surf_correct = float(prod["scalars"]["CN"][-1])
    C_bug = np.full(N + 1, C_INIT)
    dt_bug = 1.0
    n_steps = int(round(T_END / dt_bug))
    for k in range(1, n_steps + 1):
        C_bug = _mass_step_buggy(C_bug, r, dr, dt_bug, k * dt_bug, d_bug_iters=5)
    C_surf_bug = float(C_bug[-1])
    print(f"反例复现：表面 C（1800 s）正确 = {C_surf_correct:.6f}；"
          f"把上一迭代解当上一时间层 = {C_surf_bug:.6f} "
          f"（偏差 {100*(C_surf_bug-C_surf_correct)/C_surf_correct:+.2f}%）")
    return {"max_abs_diff_2_vs_50_iters": dmax,
            "pass": bool(dmax < 1e-9),
            "criterion": "2 次 vs 50 次迭代的场差 ≤1e-9 kg/kg",
            "picard_prod": prod["picard"],
            "counterexample_surface_C_correct": C_surf_correct,
            "counterexample_surface_C_buggy": C_surf_bug,
            "counterexample_rel_pct": 100 * (C_surf_bug - C_surf_correct) / C_surf_correct}


def axial_2d_links(dr_cm=0.05, dz_cm=0.5, L_CM=25.0):
    Nr = int(round(R_CM / dr_cm))
    dr = R / Nr
    Lh = L_CM / 2.0 / 100.0
    Nz = int(round(Lh / (dz_cm / 100.0)))
    dz = Lh / Nz
    r = np.arange(Nr + 1) * dr
    rp = np.minimum(r + dr / 2.0, R)
    rm = np.maximum(r - dr / 2.0, 0.0)
    r2d = rp ** 2 - rm ** 2
    dzj = np.full(Nz + 1, dz)
    dzj[0] = dz / 2.0
    dzj[Nz] = dz / 2.0
    nr, nz = Nr + 1, Nz + 1
    n = nr * nz

    def idx(i, j):
        return i * nz + j

    vol = np.empty(n)
    bcoef_geom = np.zeros(n)
    lp, lq, lAd = [], [], []
    for i in range(nr):
        for j in range(nz):
            p = idx(i, j)
            vol[p] = r2d[i] * dzj[j]
            if i > 0:
                lp.append(p); lq.append(idx(i - 1, j)); lAd.append(2.0 * rm[i] * dzj[j] / dr)
            if i < Nr:
                lp.append(p); lq.append(idx(i + 1, j)); lAd.append(2.0 * rp[i] * dzj[j] / dr)
            else:
                bcoef_geom[p] += 2.0 * R * dzj[j]
            if j > 0:
                lp.append(p); lq.append(idx(i, j - 1)); lAd.append(r2d[i] / dz)
            if j < Nz:
                lp.append(p); lq.append(idx(i, j + 1)); lAd.append(r2d[i] / dz)
            if j == 0:
                bcoef_geom[p] += r2d[i]
    return dict(r=r, z=np.arange(Nz + 1) * dz, dr=dr, dz=dz, Nr=Nr, Nz=Nz, nr=nr, nz=nz,
                n=n, idx=idx, vol=vol, bcoef_geom=bcoef_geom,
                lp=np.array(lp), lq=np.array(lq), lAd=np.array(lAd))


def assemble_axial_2d(G, links, bcoef, dt):
    Gl = 0.5 * (G[links["lp"]] + G[links["lq"]]) * links["lAd"]
    sumG = np.bincount(links["lp"], weights=Gl, minlength=links["n"])
    dod = 1.0 + (sumG + bcoef) * dt / links["vol"]
    off = -Gl * dt / links["vol"][links["lp"]]
    rows = np.concatenate([links["lp"], np.arange(links["n"])])
    cols = np.concatenate([links["lq"], np.arange(links["n"])])
    data = np.concatenate([off, dod])
    return coo_matrix((data, (rows, cols)), shape=(links["n"], links["n"])).tocsc()


def solve_axial_2d(dr_cm=0.05, dz_cm=0.5, dt=1.0 / 8, t_end=T_END, dt_out=1.0,
                   n_picard=2, L_CM=25.0, verbose=False):
    links = axial_2d_links(dr_cm, dz_cm, L_CM)
    n, vol, bcoef_geom = links["n"], links["vol"], links["bcoef_geom"]
    al = alpha()
    Ah = assemble_axial_2d(np.full(n, al), links, bcoef_geom * H / (RHO * CP), dt)
    lu_h = splu(Ah)
    bh_geom = bcoef_geom * (H / (RHO * CP)) * dt / vol

    T = np.full(n, T_INIT)
    C = np.full(n, C_INIT)
    n_steps = int(round(t_end / dt))
    gap = int(round(dt_out / dt))
    T_last = None
    C_last = None
    for step in range(1, n_steps + 1):
        t_new = step * dt
        T_air, C_air = interp_driver(t_new, "linear")
        T = lu_h.solve(T + bh_geom * T_air)
        C_old = C.copy()
        C_rhs = C_old + bcoef_geom * HM * dt / vol * C_air
        for _ in range(n_picard):
            Dv = D_of_C(np.maximum(C, 1e-12))
            Am = assemble_axial_2d(Dv, links, bcoef_geom * HM, dt)
            C = splu(Am).solve(C_rhs)
        if step % gap == 0:
            T_last = T.copy()
            C_last = C.copy()
    return dict(T=T_last.reshape(links["nr"], links["nz"]),
                C=C_last.reshape(links["nr"], links["nz"]), geom=links)


def conservation_pair(A, B):
    sc, scB = A["scalars"], B["scalars"]
    consA = float(abs((sc["mass"][0] - sc["mass"][-1]) - sc["cum_flux"][-1])
                  / (sc["mass"][0] - sc["mass"][-1]))
    consB = float(abs((scB["mass"][0] - scB["mass"][-1]) - scB["cum_flux"][-1])
                  / abs(scB["mass"][0] - scB["mass"][-1]))
    consB_abs = float(abs((scB["mass"][0] - scB["mass"][-1]) - scB["cum_flux"][-1]))
    consB_scaled = consB_abs / scB["mass"][0]
    print(f"P3  守恒校验：口径 A 相对误差 {consA:.3e}（判据 ≤1e-8） "
          f"{'PASS' if consA <= 1e-8 else 'FAIL'}")
    print(f"        口径 B 相对失水量误差 {consB:.3e}（失水量仅 {scB['mass'][0]-scB['mass'][-1]:.3e}，"
          f"该比值被机器舍入主导）")
    print(f"        口径 B 绝对闭合误差 {consB_abs:.3e}，相对初始总量 {consB_scaled:.3e}  "
          f"{'PASS' if consB_scaled <= 1e-8 else 'FAIL'}")
    return {"caseA_rel_err": consA,
            "caseB_rel_err_vs_loss": consB,
            "caseB_abs_err": consB_abs,
            "caseB_rel_err_vs_total_mass": consB_scaled,
            "criterion": "相对误差 ≤1e-8",
            "pass": bool(consA <= 1e-8 and consB_scaled <= 1e-8)}


def physical_checks(A, B, sc):
    print("S1–S10 物理自检")
    S = {}
    dT0 = float(np.max(np.abs(A["T_out"][0] - T_INIT)))
    dC0 = float(np.max(np.abs(A["C_out"][0] - C_INIT)))
    S["S1_t0_equals_init"] = {"max_dev_T": dT0, "max_dev_C": dC0,
                              "pass": bool(dT0 <= 1e-12 and dC0 <= 1e-12)}
    dT = np.diff(sc["T0"])
    S["S2_center_T_nondecreasing"] = {"min_step_delta": float(dT.min()),
                                      "n_negative_steps": int((dT < 0).sum()),
                                      "pass": bool(dT.min() >= 0.0)}
    diff3 = sc["TN"] - sc["T0"]
    S["S3_surface_T_ge_center_T"] = {"min_surface_minus_center": float(diff3.min()),
                                     "at_1800s": float(diff3[-1]),
                                     "pass": bool(diff3.min() >= -1e-12)}
    T_air_series = np.array([interp_driver(t, "linear")[0] for t in A["t_out"]])
    exc = float(np.max(A["T_out"] - T_air_series[:, None]))
    S["S4_below_oven_T"] = {"max_excess_C": exc, "criterion": 1e-9,
                            "pass": bool(exc <= 1e-9)}
    dm = np.diff(sc["mass"])
    S["S5_mass_strictly_decreasing"] = {"max_step_delta": float(dm.max()),
                                        "n_positive_steps": int((dm > 0).sum()),
                                        "mass_ratio_1800": float(sc["mass"][-1] / sc["mass"][0]),
                                        "pass": bool(dm.max() < 0.0)}
    dC_prof = np.diff(A["C_out"], axis=1)
    S["S6_C_profile_radially_nonincreasing"] = {
        "max_increase": float(dC_prof.max()), "pass": bool(dC_prof.max() <= 1e-12)}
    dT_prof = np.diff(A["T_out"], axis=1)
    S["S7_T_profile_radially_nondecreasing"] = {
        "min_increase": float(dT_prof.min()), "pass": bool(dT_prof.min() >= -1e-12)}
    dev8 = float(abs(A["C_out"][int(T_END)][0] - C_INIT))
    S["S8_center_moisture_static"] = {"abs_dev_1800": dev8, "criterion": 1e-3,
                                      "pass": bool(dev8 <= 1e-3)}
    t_air, T_oven = load_oven_temperature()
    i_out = np.arange(0, A["N"] + 1, 10)
    T_an = duhamel_temperature([T_END], i_out * DR_CM / 100.0, t_air, T_oven,
                               n_roots=320)[0]
    e9 = np.abs(A["T_out"][int(T_END)][i_out] - T_an)
    S["S9_vs_analytic"] = {"max_abs_err_C": float(e9.max()), "at_center": float(e9[0]),
                           "at_surface": float(e9[-1]), "criterion": 5e-5,
                           "pass": bool(e9.max() < 5e-5)}
    CsB = float(B["C_out"][int(T_END)][-1])
    devB = float(np.max(np.abs(B["C_out"][int(T_END)] - C_INIT)))
    S["S10_caseB_no_drying"] = {"surface_C_1800": CsB, "max_field_dev": devB,
                                "scoping_surface_ref": 2.549885,
                                "pass": bool(abs(CsB - 2.549885) < 1e-3 and devB < 1e-3)}
    for k, v in S.items():
        print(f"  {k:<40} {'PASS' if v['pass'] else 'FAIL'}  "
              + ", ".join(f"{kk}={vv:.6g}" for kk, vv in v.items()
                          if kk not in ("pass",) and isinstance(vv, (int, float))))
    return S, T_air_series


def case_AB_compare(A, B):
    print("对比验证：口径 A vs 口径 B")
    qA = HM * (A["C_out"][1:, -1]
               - np.array([interp_driver(t, "linear")[1] for t in A["t_out"][1:]]))
    TsB = B["T_out"][1:, -1]
    w_all = np.array([interp_driver(t, "linear")[1] for t in B["t_out"][1:]])
    Ta_all = np.array([interp_driver(t, "linear")[0] for t in B["t_out"][1:]])
    qB = HM * (rho_v_sat(TsB) - rho_v_inf(w_all, Ta_all)) / RHO_D
    sc, scB = A["scalars"], B["scalars"]
    cmpAB = {
        "surface_C_1800s": {"A": float(A["C_out"][int(T_END)][-1]),
                            "B": float(B["C_out"][int(T_END)][-1])},
        "center_C_1800s": {"A": float(A["C_out"][int(T_END)][0]),
                           "B": float(B["C_out"][int(T_END)][0])},
        "mass_ratio_1800s": {"A": float(sc["mass"][-1] / sc["mass"][0]),
                             "B": float(scB["mass"][-1] / scB["mass"][0])},
        "flux_kg_m2_s_1800s": {"A": float(qA[-1] * RHO_D), "B": float(qB[-1] * RHO_D)},
        "flux_ratio_A_over_B": float(abs(qA[-1] / qB[-1])),
        "rho_d_kg_m3": RHO_D,
    }
    print(f"  表面 C(1800 s)：A {cmpAB['surface_C_1800s']['A']:.6f} vs "
          f"B {cmpAB['surface_C_1800s']['B']:.9f}")
    print(f"  总水量比：A {cmpAB['mass_ratio_1800s']['A']:.6f} vs "
          f"B {cmpAB['mass_ratio_1800s']['B']:.9f}")
    print(f"  通量比 A/B = {cmpAB['flux_ratio_A_over_B']:.4e}")
    return cmpAB, qA, qB


def plane_2d_compare(A, T_air_series, quick):
    print("对比验证：1D vs 2D 含端面")
    if quick:
        print("quick：跳过 2D")
        return {"skipped": True}
    t0 = time.time()
    DT_2D = 1.0 / 4
    o2 = solve_axial_2d(dr_cm=0.05, dz_cm=0.5, dt=DT_2D, t_end=T_END, n_picard=2)
    Tm = o2["T"]
    C2 = o2["C"]
    g2 = o2["geom"]
    k_mid = g2["Nz"]
    o1 = solve_q1(case="A", dr_cm=0.05, dt=DT_2D, t_end=T_END, dt_out=T_END)
    T_1d_c = float(o1["T_out"][-1][0])
    C_1d_c = float(o1["C_out"][-1][0])
    d1 = float(abs(Tm[0, k_mid] - T_1d_c))
    d2 = float(Tm[0, 0] - T_1d_c)
    d3 = float(abs(C2[0, k_mid] - C_1d_c))
    zrow = Tm[0, :]
    d4 = float((zrow.max() - zrow.min()) / (T_air_series[-1] - T_INIT))
    cmp2d = {"dr_cm": 0.05, "dz_cm": 0.5, "Nz": g2["Nz"], "dt_s": DT_2D,
             "T_mid_center_1800": float(Tm[0, k_mid]),
             "T_end_center_1800": float(Tm[0, 0]),
             "T_1d_center_1800_same_mesh_and_dt": T_1d_c,
             "T_1d_center_1800_production": float(A["T_out"][int(T_END)][0]),
             "delta1_mid_plane_C": d1, "delta2_end_face_C": d2,
             "delta3_mid_plane_C_moisture": d3,
             "delta4_z_nonuniformity": d4,
             "criteria": {"delta1_le_0.5C": bool(d1 <= 0.5),
                          "delta3_le_5e-3": bool(d3 <= 5e-3)},
             "wall_s": time.time() - t0}
    print(f"  Δ1 中截面偏差 = {d1:.4f} ℃  {'PASS' if d1 <= 0.5 else 'FAIL'}")
    print(f"  Δ2 端面中心偏差 = {d2:.4f} ℃")
    print(f"  Δ3 中截面水分偏差 = {d3:.3e} kg/kg  {'PASS' if d3 <= 5e-3 else 'FAIL'}")
    print(f"  Δ4 z 向不均匀度 = {d4:.4f}; 耗时 {cmp2d['wall_s']:.1f} s")
    return cmp2d


def diffusivity_and_interp_compare(A, N):
    print("对比验证：常数 D vs D(C)；线性 vs 阶梯插值")
    D_CONST = 1.0 / 8
    oD = solve_q1(case="A", dr_cm=DR_CM, dt=D_CONST, t_end=T_END, dt_out=T_END,
                  D_CONST=float(D_of_C(C_INIT)))
    oS = solve_q1(case="A", dr_cm=DR_CM, dt=D_CONST, t_end=T_END, dt_out=T_END,
                  INTERP="step")
    oL = solve_q1(case="A", dr_cm=DR_CM, dt=D_CONST, t_end=T_END, dt_out=T_END)
    cmpDC = {"D_const_value": float(D_of_C(C_INIT)),
             "surface_C_1800_variableD": float(oL["C_out"][-1][-1]),
             "surface_C_1800_constD": float(oD["C_out"][-1][-1]),
             "surface_diff": float(oD["C_out"][-1][-1] - oL["C_out"][-1][-1]),
             "center_diff": float(oD["C_out"][-1][0] - oL["C_out"][-1][0]),
             "r1cm_diff": float(oD["C_out"][-1][N // 2] - oL["C_out"][-1][N // 2]),
             "mass_ratio_variableD": float(oL["scalars"]["mass"][-1] / oL["scalars"]["mass"][0]),
             "mass_ratio_constD": float(oD["scalars"]["mass"][-1] / oD["scalars"]["mass"][0])}
    cmpINT = {"linear_surface_C_1800": float(oL["C_out"][-1][-1]),
              "step_surface_C_1800": float(oS["C_out"][-1][-1]),
              "surface_diff": float(oS["C_out"][-1][-1] - oL["C_out"][-1][-1]),
              "linear_center_C_1800": float(oL["C_out"][-1][0]),
              "step_center_C_1800": float(oS["C_out"][-1][0]),
              "center_diff": float(oS["C_out"][-1][0] - oL["C_out"][-1][0]),
              "T_center_diff_1800": float(oS["T_out"][-1][0] - oL["T_out"][-1][0]),
              "T_surface_diff_1800": float(oS["T_out"][-1][-1] - oL["T_out"][-1][-1])}
    print(f"  常数 D：表面 {cmpDC['surface_C_1800_constD']:.6f} vs 变系数 "
          f"{cmpDC['surface_C_1800_variableD']:.6f}（差 {cmpDC['surface_diff']:+.6f}）")
    print(f"  阶梯插值：表面 {cmpINT['step_surface_C_1800']:.6f} vs 线性 "
          f"{cmpINT['linear_surface_C_1800']:.6f}（差 {cmpINT['surface_diff']:+.6f}）")
    return cmpDC, cmpINT


def driving_envelope(i_out, dt_scan):
    print("对比验证：驱动势包络 h_m × {0, 0.001, …, 1000, ∞}")
    env = []
    mults = [0.0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    for m in mults:
        o = solve_q1(case="A", dr_cm=DR_CM, dt=dt_scan, t_end=T_END, dt_out=T_END, HM=HM * m)
        env.append({"hm_mult": m, "hm_value": HM * m,
                    "C_1800": [float(v) for v in o["C_out"][-1][i_out]],
                    "mass_ratio": float(o["scalars"]["mass"][-1] / o["scalars"]["mass"][0])})
    o_inf = solve_q1(case="A", dr_cm=DR_CM, dt=dt_scan, t_end=T_END, dt_out=T_END,
                     HM_INF=True)
    env.append({"hm_mult": float("inf"), "hm_value": None,
                "C_1800": [float(v) for v in o_inf["C_out"][-1][i_out]],
                "mass_ratio": float(o_inf["scalars"]["mass"][-1] / o_inf["scalars"]["mass"][0])})
    for e in env:
        print(f"  h_m×{str(e['hm_mult']):<7} C(2cm,1800) = {e['C_1800'][-1]:.6f}"
              f"   C(1cm)={e['C_1800'][10]:.6f}  mass={e['mass_ratio']:.6f}")
    mon = all(env[i]["C_1800"][k] >= env[i + 1]["C_1800"][k] - 1e-12
              for i in range(len(env) - 1) for k in range(len(i_out)))
    return env, {
        "rows": env, "monotone_nonincreasing_in_hm": bool(mon),
        "acceptance": {"hm0_max_dev_from_init": float(np.max(np.abs(
            np.array(env[0]["C_1800"]) - C_INIT))),
            "hm_inf_surface_vs_Cair": float(abs(env[-1]["C_1800"][-1] - 0.03307))}}


def parameter_sensitivity(N, dt_scan):
    print("灵敏度：单参数（h、h_m、k、ρ、c₀）+ 多参数联合（h×h_m）")
    scans = {
        "h": ("H", [H * 0.5, H, H * 2.0]),
        "h_m": ("HM", [HM * 0.5, HM, HM * 2.0]),
        "k": ("K", [K * 0.5, K, K * 2.0]),
        "rho": ("RHO", [RHO * 0.5, RHO, RHO * 2.0]),
        "c0": ("C_INIT", [C_INIT * 0.9, C_INIT, C_INIT * 1.1]),
    }
    base_vals = {"H": H, "HM": HM, "K": K, "RHO": RHO, "C_INIT": C_INIT}
    sens = {}
    for name, (key, vals) in scans.items():
        rows = []
        for v in vals:
            o = solve_q1(case="A", dr_cm=DR_CM, dt=dt_scan, t_end=T_END, dt_out=T_END,
                         **{key: v})
            rows.append({"value": float(v), "mult": float(v / base_vals[key]),
                         "T_center_1800": float(o["T_out"][-1][0]),
                         "T_surface_1800": float(o["T_out"][-1][-1]),
                         "C_center_1800": float(o["C_out"][-1][0]),
                         "C_r1cm_1800": float(o["C_out"][-1][N // 2]),
                         "C_surface_1800": float(o["C_out"][-1][-1]),
                         "mass_ratio": float(o["scalars"]["mass"][-1] / o["scalars"]["mass"][0])})
        sens[name] = {"param_key": key, "rows": rows}
        print(f"  {name:<5} " + " | ".join(
            f"×{row['mult']:.2f}: C_s={row['C_surface_1800']:.4f} T_c={row['T_center_1800']:.4f}"
            for row in rows))

    o_rho = solve_q1(case="A", dr_cm=DR_CM, dt=dt_scan, t_end=T_END, dt_out=T_END,
                     RHO=976.4)
    sens["alpha_pm19pct"] = {
        "note": " ρ 换成附录3 的 ρ(C=2.55)=976.4  α 降 19.07%，仅影响温度场",
        "T_center_1800": float(o_rho["T_out"][-1][0]),
        "T_surface_1800": float(o_rho["T_out"][-1][-1]),
        "C_surface_1800": float(o_rho["C_out"][-1][-1]),
    }

    joint = []
    for hh in [H * 0.5, H, H * 2.0]:
        for mm in [HM * 0.5, HM, HM * 2.0]:
            o = solve_q1(case="A", dr_cm=DR_CM, dt=dt_scan, t_end=T_END, dt_out=T_END,
                         H=hh, HM=mm)
            joint.append({"h": float(hh), "hm": float(mm),
                          "T_center_1800": float(o["T_out"][-1][0]),
                          "C_surface_1800": float(o["C_out"][-1][-1]),
                          "mass_ratio": float(o["scalars"]["mass"][-1] / o["scalars"]["mass"][0])})
    sens["joint_h_x_hm"] = {"rows": joint}
    return sens, joint


def error_budget(conv):
    return {
        "temperature_time_discretization": {
            "min_combo": conv["table2_dt_sweep_temperature"]["min_combo_meeting_acceptance"],
            "order_p": conv["table2_dt_sweep_temperature"]["p_order_center"]},
        "temperature_space_discretization": {
            "p_order": conv["table1_dr_sweep_temperature"]["p_order_center"], },
        "moisture_surface_grid_uncertainty": {
            "p_order": conv["table4_dr_sweep_moisture_surface"]["p_order"],
            "richardson": conv["table4_dr_sweep_moisture_surface"]["richardson_surface_C"],
            "uncertainty": conv["table4_dr_sweep_moisture_surface"]["grid_uncertainty_vs_richardson"]},
        "model_error_latent_heat": {
            "note": "能量方程未耦合蒸发潜热  系统性高估表面温度；"
                    "潜热/显热比 ≈5.2~18.5",
            "ratio_range": [5.2, 18.5]},
    }


def write_artifacts(res, VER, sens, cmpAB, cmp2d, cmpDC, cmpINT):
    for fn, obj in [("q1_verification.json", VER),
                    ("q1_sensitivity.json", sens),
                    ("q1_compare.json", {"caseAB": cmpAB, "twoD": cmp2d,
                                         "D_const": cmpDC, "interp": cmpINT,
                                         "envelope": VER["compare_envelope_hm"]})]:
        (res / fn).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_figure_inputs(res, conv, sens, joint, env, cmpAB, i_out):
    figd = res / "fig_data"
    eb = [{"label": "Δr 空间（温度，Δt=1 s）",
           "x": [row["dr_cm"] for row in conv["table1_dr_sweep_temperature"]["rows"]],
           "err_center": [row["err_center"] for row in conv["table1_dr_sweep_temperature"]["rows"]],
           "err_surface": [row["err_surf"] for row in conv["table1_dr_sweep_temperature"]["rows"]]},
          {"label": "Δt 时间（温度，Δr=0.01 cm）",
           "x": [row["dt"] for row in conv["table2_dt_sweep_temperature"]["rows"]],
           "err_center": [row["err_center"] for row in conv["table2_dt_sweep_temperature"]["rows"]],
           "err_surface": [row["err_surf"] for row in conv["table2_dt_sweep_temperature"]["rows"]]},
          {"label": "Δr 空间（水分表面列，Δt=1/32 s）",
           "x": [row["dr_cm"] for row in conv["table4_dr_sweep_moisture_surface"]["rows"]],
           "value": [row["C_surf"] for row in conv["table4_dr_sweep_moisture_surface"]["rows"]]}]
    (figd / "error_bars.json").write_text(json.dumps(eb, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    with open(figd / "sensitivity.csv", "w", encoding="utf-8-sig") as f:
        f.write("param,mult,value,T_center_1800,T_surface_1800,C_center_1800,"
                "C_r1cm_1800,C_surface_1800,mass_ratio\n")
        for name, blk in sens.items():
            if name in ("joint_h_x_hm", "alpha_pm19pct"):
                continue
            for row in blk["rows"]:
                f.write(",".join([name, f"{row['mult']:.4f}", f"{row['value']:.6g}",
                                  f"{row['T_center_1800']:.9f}", f"{row['T_surface_1800']:.9f}",
                                  f"{row['C_center_1800']:.9f}", f"{row['C_r1cm_1800']:.9f}",
                                  f"{row['C_surface_1800']:.9f}", f"{row['mass_ratio']:.9f}"]) + "\n")
        for row in joint:
            f.write(",".join(["h_x_hm_joint", f"{row['h']:.6g}x{row['hm']:.6g}", "",
                              f"{row['T_center_1800']:.9f}", "", "", "",
                              f"{row['C_surface_1800']:.9f}", f"{row['mass_ratio']:.9f}"]) + "\n")
    with open(figd / "envelope_hm.csv", "w", encoding="utf-8-sig") as f:
        f.write("hm_mult,r_cm," + ",".join(f"C_1800_r{r:g}cm" for r in i_out * DR_CM * 100) + "\n")
        for e in env:
            f.write(f"{e['hm_mult']}," + ",".join(f"{v:.9f}" for v in e["C_1800"]) + "\n")
    (figd / "ab_compare.json").write_text(json.dumps(cmpAB, ensure_ascii=False, indent=2),
                                          encoding="utf-8")


def main():
    quick = "--quick" in sys.argv
    t_start = time.time()
    res = ROOT / "results"
    (res / "fig_data").mkdir(parents=True, exist_ok=True)
    VER = {}

    A = solve_q1(case="A", dr_cm=DR_CM, dt=DT_IN, t_end=T_END, dt_out=1.0)
    B = solve_q1(case="B", dr_cm=DR_CM, dt=DT_IN, t_end=T_END, dt_out=1.0)
    sc = A["scalars"]
    N = A["N"]
    i_out = np.arange(0, N + 1, 10)
    dt_scan = 1.0 / 8

    VER["P0_picard_frozen_old_layer"] = verify_picard_old_layer()
    VER["P3_conservation"] = conservation_pair(A, B)
    S, T_air_series = physical_checks(A, B, sc)
    VER["S_checks"] = S

    cmpAB, qA, qB = case_AB_compare(A, B)
    VER["compare_2D"] = plane_2d_compare(A, T_air_series, quick)

    cmpDC, cmpINT = diffusivity_and_interp_compare(A, N)
    VER["compare_D_const"] = cmpDC
    VER["compare_interp"] = cmpINT

    env, env_block = driving_envelope(i_out, dt_scan)
    VER["compare_envelope_hm"] = env_block

    sens, joint = parameter_sensitivity(N, dt_scan)
    VER["sensitivity"] = sens

    conv = json.loads((res / "q1_convergence.json").read_text(encoding="utf-8"))
    VER["error_analysis"] = error_budget(conv)

    write_artifacts(res, VER, sens, cmpAB, VER["compare_2D"], cmpDC, cmpINT)
    write_figure_inputs(res, conv, sens, joint, env, cmpAB, i_out)

    print(f"总耗时 {time.time() - t_start:.1f} s。已写 results/q1_verification.json 等")

if __name__ == "__main__":
    sys.exit(main())
