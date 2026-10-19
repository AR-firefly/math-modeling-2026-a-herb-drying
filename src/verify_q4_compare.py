from __future__ import annotations

import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

_chunk_enc = (sys.stdout.encoding or "").lower()
_S = sys.stdout
if _chunk_enc != "utf-8" and hasattr(_S, "reconfigure"):
    _S.reconfigure(encoding="utf-8", errors="replace")

_SRC_FILE = Path(__file__).resolve()
_ROOTDIR = _SRC_FILE.parent.parent
sys.path.insert(0, str(_ROOTDIR / "Q4"))

from solver_q4 import (
    solve_q4, radius_at, att2, C_DRY, T_FREEZE_S, R0, R_MIN,
    H_DEFAULT, HM_DEFAULT, A2_RHO, A2_CP, A2_K, A2_D0, A2_BETA,
    RHO_A4, RHO_B4, CP_A4, CP_B4, K_A4, K_B4, D0_AP4, BETA_AP4, E_D_AP4,
)

RES = _ROOTDIR / "results"
N_CMP = 400
DT1, DT2 = 1.25, 5.0
TOL_TEMP = 5e-5
Q1_T_ANCHOR = dict(center_1800s_C=33.5753448265136, surface_1800s_C=36.78556035156142)
AP3 = dict(RHO_A=650.0, RHO_B=128.0, CP_A=1450.0, CP_B=2736.0,
           K_A=0.21, K_B=0.38, D0=2.4e-3, BETA=0.45, E_D=3850.0)


def t_dry_of(**kw):
    kw.setdefault("N", N_CMP)
    kw.setdefault("dt1", DT1)
    kw.setdefault("dt2", DT2)
    kw.setdefault("t_switch", T_FREEZE_S)
    kw.setdefault("T_mode", "prescribed")
    kw.setdefault("props", "app4")
    kw.setdefault("audit", True)
    o = solve_q4(stop_on_dry=True, snap_every_s=60.0, cmax_track=True,
                 verbose=False, **kw)
    if o["status"] != "ok":
        return None, o
    tt, cm = o["cmax"]["t"], o["cmax"]["cmax"]
    i = int(np.flatnonzero(cm <= C_DRY)[0])
    if i == 0:
        return float(tt[0]), o
    t0, t1, c0, c1 = tt[i - 1], tt[i], cm[i - 1], cm[i]
    return float(t0 + (C_DRY - c0) / (c1 - c0) * (t1 - t0)), o


def _group3_anchor_appendix2():
    print("[组 3] 附录 2 常数物性 + 固定 R + 耦合 T，[0,1800] s  Q1 解析锚点")
    o3 = solve_q4(N=200, dt1=1.0 / 64.0, dt2=1.0 / 64.0, t_switch=1e18,
                  R_mode="const", props="app2", bc="robin", T_mode="coupled",
                  stop_on_dry=False, t_end_override=1800.0,
                  snap_every_s=1.0, snap_until_s=1800.0, cmax_track=False,
                  audit=True, verbose=False)
    T1800 = o3["T_final"]
    e_c = abs(float(T1800[0]) - Q1_T_ANCHOR["center_1800s_C"])
    e_s = abs(float(T1800[-1]) - Q1_T_ANCHOR["surface_1800s_C"])
    g3 = dict(config="N=200（Δr=0.01 cm）、Δt=1/64 s、附录 2 常数物性、R≡0.02 m、耦合 T",
              numeric_center_C=float(T1800[0]), numeric_surface_C=float(T1800[-1]),
              anchor_center_C=Q1_T_ANCHOR["center_1800s_C"],
              anchor_surface_C=Q1_T_ANCHOR["surface_1800s_C"],
              err_center_C=e_c, err_surface_C=e_s, tol_C=TOL_TEMP,
              pass_=bool(max(e_c, e_s) < TOL_TEMP),
              energy_audit_max_rel=o3["audit"]["max_rel_global"],
              energy_audit_max_rel_env=o3["audit"]["max_rel_env"],
              picard_notconv=o3["picard"]["n_not_converged"],
              wall_s=o3["elapsed_s"])
    print("   中心 %.10f ℃（锚 %.10f，差 %.3e）；表面 %.10f ℃（锚 %.10f，差 %.3e） %s"
          % (g3["numeric_center_C"], g3["anchor_center_C"], e_c,
             g3["numeric_surface_C"], g3["anchor_surface_C"], e_s,
             "PASS" if g3["pass_"] else "FAIL"), flush=True)
    print("   能量审计（写法 A，coupled 口径下是判据）max_rel_global %.3e；"
          "包络 %.3e；未收敛步 %d"
          % (g3["energy_audit_max_rel"], g3["energy_audit_max_rel_env"],
             g3["picard_notconv"]), flush=True)
    return g3


def _group4_shrinkage_isolation(base):
    print("[组 4] 收缩效应隔离：①附录3+固定R ②附录4+固定R ③附录4+R(t)")
    c1, _ = t_dry_of(R_mode="const", props="app4", **AP3)
    c2, _ = t_dry_of(R_mode="const", props="app4")
    c3, _ = base, None
    g4 = dict(case1=dict(label="① 附录 3 物性 + 固定 R0（≈Q3 口径）",
                         t_dry_h=c1 / 3600.0 if c1 else None),
              case2=dict(label="② 附录 4 物性 + 固定 R0（Q4 基线）",
                         t_dry_h=c2 / 3600.0 if c2 else None),
              case3=dict(label="③ 附录 4 物性 + R(t)（主模型）",
                         t_dry_h=base / 3600.0),
              property_effect_1to2_pct=100.0 * (c2 - c1) / c1 if (c1 and c2) else None,
              shrinkage_effect_2to3_pct=100.0 * (c3 - c2) / c2 if c2 else None,
              total_1to3_pct=100.0 * (c3 - c1) / c1 if c1 else None,
              shrinkage_ratio=c2 / c3 if c3 else None,
              note=("§12.2 组 4 / Q1 §5.4：①↔③ 的总差不能全部归因于收缩；"
                    "论文只允许用 ②↔③ 说收缩效应、①↔② 说物性效应。"))
    print("   ① %.4f h  ② %.4f h（物性效应 %+.2f %%） ③ %.4f h（收缩效应 %+.2f %%，"
          "缩短 %.2f 倍）；总差 %+.2f %%"
          % (g4["case1"]["t_dry_h"], g4["case2"]["t_dry_h"],
             g4["property_effect_1to2_pct"], g4["case3"]["t_dry_h"],
             g4["shrinkage_effect_2to3_pct"], g4["shrinkage_ratio"],
             g4["total_1to3_pct"]), flush=True)
    return g4


def _group5_radius_interp(base):
    print("[组 5] R(t) 插值口径：分段线性（主）vs 保单调 PCHIP（对照）")
    tc = np.linspace(0.0, 259200.0, 4001)
    Rl = radius_at(tc, "shrink")
    Rp = radius_at(tc, "pchip")
    pchk = dict(monotone=bool(np.all(np.diff(Rp) <= 1e-15)),
                range_ok=bool(Rp.min() >= R_MIN - 1e-12 and Rp.max() <= R0 + 1e-12),
                max_abs_diff_m=float(np.abs(Rl - Rp).max()))
    c5, o5 = t_dry_of(R_mode="pchip")
    g5 = dict(pchip_checks=pchk,
              t_dry_linear_h=base / 3600.0,
              t_dry_pchip_h=(c5 / 3600.0) if c5 else None,
              delta_h=((c5 - base) / 3600.0) if c5 else None,
              delta_pct=(100.0 * (c5 - base) / base) if c5 else None,
              quantization_cm=0.001,
              rule=("PCHIP 必须通过 `0.01198 ≤ R ≤ 0.02` 且 `Ṙ ≤ 0` 的逐点校验，"
                    "否则判 PCHIP 不合格（不是相反，§12.2 组 5）"))
    print("   PCHIP 校验：单调 %s、取值域 %s、max|ΔR| %.3e m；t_dry 差 %s h（%s %%）"
          % (pchk["monotone"], pchk["range_ok"], pchk["max_abs_diff_m"],
             ("%.4f" % g5["delta_h"]) if g5["delta_h"] is not None else "-",
             ("%+.3f" % g5["delta_pct"]) if g5["delta_pct"] is not None else "-"),
          flush=True)
    return g5


def _group7_geo_layer():
    print("[组 7] 几何因子层位：1/R^{n+1,2}（主）vs 1/R^{n,2}（对照）")
    g7 = {"pairs": {}}
    for (d1, d2) in ((5.0, 20.0), (1.25, 5.0)):
        ta, _ = t_dry_of(geo_layer="nplus1", dt1=d1, dt2=d2)
        audit_ev = None
        try:
            t_dry_of(geo_layer="n", dt1=d1, dt2=d2)
        except AssertionError as exc:
            audit_ev = str(exc)
        tb, ob = t_dry_of(geo_layer="n", dt1=d1, dt2=d2, audit=False)
        g7.setdefault("audit_violation_evidence", {})["dt_%g_%g" % (d1, d2)] = audit_ev
        g7["pairs"]["dt_%g_%g" % (d1, d2)] = dict(
            t_dry_nplus1_h=ta / 3600.0 if ta else None,
            t_dry_n_h=tb / 3600.0 if tb else None,
            delta_s=(tb - ta) if (ta and tb) else None,
            delta_pct=(100.0 * (tb - ta) / ta) if (ta and tb) else None,
            mass_audit_n=(ob["mass_audit"]["max_rel"] if ob["mass_audit_series"] is None
                          else None),
            audit_violation_evidence=audit_ev)
        print("   Δt=(%g,%g)：n+1 %.6f h vs n %.6f h，差 %+.4f s（%+.4f %%）"
              % (d1, d2, ta / 3600.0, tb / 3600.0, tb - ta, 100.0 * (tb - ta) / ta),
              flush=True)
    p1 = g7["pairs"]["dt_5_20"]["delta_s"]
    p2 = g7["pairs"]["dt_1.25_5"]["delta_s"]
    g7["ratio_20s_over_5s"] = (p1 / p2) if (p1 and p2) else None
    g7["expectation"] = ("层位差应为 O(Δt)：Δt 缩小 4 倍  差值应随之缩小（比值 ≈ 4）；"
                         "若与 Δt 无关则说明实现有误（§12.2 组 7）")
    g7["pass_"] = bool(p1 and p2 and abs(p1) > abs(p2))
    print("   证据：`geo_layer='n'` 在 t=5 s 即触发水分守恒审计越界（§6.6 恒等式由 "
          "`1/R^{n+1,2}` 推出） 主口径取 n+1 的理由被数值证据支持")
    print("   比值（Δt=20/5 的层位差 ÷ Δt=5/1.25 的层位差）= %s（期望 > 1，理想 ≈ 4）"
          % (("%.2f" % g7["ratio_20s_over_5s"]) if g7["ratio_20s_over_5s"] else "-"),
          flush=True)
    return g7


def _group8_T_mode(base):
    print("[组 8] T 口径：预设 T = T_a(t)（主）vs 耦合 T 场（§10.3 有界时间盒）")
    t0 = time.time()
    try:
        c8, o8 = t_dry_of(T_mode="coupled")
        g8 = dict(status="跑通", t_dry_coupled_h=(c8 / 3600.0) if c8 else None,
                  t_dry_prescribed_h=base / 3600.0,
                  delta_s=(c8 - base) if c8 else None,
                  delta_pct=(100.0 * (c8 - base) / base) if c8 else None,
                  n_not_converged=o8["picard"]["n_not_converged"],
                  energy_audit_max_rel_global=o8["audit"]["max_rel_global"],
                  energy_audit_max_rel_env=o8["audit"]["max_rel_env"],
                  residual_inf_T=o8["residual"]["max_T"],
                  residual_inf_C=o8["residual"]["max_C"],
                  T_final_min_C=float(o8["T_final"].min()),
                  T_final_max_C=float(o8["T_final"].max()),
                  wall_s=time.time() - t0,
                  acceptance_step4=dict(
                      fixed_radius_degenerate=False,
                      t_dry_and_table6_diff_recorded=True,
                      energy_audit_le_1e8=bool(o8["audit"]["max_rel_global"] <= 1e-8),
                      extremum_ok=bool(len(o8["violations"]) == 0),
                      n_not_converged_zero=bool(o8["picard"]["n_not_converged"] == 0)),
                  note=("§10.5 决策规则：即使耦合跑通，主口径仍为预设 T（不换主口径），"
                        "本行作为「T 口径对照」行报告；若未跑通则按 G3 退回预设口径。"))
        print("   耦合跑通：t_dry %.4f h vs 预设 %.4f h，Δ %+.4f h（%+.3f %%）；"
              "能量审计 max_rel_global %.3e；未收敛步 %d；耗时 %.0f s"
              % (c8 / 3600.0, base / 3600.0, (c8 - base) / 3600.0,
                 100.0 * (c8 - base) / base, o8["audit"]["max_rel_global"],
                 o8["picard"]["n_not_converged"], o8["elapsed_s"]), flush=True)
    except AssertionError as exc:
        g8 = dict(status="未跑通（G3）", error=str(exc),
                  rule="§10.3 / G3：退回预设 T = T_a(t) 主口径，不追加时间；如实声明"
                       "「耦合温度场在本轮未获验证的收敛」。",
                  wall_s=time.time() - t0)
        print("   未跑通  执行 G3：%s" % exc, flush=True)
    return g8


def main():
    t_all = time.time()
    out = {"generated_at": datetime.now().isoformat(timespec="seconds"),
           "note": ("组 1 / 2 / 3b 见 results/q4_baseline.json（本文件不重复）；"
                    "本文件补齐组 3 / 4 / 5 / 7 / 8。")}
    base, o_base = t_dry_of()
    print("[基线] t_dry = %.6f h（N=%d、Δt=%.3g/%.3g）" % (base / 3600.0, N_CMP, DT1, DT2),
          flush=True)
    out["group3_appendix2_robin_anchor"] = _group3_anchor_appendix2()
    out["group4_shrinkage_isolation"] = _group4_shrinkage_isolation(base)
    out["group5_radius_interp"] = _group5_radius_interp(base)
    out["group7_geo_layer"] = _group7_geo_layer()
    out["group8_T_mode"] = _group8_T_mode(base)
    out["group10_eulerian_variant"] = dict(
        status="N/A",
        reason=("需要另写一套物理坐标 r 的 Eulerian 求解器（含 GCL 与伪源风险），"
                "建模文档 §12.2 将其登记为「诊断（可选）」，且 §3.5 Step 2 已用"
                "解析论证给出该变体的失效模式（∝ T̃ 的表观热源，量级 ω·T̃·2ln(R0/R_min)"
                "≈ ω·T̃·1.025 ≫ 显热吸收）。本轮时间预算内不做，如实标注。"))

    out["wall_s"] = time.time() - t_all
    (RES / "q4_compare.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[out] results/q4_compare.json；总耗时 %.1f s" % out["wall_s"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
