# -*- coding: utf-8 -*-
"""
q2_case_runner.py —— Q2 检验算例的**单例执行器**（每个进程只跑一个算例）

用途：五类检验需要几十个独立算例（网格/时间步加密、界面系数、能量形式、灵敏度、
外推情景…）。每个算例都是**确定性的独立 PDE 解**，因此用「一个算例一个进程」的方式
并行（`subprocess` + 有限并发），避免在同一个 numpy 进程里串行等几小时。

用法
----
    python src/q2_case_runner.py <case_id> [--outdir results/q2_parts]

产物（每个算例一对）
----
    <outdir>/<case_id>.npz    t_1s、T_21、C_21（21 个交付半径，1 s，全精度）
                              + 可选 cmax_t / cmax_v / cmax_c0 / cmax_cR / cmax_cmin
                              + 可选 prof_t / prof_r_cm / prof_T / prof_C（全场剖面）
    <outdir>/<case_id>.json   配置、标量结果、Picard/floor/审计/极值/墙钟

算例清单见 `CASES`（key = case_id，value = build_kwargs() 或 None=未知）。
所有算例的输出都对齐到**同一条 21 点交付半径轴**与**同一套输出时刻**，便于逐点比对。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

if getattr(sys.stdout, "reconfigure", None):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from solver_q2 import (                                    # noqa: E402
    solve_q2, node_index_for, driver_table, D_FLOOR, C_DRY, T_FREEZE_S,
)

R_OUT_CM = np.arange(0.0, 2.0 + 1e-9, 0.1)                 # 21 个交付半径
T_SWEEP_END = 10800.0                                      # 3 h 交付窗口
PROF_TIMES = np.array([1800.0, 3600.0, 7200.0, 10800.0])


# ============================================================================
# 算例定义
# ============================================================================
def _base(dr_cm=0.01, dt=1.0 / 64, t_end=T_SWEEP_END):
    return dict(dr_cm=dr_cm, dt1=dt, dt2=dt, t_switch=1e18,
                stop_on_dry=False, t_end_override=t_end,
                snap_every_s=1.0, snap_until_s=t_end, snap_end=t_end,
                cmax_track=False, verbose=False)


def build_case(cid):
    """返回 (kwargs, 说明)。kwargs 直接喂给 `solve_q2`。"""
    # ---- 时间步加密序列（Δt_I）----
    if cid.startswith("dtI_"):
        lab = cid.split("_")[1]
        dt = 1.0 / float(lab)
        return _base(dt=dt), "Δt_I=1/%s s 单段（t_switch 关闭），Δr=0.01 cm，[0,10800] s" % lab

    # ---- 空间网格加密序列 ----
    if cid.startswith("dr_"):
        lab = cid.split("_")[1]
        # 命名：dr_100→0.02 cm, dr_200→0.01, dr_400→0.005, dr_800→0.0025
        dr = {"100": 0.02, "200": 0.01, "400": 0.005, "800": 0.0025}[lab]
        kw = _base(dr_cm=dr)
        kw["profile_times"] = tuple(PROF_TIMES)
        return kw, "Δr=%g cm（N=%s），Δt=1/64 s，[0,10800] s" % (dr, lab)

    # ---- 界面系数 × 网格 ----
    if cid.startswith("face_"):
        _, rule, lab = cid.split("_")
        dr = {"100": 0.02, "200": 0.01, "400": 0.005, "800": 0.0025}[lab]
        kw = _base(dr_cm=dr)
        kw["face_rule"] = rule
        return kw, "界面系数 %s，Δr=%g cm（N=%s）" % (rule, dr, lab)

    # ---- 能量形式 ----
    if cid.startswith("energy_"):
        form = cid.split("_", 1)[1]
        return (_base() | dict(energy_form=form),
                "能量形式 %s（capacity 主口径；product_0C/product_K 为乘积形式两档 T_ref）" % form)

    # ---- 水分形式替代（对比 3）全时长：∂(ρC)/∂t 形式的 t_dry（**必须先于通用 moist_**）----
    if cid == "moist_rhoC_full":
        return (dict(dr_cm=0.01, dt1=1.0 / 16, dt2=1.0 / 2, t_switch=T_FREEZE_S,
                     stop_on_dry=True, snap_every_s=60.0, snap_until_s=T_SWEEP_END,
                     cmax_track=True, verbose=False, moisture_form="rhoC"),
                "水分形式替代 ∂(ρC)/∂t（全时长，Δt_I=1/16、Δt_II=1/2 灵敏度口径）")

    # ---- 水分形式 ----
    if cid.startswith("moist_"):
        form = cid.split("_", 1)[1]
        return (_base() | dict(moisture_form=form), "水分形式 %s" % form)

    # ---- 物性口径（附录 2 退化回归，整段 3 h）----
    if cid == "props_ap2_3h":
        return (_base() | dict(props_model="appendix2"),
                "附录 2 常数物性（退化回归口径），整段 3 h")

    # ---- 退化回归（对比 1）：附录 2 物性 + [0,1800] s，复现 Q1 三锚点 ----
    if cid == "degenerate_q1":
        return (dict(dr_cm=0.01, dt1=1.0 / 64, dt2=1.0 / 64, t_switch=1e18,
                     stop_on_dry=False, t_end_override=1800.0,
                     snap_every_s=1.0, snap_until_s=1800.0, snap_end=1800.0,
                     cmax_track=False, verbose=False, props_model="appendix2"),
                "退化回归（对比 1）：附录 2 常数物性（ρ=820, c_p=2600, k=0.36, "
                "D=7e-9·e^(−0.89/C)），[0,1800] s，Δt=1/64 s —— 复现 Q1 三锚点")


    # ---- 附件 1 外推情景（全时长，含 C_max 轨迹）----
    if cid.startswith("scen_"):
        body = cid[5:]
        ta, ca = body.split("__")
        return (dict(dr_cm=0.01, dt1=1.0 / 16, dt2=1.0 / 2, t_switch=T_FREEZE_S,
                     stop_on_dry=True, snap_every_s=60.0, snap_until_s=T_SWEEP_END,
                     cmax_track=True, verbose=False,
                     ta_scenario=ta, ca_scenario=ca),
                "外推情景 T_a=%s, C_a=%s（Δt_I=1/16, Δt_II=1/2：灵敏度口径，见文档）" % (ta, ca))

    # ---- 灵敏度单参数 ----
    if cid.startswith("sens_"):
        body = cid[5:]
        name, raw = body.split("__")
        val = raw if name == "hmD0" else float(raw)
        kw = dict(dr_cm=0.01, dt1=1.0 / 16, dt2=1.0 / 2, t_switch=T_FREEZE_S,
                  stop_on_dry=True, snap_every_s=60.0, snap_until_s=T_SWEEP_END,
                  cmax_track=True, verbose=False)
        from solver_q2 import (H_DEFAULT, HM_DEFAULT, D0_AP3, BETA_AP3, E_D_AP3,
                               RHO_A, RHO_B, CP_A, CP_B)
        if name == "h":
            kw["H"] = H_DEFAULT * val
        elif name == "hm":
            kw["HM"] = HM_DEFAULT * val
        elif name == "D0":
            kw["D0"] = D0_AP3 * val
        elif name == "beta":
            kw["BETA"] = val                      # 绝对值 0.40/0.45/0.50（±10 %）
        elif name == "ED":
            kw["E_D"] = val                       # 绝对值 3657.5/3850/4042.5（±5 %）
        elif name == "omega":
            kw["RHO_SCALE"] = val                 # ρ(C) 整体 ×val（§10.1「ρc_p 系数 ±10%」）
        elif name == "cp":
            kw["CP_SCALE"] = val                  # c_p(C) 整体 ×val
        elif name == "hmD0":                      # 组合 hm × D0（3×3 的 off-diagonal 角）
            v_hm, v_d0 = (float(x) for x in raw.split("x"))
            kw["HM"] = HM_DEFAULT * v_hm
            kw["D0"] = D0_AP3 * v_d0
        else:
            raise ValueError("未知灵敏度参数 %r" % name)
        return kw, "灵敏度 %s = %s（Δt_I=1/16, Δt_II=1/2：灵敏度口径）" % (name, raw)

    # ---- Δt_II 加密序列（从 t=14400 s 检查点重启）----
    if cid.startswith("dtII_"):
        lab = cid.split("_")[1]
        dt2 = 1.0 / float(lab)
        # 只跑到 t=28800 s：Δt_II 的判据窗口就定义在 [14400, 28800]（II 段前 4 h）。
        # 不跑到 t_dry 是**有意的**——Δt 不改变 t_dry（§7.4：决定 t_dry 的是场误差，
        # 而场误差由网格/模型决定），跑到尾段只会把算例成本放大 20 倍。
        return (dict(dr_cm=0.01, dt1=1.0 / 64, dt2=dt2, t_switch=T_FREEZE_S,
                     stop_on_dry=False, t_end_override=28800.0,
                     snap_every_s=60.0, snap_until_s=28800.0,
                     cmax_track=True, verbose=False, restart_from_ckpt=14400.0),
                "Δt_II=1/%s s（自 t=14400 s 检查点重启，Δt_I 沿用生产 1/64 s；"
                "只跑到 t=28800 s = II 段前 4 h，即判据窗口）" % lab)

    # ---- 尾部网格误差（t_dry 处的网格不确定度，必须实测而非用 [0,10800] 代理）----
    if cid.startswith("tailgrid_"):
        lab = cid.split("_")[1]
        dr = {"100": 0.02, "200": 0.01, "400": 0.005}[lab]
        return (dict(dr_cm=dr, dt1=1.0 / 16, dt2=1.0 / 2, t_switch=T_FREEZE_S,
                     stop_on_dry=True, snap_every_s=60.0, snap_until_s=T_SWEEP_END,
                     cmax_track=True, verbose=False),
                "尾部网格误差：Δr=%g cm（N=%s），Δt 与灵敏度基线**逐项相同**（1/16、1/2），"
                "跑满到 t_dry + 3 h → 与 sens_h__1.0 的 C_max 轨迹可直接比" % (dr, lab))

    # ---- 数据误差：附件 1 的读取方式（线性插值 vs 零阶保持）----
    if cid == "interp_step":
        return (_base() | dict(INTERP="step"),
                "附件 1 零阶保持（左端值）读取，[0,10800] s —— 数据误差对照项")

    # ---- 单段基准（全程 Δt=1/64 s，跑 [0, 28800]）----
    if cid.startswith("single_seg"):
        t_end = 28800.0
        return (dict(dr_cm=0.01, dt1=1.0 / 64, dt2=1.0 / 64, t_switch=1e18,
                     stop_on_dry=False, t_end_override=t_end,
                     snap_every_s=60.0, snap_until_s=t_end, snap_end=t_end,
                     cmax_track=True, verbose=False),
                "单段基准：全程 Δt=1/64 s，[0,%g] s（跨接伪影对照用）" % t_end)

    raise ValueError("未知算例 %r" % cid)


def load_checkpoint(t_s=14400.0):
    p = ROOT / "results" / ("q2_checkpoint_t%ds.npz" % int(round(t_s)))
    if not p.is_file():
        raise SystemExit("检查点不存在：%s（需先跑 src/run_q2.py）" % p)
    z = np.load(p)
    return dict(t=float(z["t_s"][0]), T=z["T_C"], C=z["C_kgkg"],
                cum_flux=float(z["cum_flux"][0]), mass=float(z["mass"][0]),
                n_steps=int(z["n_steps"][0]))


def run_case(cid, outdir):
    kw, note = build_case(cid)
    restart = kw.pop("restart_from_ckpt", None)
    if restart is not None:
        kw["t_start_state"] = load_checkpoint(restart)
    dr_cm = kw["dr_cm"]
    i_out = np.array([node_index_for(rc, dr_cm) for rc in R_OUT_CM])
    kw["snap_nodes"] = i_out
    t0 = time.time()
    o = solve_q2(**kw)
    wall = time.time() - t0

    outdir.mkdir(parents=True, exist_ok=True)
    tt = o["snap_t"]
    T21 = o["snap_T"]
    C21 = o["snap_C"]
    arrs = dict(t_s=tt, T_21=T21, C_21=C21, r_out_cm=R_OUT_CM)
    if o["cmax"]["t"] is not None:
        arrs.update(cmax_t=o["cmax"]["t"], cmax_v=o["cmax"]["cmax"],
                    cmax_c0=o["cmax"]["c0"], cmax_cR=o["cmax"]["cR"],
                    cmax_cmin=o["cmax"]["cmin"])
    # 全场剖面（仅网格算例请求；4 个时刻，全场节点）
    if o["profiles"]:
        pts = sorted(o["profiles"])
        arrs["prof_t"] = np.array(pts)
        arrs["prof_T"] = np.array([o["profiles"][k][0] for k in pts])
        arrs["prof_C"] = np.array([o["profiles"][k][1] for k in pts])
        arrs["prof_r_cm"] = o["r"] * 100.0
    np.savez_compressed(outdir / ("%s.npz" % cid), **arrs)

    # 交付半径上的最终标量
    j_end = int(np.argmin(np.abs(tt - T_SWEEP_END)))
    meta = dict(
        case=cid, note=note, wall_s=wall, status=o["status"],
        n_steps=o["n_steps"], t_final=o["t_final"], t_dry=o["t_dry"], t_end=o["t_end"],
        dr_cm=dr_cm, N=o["N"], dt1=kw.get("dt1"), dt2=kw.get("dt2"),
        t_switch=kw.get("t_switch"), snap_until_s=kw.get("snap_until_s"),
        picard={k: v for k, v in o["picard"].items() if k != "iters"},
        residual=o["residual"], floor_count=o["floor"]["count"],
        n_violations=len(o["violations"]),
        audit_max_abs=o["audit"]["max_abs"], audit_max_rel_global=o["audit"]["max_rel_global"],
        mass_ratio=o["mass"]["ratio"], mass_rel_err=o["mass"]["rel_err"],
        T_at_3h_C=[float(v) for v in T21[j_end]],
        C_at_3h_kgkg=[float(v) for v in C21[j_end]],
        C_max_3h=float(C21[j_end].max()), C_surf_3h=float(C21[j_end, -1]),
        T_center_3h=float(T21[j_end, 0]), T_surf_3h=float(T21[j_end, -1]),
        params={k: o["params"][k] for k in
                ("H", "HM", "D0", "BETA", "E_D", "RHO_A", "RHO_B", "CP_A", "CP_B",
                 "K_A", "K_B", "props_model", "energy_form", "moisture_form",
                 "face_rule", "ta_scenario", "ca_scenario",
                 # 附录 2 退化档（props_model='appendix2'）的**生效**物性参数。
                 # ⚠️ 上面 9 个附录 3 键（D0/BETA/E_D/RHO_A/B/CP_A/B/K_A/B）在该档
                 #    **全部不生效**（solver_q2.py:266/278/288/298-299 走 A2_* 分支），
                 #    只落附录 3 值会让复核者误判「把附录 3 的 D 当附录 2 用」
                 #    （2a 终审 S4）。此处把生效值一并落盘，两类键并列可辨。
                 "A2_RHO", "A2_CP", "A2_K", "A2_D0", "A2_BETA")},
    )
    if o["cmax"]["t"] is not None:
        cm = o["cmax"]["cmax"]
        tcx = o["cmax"]["t"]
        hit = np.flatnonzero(cm <= C_DRY)
        if hit.size:
            n = int(hit[0])
            meta["t_dry_after"] = float(tcx[n])
            if n > 0 and cm[n] != cm[n - 1]:
                w = (C_DRY - cm[n - 1]) / (cm[n] - cm[n - 1])
                meta["t_dry_linear"] = float(tcx[n - 1] + w * (tcx[n] - tcx[n - 1]))
    (outdir / ("%s.json" % cid)).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[case] %-22s wall=%7.1f s  steps=%9d  status=%s  t_dry=%s  floor=%d  viol=%d"
          % (cid, wall, o["n_steps"], o["status"],
             ("%.3f" % o["t_dry"]) if o["t_dry"] else "-",
             o["floor"]["count"], len(o["violations"])), flush=True)
    return meta


def main():
    ap = argparse.ArgumentParser(description="Q2 单算例执行器")
    ap.add_argument("case", help="算例 id（见 q2_case_runner.CASES / build_case）")
    ap.add_argument("--outdir", default=str(ROOT / "results" / "q2_parts"))
    args = ap.parse_args()
    run_case(args.case, Path(args.outdir))


if __name__ == "__main__":
    main()
