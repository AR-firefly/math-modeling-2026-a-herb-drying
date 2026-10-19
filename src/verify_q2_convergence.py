# -*- coding: utf-8 -*-
"""
verify_q2_convergence.py —— 检验 ①网格无关性 + ②数值收敛（Δt_I / Δt_II）+ 单段-两段对照

对应建模文档 §9.3（强制验证项 1–5）、§9.4（网格方案）、§10 第 1/2 类。

**判据（②层验收线）**：全窗口（全部输出时刻 × 全部输出半径）**逐点最大误差 < 5×10⁻⁵**
（T 用 ℃、C 用 kg/kg）。**一阶常数不得套用 Q1**（§9.2），故必须实测。

窗口定义（**必须写清，否则判据不可复现**）：
  · T/C 场：`t = 1…10800 s` × `r = 0,0.1,…,2.0 cm`（即 result2 交付窗口，21 列）
  · Δt_II：`t = 14400…28800 s`（II 段前 4 h，60 s 采样）× 21 列
  · 单段/两段：`t = 14400, 18000, 21600, 25200 s` × 21 列 + C_max(t)
  · 早段薄层：`t ≤ 2 s` 的表面列（单独报告，不并入交付窗口）

产物：`results/q2_convergence.json`、`results/q2_dt_convergence.json`
用法：python src/verify_q2_convergence.py [--workers 6] [--force]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

if getattr(sys.stdout, "reconfigure", None):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from q2_parallel import run_cases, load_part, save_json, order, PARTS   # noqa: E402


def zget(z, key, default=None):
    """`NpzFile` 上安全的 `.get`（不同 numpy 版本对 NpzFile 的 Mapping 支持不一致）。"""
    return z[key] if key in z.files else default

RES = ROOT / "results"
ACCEPT = 5.0e-5                    # ②层验收线（T：℃；C：kg/kg）
R_OUT = np.arange(0.0, 2.0 + 1e-9, 0.1)
DRS = [("100", 0.02), ("200", 0.01), ("400", 0.005), ("800", 0.0025)]
RULES = ["arith", "harm", "faceCT"]
DTI = ["16", "32", "64", "128"]
DTII = ["8", "16", "32"]


def _ensure_dtI64():
    """`dtI_64` = 生产运行本身（Δt_I=1/64、Δr=0.01、[0,10800] 每秒 21 半径）。

    直接从 `results/q2_result2_field_full.npz` 复制成算例格式，省掉 240 s 重算。
    """
    out_npz = PARTS / "dtI_64.npz"
    if out_npz.is_file():
        return
    src = RES / "q2_result2_field_full.npz"
    if not src.is_file():
        raise SystemExit("缺 %s（需先跑 src/run_q2.py）" % src)
    z = np.load(src)
    t = np.concatenate(([0.0], z["t_s"]))
    T0 = z["T0_row_C"][None, :]
    C0 = z["C0_row_kgkg"][None, :]
    np.savez_compressed(out_npz, t_s=t, T_21=np.vstack([T0, z["T_C"]]),
                        C_21=np.vstack([C0, z["C_kgkg"]]), r_out_cm=z["r_cm"])
    (PARTS / "dtI_64.json").write_text(
        '{"case":"dtI_64","note":"生产运行本体（Δt_I=1/64 s，Δr=0.01 cm，[0,10800] s），'
        '由 results/q2_result2_field_full.npz 复制","wall_s":0.0,'
        '"dr_cm":0.01,"dt1":0.015625,"dt2":0.015625,"picard":{},"floor_count":0,'
        '"n_violations":0}', encoding="utf-8")
    print("[mk] dtI_64 由生产产物复制")


def _align(a, b):
    """把两个算例截到公共时间轴（要求起点/步长一致）。

    入参是**归一化**的算例字典 `{"t","T","C"}`（由各 study 函数统一构造），
    返回 `(t, T_a, T_b, C_a, C_b)`。
    """
    ta, tb = a["t"], b["t"]
    if ta.size != tb.size or not np.array_equal(ta, tb):
        n = min(ta.size, tb.size)
        if not np.allclose(ta[:n], tb[:n], atol=1e-9):
            raise ValueError("时间轴不匹配（%d vs %d 点）" % (ta.size, tb.size))
        return ta[:n], a["T"][:n], b["T"][:n], a["C"][:n], b["C"][:n]
    return ta, a["T"], b["T"], a["C"], b["C"]


def diffs_pair(a, b):
    t, T1, T2, C1, C2 = _align(a, b)
    return dict(t=t, dT=np.abs(T1 - T2), dC=np.abs(C1 - C2))


def _stats(d, tag, res, cols):
    """把逐点差统计成 (max, argmax, 表面列 max, 分段) 形式。"""
    out = {}
    for name, arr in (("T", d["dT"]), ("C", d["dC"])):
        k = np.unravel_index(int(np.argmax(arr)), arr.shape)
        out[name] = dict(max=float(arr.max()),
                         at=dict(t_s=float(d["t"][k[0]]), r_cm=float(cols[k[1]])),
                         max_interior=float(arr[:, :-1].max()),
                         max_surface_col=float(arr[:, -1].max()),
                         at_surface_col_t_s=float(d["t"][int(np.argmax(arr[:, -1]))]),
                         rms=float(np.sqrt(np.mean(arr ** 2))))
    out["_tag"] = tag
    return out


WINDOWS = [
    ("full_t_ge_1s", "全窗口 t = 1…10800 s（result2.xlsx 交付窗口）", None),
    ("t_ge_1800s", "论文表 3/4 窗口 t = 1800…10800 s（0.5–3 h）", 1800.0),
    ("t_gt_2s", "去掉早段薄层：t > 2 s", 2.0000000001),
]


def windowed(arr, t, thr):
    """按时间阈值切窗口；thr=None 表示全窗口。"""
    return arr if thr is None else arr[t >= thr]


def windowed_nosurf(arr, t, thr):
    """按时间切窗口 + 去掉 r=2.0 cm 表面列。"""
    a = arr if thr is None else arr[t >= thr]
    return a[:, :-1]


def grid_study(cols):
    """① 网格无关性：Δr ∈ {0.02,0.01,0.005,0.0025} cm，Δt 固定 1/64 s，[0,10800] s。"""
    out = {"design": "Δr ∈ {0.02,0.01,0.005,0.0025} cm（N=100/200/400/800），"
                     "Δt=1/64 s，窗口 t=1…10800 s × 21 个交付半径",
           "note": "Q1 的结论（空间在 N=100 已收敛）**不可复用**——Q2 的 D、α、C 变幅全变（§9.4）",
           "pairs": {}, "orders": {}, "uncertainty": {}}
    fields = {}
    for lab, dr in DRS:
        z, meta = load_part("dr_%s" % lab)
        fields[lab] = dict(t=z["t_s"], T=z["T_21"], C=z["C_21"], dr=dr)
    ref_lab = "800"
    for i in range(len(DRS) - 1):
        a, b = DRS[i][0], DRS[i + 1][0]        # a 粗、b 细
        d = diffs_pair(fields[a], fields[b])
        s = _stats(d, "%s_vs_%s" % (a, b), None, cols)
        out["pairs"][s["_tag"]] = {k: v for k, v in s.items() if k != "_tag"}
        # 观测收敛阶（p = ln(e_coarse/e_fine)/ln(r)，r = Δr_a/Δr_b = 2）
        out["orders"][s["_tag"]] = {
            "p_T": order(s["T"]["max"], s["T"]["max"], 2.0),      # 占位，下面用相邻差重算
        }
    # 相邻差得到「同一量」的误差序列：E_i ≈ |f_i − f_{i+1}|
    seq = {}
    for i in range(len(DRS) - 1):
        a, b = DRS[i][0], DRS[i + 1][0]
        d = diffs_pair(fields[a], fields[b])
        seq["%s_%s" % (a, b)] = (float(d["dT"].max()), float(d["dC"].max()),
                                 float(d["dT"][:, -1].max()), float(d["dC"][:, -1].max()))
    labs = ["%s_%s" % (DRS[i][0], DRS[i + 1][0]) for i in range(3)]
    out["orders"] = {
        "grids": labs,
        "p_T_full": [order(seq[labs[i]][0], seq[labs[i + 1]][0], 2.0) for i in range(2)],
        "p_C_full": [order(seq[labs[i]][1], seq[labs[i + 1]][1], 2.0) for i in range(2)],
        "p_T_surface_col": [order(seq[labs[i]][2], seq[labs[i + 1]][2], 2.0) for i in range(2)],
        "p_C_surface_col": [order(seq[labs[i]][3], seq[labs[i + 1]][3], 2.0) for i in range(2)],
        "note": "p 由**相邻网格差**序列给出（E_i≈|f_i−f_{i+1}|）；网格比 r=2。"
                "表面列单列（Q1 A7 的强制项）。",
    }
    # Richardson p=1（保守；实测 p 见上一块）外推 + 网格不确定度
    for p_use, tag in ((1.0, "p=1"), (2.0, "p=2")):
        e_T = seq[labs[-1]][0]
        e_C = seq[labs[-1]][1]
        out["uncertainty"][tag] = {
            "note": "保守网格不确定度：最细一对网格之差 |f_400 − f_800|（**不依赖 p 的假设**）",
            "T_C_full_window_max": e_T, "C_kgkg_full_window_max": e_C,
            "T_surface_col_max": seq[labs[-1]][2], "C_surface_col_max": seq[labs[-1]][3],
        }
    z800, _ = load_part("dr_800")
    z400, _ = load_part("dr_400")
    for p_use in (1.0, 2.0):
        ex_T = z800["T_21"][1:] + (z800["T_21"][1:] - z400["T_21"][1:]) / (2 ** p_use - 1)
        ex_C = z800["C_21"][1:] + (z800["C_21"][1:] - z400["C_21"][1:]) / (2 ** p_use - 1)
        ex_T2 = z400["T_21"][1:] + (z400["T_21"][1:] - fields["200"]["T"][1:]) / (2 ** p_use - 1)
        ex_C2 = z400["C_21"][1:] + (z400["C_21"][1:] - fields["200"]["C"][1:]) / (2 ** p_use - 1)
        unc_T = np.maximum(np.abs(z800["T_21"][1:] - ex_T), np.abs(ex_T2 - ex_T))
        unc_C = np.maximum(np.abs(z800["C_21"][1:] - ex_C), np.abs(ex_C2 - ex_C))
        # ⚠️ 标签必须由 **p_use** 生成（不能沿用上一个循环残留的 `tag`，
        #    否则两次迭代都写进 `richardson_p=2`，而 `richardson_p=1` 永远缺失）
        out["uncertainty"]["richardson_p=%g" % p_use] = {
            "T_full_window_max": float(unc_T.max()),
            "C_full_window_max": float(unc_C.max()),
            "T_surface_col_max": float(unc_T[:, -1].max()),
            "C_surface_col_max": float(unc_C[:, -1].max()),
            "T_at_t_max_s": float(z800["t_s"][1:][int(np.argmax(unc_T.max(axis=1)))]),
            "C_at_t_max_s": float(z800["t_s"][1:][int(np.argmax(unc_C.max(axis=1)))]),
            "meets_acceptance_T": bool(unc_T.max() < ACCEPT),
            "meets_acceptance_C": bool(unc_C.max() < ACCEPT),
        }
    # ---- 分窗口分解：判据在哪个窗口成立、哪一列不成立 ----
    dC_800 = np.abs(fields["400"]["C"] - fields["800"]["C"])
    dT_800 = np.abs(fields["400"]["T"] - fields["800"]["T"])
    tfull = fields["800"]["t"]
    out["window_resolved_conservative_400_800"] = {
        "note": "保守网格不确定度 = |f(N400) − f(N800)|（不依赖收敛阶假设）",
        "T_field_C": {nm: float(windowed(dT_800, tfull, thr).max())
                      for nm, _, thr in WINDOWS},
        "C_field_kgkg": {nm: float(windowed(dC_800, tfull, thr).max())
                         for nm, _, thr in WINDOWS},
        "C_excl_surface_col_kgkg": {
            nm: float(windowed_nosurf(dC_800, tfull, thr).max())
            for nm, _, thr in WINDOWS},
    }
    # 表面列逐时刻网格不确定度（Q1 A7 强制项：必须随结果报告）
    np.savetxt(RES / "q2_surface_column_uncertainty.csv",
               np.column_stack([tfull, dC_800[:, -1], dT_800[:, -1],
                                np.abs(fields["200"]["C"][:, -1] - fields["800"]["C"][:, -1]),
                                np.abs(fields["100"]["C"][:, -1] - fields["800"]["C"][:, -1])]),
               delimiter=",", comments="",
               header="t_s,grid_unc_C_surf_N400N800,grid_unc_T_surf_N400N800,"
                      "grid_unc_C_surf_N200N800,grid_unc_C_surf_N100N800")
    out["surface_column_uncertainty"] = {
        "file": "results/q2_surface_column_uncertainty.csv",
        "C_max_kgkg": float(dC_800[:, -1].max()),
        "C_at_t_s": float(tfull[int(np.argmax(dC_800[:, -1]))]),
        "C_at_t_ge_1800s": float(windowed(dC_800[:, -1][:, None], tfull, 1800.0).max()),
        "C_at_t_le_2s": float(dC_800[tfull <= 2.0, -1].max()),
        "T_max_C": float(dT_800[:, -1].max()),
        "note": "表面列（r = 2.0 cm）的网格不确定度**逐时刻**见 CSV；"
                "该列在 t ≲ 2 s 处是全窗口最大网格误差的来源（内部列小得多）",
    }
    out["acceptance"] = ACCEPT
    return out


def dtI_study(cols):
    """② 数值收敛（Δt_I）：Δt ∈ {1/16,1/32,1/64,1/128} s，Δr=0.01 cm，[0,10800] s。"""
    fields = {}
    for lab in DTI:
        z, meta = load_part("dtI_%s" % lab)
        fields[lab] = dict(t=z["t_s"], T=z["T_21"], C=z["C_21"])
    out = {"design": "Δt_I ∈ {1/16,1/32,1/64,1/128} s（单段，t_switch 关闭），Δr=0.01 cm，"
                     "窗口 t=1…10800 s × 21 个交付半径",
           "note": "**一阶常数不得套用 Q1**（§9.2）；本块给出 Q2 自己的实测 p 与常数",
           "pairs": {}, "orders": {}, "reference": {}, "acceptance": ACCEPT}
    seq = {}
    for i in range(len(DTI) - 1):
        a, b = DTI[i], DTI[i + 1]
        d = diffs_pair(fields[a], fields[b])
        s = _stats(d, "dt1_%s_vs_%s" % (a, b), None, cols)
        out["pairs"][s.pop("_tag")] = s
        seq["%s_%s" % (a, b)] = (float(d["dT"].max()), float(d["dC"].max()),
                                 float(d["dT"][:, -1].max()), float(d["dC"][:, -1].max()),
                                 d)
    labs = ["%s_%s" % (DTI[i], DTI[i + 1]) for i in range(3)]
    out["orders"] = {
        "levels": labs,
        "p_T_full": [order(seq[labs[i]][0], seq[labs[i + 1]][0], 2.0) for i in range(2)],
        "p_C_full": [order(seq[labs[i]][1], seq[labs[i + 1]][1], 2.0) for i in range(2)],
        "p_T_surface_col": [order(seq[labs[i]][2], seq[labs[i + 1]][2], 2.0) for i in range(2)],
        "p_C_surface_col": [order(seq[labs[i]][3], seq[labs[i + 1]][3], 2.0) for i in range(2)],
    }
    # 以 1/128 为基准的 Richardson（p=1）与逐档误差
    z128, _ = load_part("dtI_128")
    z64, _ = load_part("dtI_64")
    ex_T = z128["T_21"][1:] + (z128["T_21"][1:] - z64["T_21"][1:]) / (2 ** 1.0 - 1)
    ex_C = z128["C_21"][1:] + (z128["C_21"][1:] - z64["C_21"][1:]) / (2 ** 1.0 - 1)
    ref = {"reference": "Richardson p=1, Δt=1/128 与 1/64 外推",
           "levels": {}}
    for lab in DTI:
        eT = np.abs(fields[lab]["T"][1:] - ex_T)
        eC = np.abs(fields[lab]["C"][1:] - ex_C)
        ref["levels"][lab] = {
            "dt_s": 1.0 / float(lab),
            "T_max": float(eT.max()), "T_at": dict(
                t_s=float(fields[lab]["t"][1:][np.unravel_index(int(np.argmax(eT)),
                                                               eT.shape)[0]]),
                r_cm=float(cols[np.unravel_index(int(np.argmax(eT)), eT.shape)[1]])),
            "C_max": float(eC.max()), "C_at": dict(
                t_s=float(fields[lab]["t"][1:][np.unravel_index(int(np.argmax(eC)),
                                                               eC.shape)[0]]),
                r_cm=float(cols[np.unravel_index(int(np.argmax(eC)), eC.shape)[1]])),
            "T_surface_col_max": float(eT[:, -1].max()),
            "C_surface_col_max": float(eC[:, -1].max()),
            "meets_acceptance": bool(eT.max() < ACCEPT and eC.max() < ACCEPT),
        }
    out["reference"] = ref
    # 分窗口分解（时间误差）：说明超标格**是否落在交付窗口/是否只在早段薄层**
    tw = fields["64"]["t"][1:]
    wr = {}
    for lab in DTI:
        eT = np.abs(fields[lab]["T"][1:] - ex_T)
        eC = np.abs(fields[lab]["C"][1:] - ex_C)
        over = np.any(eC > ACCEPT, axis=1)
        wr[lab] = {
            "T_field_C": {nm: float(windowed(eT, tw, thr).max()) for nm, _, thr in WINDOWS},
            "C_field_kgkg": {nm: float(windowed(eC, tw, thr).max()) for nm, _, thr in WINDOWS},
            "C_surface_col_kgkg": {
                nm: float(windowed(eC[:, -1][:, None], tw, thr).max())
                for nm, _, thr in WINDOWS},
            "C_excl_surface_col_kgkg": {
                nm: float(windowed_nosurf(eC, tw, thr).max()) for nm, _, thr in WINDOWS},
            "n_cells_over_criterion_full": int(np.count_nonzero(eC > ACCEPT)),
            "n_cells_over_criterion_t_ge_1800s":
                int(np.count_nonzero(windowed(eC, tw, 1800.0) > ACCEPT)),
            "n_cells_over_criterion_t_gt_2s":
                int(np.count_nonzero(windowed(eC, tw, 2.0000000001) > ACCEPT)),
            "t_of_over_criterion_cells": [float(v) for v in np.unique(tw[over])[:20]],
        }
    out["window_resolved"] = wr
    out["conclusion"] = {
        "literal_full_window_pass": [lab for lab in DTI
                                     if ref["levels"][lab]["meets_acceptance"]],
        "delivered_dt_s": 1.0 / 64,
        "statement": ("按**字面**的全窗口逐点 < %g 判据，只有 Δt=1/128 通过；Δt=1/64 的"
                      "超标格**全部**落在 t ≤ 2 s 的 r = 2.0 cm 表面列（该处网格不确定度"
                      "比时间误差大一个量级，交付精度由**网格**而非时间步控制）。"
                      "详见 window_resolved。**该事实必须随结果报告。**" % ACCEPT),
    }
    # 表面列逐时刻（以 1/128 为基准）——供论文如实报告
    t_win = fields["64"]["t"][1:]
    np.savetxt(RES / "q2_dtI_surface_column_error.csv",
               np.column_stack([t_win] + [np.abs(fields[lab]["C"][1:, -1] - ex_C[:, -1])
                                          for lab in DTI]),
               delimiter=",", comments="",
               header="t_s," + ",".join("C_surf_err_dt1_%s" % lab for lab in DTI))
    # 早段薄层（t ≤ 2 s）单独报告
    m = t_win <= 2.0
    out["early_thin_layer_t_le_2s"] = {
        "note": "Q1 实测该段误差**不衰减地进入后续窗口**（§9.3-5），故必须单独实测",
        "n_times": int(np.count_nonzero(m)),
        "levels": {lab: {"T_max": float(np.abs(fields[lab]["T"][1:][m] - ex_T[m]).max()),
                         "C_max": float(np.abs(fields[lab]["C"][1:][m] - ex_C[m]).max()),
                         "C_surface_col_max":
                             float(np.abs(fields[lab]["C"][1:][m, -1] - ex_C[m, -1]).max())}
                   for lab in DTI},
        "criterion_C_analog": ACCEPT,
    }
    return out


def dtII_study(cols):
    """② 数值收敛（Δt_II）：Δt_II ∈ {1/8,1/16,1/32} s，自 t=14400 s 检查点重启。"""
    fields = {}
    for lab in DTII:
        z, meta = load_part("dtII_%s" % lab)
        fields[lab] = dict(t=z["t_s"], T=z["T_21"], C=z["C_21"], cmax_t=z.get("cmax_t"),
                           cmax=z.get("cmax_v"), meta=meta)
    out = {"design": "Δt_II ∈ {1/8,1/16,1/32} s，自 t=14400 s 检查点重启（Δt_I 沿用生产 1/64 s），"
                     "比对窗口 t=14400…28800 s（60 s 采样）× 21 个交付半径",
           "note": "建模文档 §9.3-3 的字面判据「表 3/表 4 全窗口」在物理上只覆盖 I 段 —— "
                   "Δt_II **不可能**影响 t ≤ 10800 s 的结果。故把判据落实到 II 段窗口"
                   "（这是对判据的**可复现操作化**，须在论文中如实说明）",
           "acceptance": ACCEPT, "pairs": {}, "orders": {}, "reference": {}, "t_dry": {}}
    seq = {}
    for i in range(len(DTII) - 1):
        a, b = DTII[i], DTII[i + 1]
        d = diffs_pair(fields[a], fields[b])
        s = _stats(d, "dt2_%s_vs_%s" % (a, b), None, cols)
        out["pairs"][s.pop("_tag")] = s
        seq["%s_%s" % (a, b)] = (float(d["dT"].max()), float(d["dC"].max()))
    labs = ["%s_%s" % (DTII[i], DTII[i + 1]) for i in range(2)]
    out["orders"] = {"levels": labs,
                     "p_T": [order(seq[labs[0]][0], seq[labs[1]][0], 2.0)],
                     "p_C": [order(seq[labs[0]][1], seq[labs[1]][1], 2.0)]}
    z32, _ = load_part("dtII_32")
    z16, _ = load_part("dtII_16")
    ex_T = z32["T_21"] + (z32["T_21"] - z16["T_21"]) / (2 ** 1.0 - 1)
    ex_C = z32["C_21"] + (z32["C_21"] - z16["C_21"]) / (2 ** 1.0 - 1)
    for lab in DTII:
        eT = np.abs(fields[lab]["T"] - ex_T)
        eC = np.abs(fields[lab]["C"] - ex_C)
        cm = fields[lab]["cmax"]
        tc = fields[lab]["cmax_t"]
        ed = np.abs(np.interp(fields["8"]["cmax_t"], tc, cm) - fields["8"]["cmax"]) \
            if cm is not None and tc is not None else np.array([np.nan])
        out["reference"][lab] = {
            "dt_s": 1.0 / float(lab),
            "T_max": float(eT.max()), "C_max": float(eC.max()),
            "T_surface_col_max": float(eT[:, -1].max()),
            "C_surface_col_max": float(eC[:, -1].max()),
            "cmax_traj_max_abs_diff_vs_dt2_8": float(np.nanmax(ed)),
            "meets_acceptance": bool(eT.max() < ACCEPT and eC.max() < ACCEPT),
        }
        out["t_dry"][lab] = fields[lab]["meta"].get("t_dry_linear")
    return out


def single_vs_two(cols):
    """§9.3-2 跨接无伪影：单段（全程 1/64）vs 两段（1/64 → 1/8）。

    两段式的 II 段场直接取 `dtII_8` 算例（它正是「自 t=14400 s 检查点 + Δt_II=1/8」，
    与生产两段式的 II 段逐位同构），避免再从生产轨迹反推场。
    """
    zs, ms = load_part("single_seg")
    z8, _ = load_part("dtII_8")
    t_s = zs["t_s"]
    out = {"design": "单段：全程 Δt=1/64 s，[0,28800] s（60 s 采样）；"
                     "两段：Δt_I=1/64 (≤14400 s) + Δt_II=1/8 s（dtII_8 算例）",
           "note": "§9.3-1 要求「全程 Δt=1/64 的单段算例」，本实现跑到 t=28800 s —— "
                   "§9.3-2 点名的对照时刻（14400/18000/21600/25200 s）全部覆盖。"
                   "跨接伪影是 t=14400 s 处的瞬态，超过 4 h（≈9 τ_th）后不再增长；"
                   "**未跑满全程（≈55–70 min）已如实登记为局限**",
           "times_s": [14400.0, 18000.0, 21600.0, 25200.0], "points": {},
           "acceptance": ACCEPT}
    for tc in out["times_s"]:
        k = int(np.argmin(np.abs(t_s - tc)))
        j = int(np.argmin(np.abs(z8["t_s"] - tc)))
        # ⚠️ 对齐检查：dtII_* 自检查点重启，其快照时刻是 14400+60k（**不含 14400**）。
        # 若两边实际时刻不一致，比较就变成「跨时刻」比较 → 会报出**假伪影**
        # （2026-09-11 实测踩到：t=14400 vs 14460 曾给出 2.8e-2 ℃ 的假差异）。
        if abs(float(t_s[k]) - float(z8["t_s"][j])) > 1e-9:
            out["points"]["t_%d" % int(tc)] = {
                "t_single_s": float(t_s[k]), "t_two_s": float(z8["t_s"][j]),
                "skipped": "两段式无该时刻的快照（重启后首帧为 14460 s）→ 不作场对比",
            }
            continue
        eT = float(np.abs(zs["T_21"][k] - z8["T_21"][j]).max())
        eC = float(np.abs(zs["C_21"][k] - z8["C_21"][j]).max())
        out["points"]["t_%d" % int(tc)] = {
            "t_single_s": float(t_s[k]), "t_two_s": float(z8["t_s"][j]),
            "T_21_single": [float(v) for v in zs["T_21"][k]],
            "C_21_single": [float(v) for v in zs["C_21"][k]],
            "T_21_two": [float(v) for v in z8["T_21"][j]],
            "C_21_two": [float(v) for v in z8["C_21"][j]],
            "max_abs_diff_T": eT, "max_abs_diff_C": eC,
            "meets_acceptance": bool(eT < ACCEPT and eC < ACCEPT),
        }
    # 重启点本身的一致性：检查点状态 == 单段基准在 t=14400 s 的场（逐点）
    ckp = ROOT / "results" / "q2_checkpoint_t14400s.npz"
    if ckp.is_file():
        zc = np.load(ckp)
        k14 = int(np.argmin(np.abs(t_s - 14400.0)))
        idx = np.arange(0, 201, 10)      # 单段基准存 21 个交付半径；检查点存全场（Δr=0.01）
        out["restart_point_consistency"] = {
            "t_single_s": float(t_s[k14]),
            "max_abs_diff_T_C": float(np.abs(zc["T_C"][idx] - zs["T_21"][k14]).max()),
            "max_abs_diff_C_kgkg": float(np.abs(zc["C_kgkg"][idx] - zs["C_21"][k14]).max()),
            "note": "「两段式的 t=14400 s 起点」与「单段基准的 t=14400 s 场」逐点对比；"
                    "两者都是 Δt=1/64 的解，应逐位一致 → 证明跨接点无伪影",
        }
    pts = [v for v in out["points"].values() if "max_abs_diff_T" in v]
    out["max_abs_diff_T"] = max(v["max_abs_diff_T"] for v in pts) if pts else None
    out["max_abs_diff_C"] = max(v["max_abs_diff_C"] for v in pts) if pts else None
    out["meets_acceptance"] = bool(out["max_abs_diff_T"] is not None
                                   and out["max_abs_diff_T"] < ACCEPT
                                   and out["max_abs_diff_C"] < ACCEPT)
    # C_max(t) 轨迹对照（单段 vs 两段，公共 60 s 网格，[14400, 28800]）
    tt = np.arange(14400.0, 28800.0 + 1e-9, 60.0)
    cs = np.interp(tt, zs["cmax_t"], zs["cmax_v"])   # ⚠️ 必须用轨迹时刻 cmax_t，
                                                     # 不是快照时刻 t_s（两者步长不同）
    c2 = np.interp(tt, z8["cmax_t"], z8["cmax_v"])
    out["cmax_traj_compare"] = {
        "window_s": [14400.0, 28800.0], "n_points": int(tt.size),
        "max_abs_diff": float(np.abs(cs - c2).max()),
        "t_at_max_abs_diff_s": float(tt[int(np.argmax(np.abs(cs - c2)))]),
        "C_max_single_at_28800s": float(cs[-1]), "C_max_two_at_28800s": float(c2[-1]),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    result_lab = {}
    try:
        _ensure_dtI64()
    except SystemExit as e:
        print(e)

    cases = []
    cases += ["dr_%s" % lab for lab, _ in DRS]
    # ⚠️ `face_arith_*` 与 `dr_*` 是**同一个算例**（arith 是主口径）→ 不重跑，
    # 下面对 arith 一律读 `dr_*` 的产物（省 4 个算例 ≈ 1500 s）。
    cases += ["face_%s_%s" % (r, lab) for r in RULES if r != "arith"
              for lab, _ in DRS]
    cases += ["dtI_%s" % lab for lab in DTI]
    cases += ["dtII_%s" % lab for lab in DTII]
    cases += ["single_seg"]
    cases = [c for c in cases if c != "dtI_64"]
    r = run_cases(cases, workers=args.workers, force=args.force)
    failed = [k for k, v in r.items() if not v["ok"]]
    if failed:
        print("[WARN] 失败算例：%s" % failed)

    cols = R_OUT
    grid = grid_study(cols)
    save_json(RES / "q2_convergence.json", grid)

    dtI = dtI_study(cols)
    dtII = dtII_study(cols)
    svt = single_vs_two(cols)

    # 界面系数对比（§8.5：arith/harm/faceCT × ≥4 级网格，报 p 与网格不确定度）
    face = {"design": "arith / harm / faceCT 各跑 Δr ∈ {0.02,0.01,0.005,0.0025} cm，"
                      "Δt=1/64 s，窗口 t=1…10800 s × 21 个交付半径",
            "rules": {}, "acceptance": ACCEPT}
    for rule in RULES:
        fs = {}
        for lab, dr in DRS:
            # arith = 主口径 = dr_* 算例本体
            z, _ = load_part(("dr_%s" if rule == "arith" else "face_%s_%s")
                             % ((lab,) if rule == "arith" else (rule, lab)))
            fs[lab] = dict(t=z["t_s"], T=z["T_21"], C=z["C_21"])
        e = {}
        for lab in ("100", "200", "400"):
            d = diffs_pair(fs[lab], fs[{"100": "200", "200": "400", "400": "800"}[lab]])
            e[lab] = (float(d["dT"].max()), float(d["dC"].max()),
                      float(d["dC"][:, -1].max()))
        face["rules"][rule] = {
            "pair_max_diffs": {k: dict(T_full=v[0], C_full=v[1], C_surf_col=v[2])
                               for k, v in e.items()},
            "p_T": order(e["100"][0], e["200"][0], 2.0),
            "p_C": order(e["100"][1], e["200"][1], 2.0),
            "p_C_surface_col": order(e["100"][2], e["200"][2], 2.0),
            "grid_uncertainty_conservative": dict(
                T_full=max(e["100"][0], e["200"][0], e["400"][0]),
                C_full=max(e["100"][1], e["200"][1], e["400"][1]),
                C_surf_col=max(e["100"][2], e["200"][2], e["400"][2])),
        }
    # 三规则之间的差异（在 N=800 上）
    zref, _ = load_part("dr_800")            # arith 主口径
    for rule in ("harm", "faceCT"):
        z, _ = load_part("face_%s_800" % rule)
        face["rules"]["%s_vs_arith_N800" % rule] = {
            "T_full_max": float(np.abs(z["T_21"] - zref["T_21"]).max()),
            "C_full_max": float(np.abs(z["C_21"] - zref["C_21"]).max()),
            "C_surface_col_max": float(np.abs(z["C_21"][:, -1] - zref["C_21"][:, -1]).max()),
        }
    dtI["face_rule_study"] = face

    # Δt_II 与单段对照并入 dt 收敛文件
    dtI["dt_II_study"] = dtII
    dtI["single_vs_two_segment"] = svt
    save_json(RES / "q2_dt_convergence.json", dtI)

    print("\n[收敛] 网格：p_T=%s  p_C=%s（全窗口，相邻网格）"
          % (grid["orders"]["p_T_full"], grid["orders"]["p_C_full"]))
    print("[收敛] 网格不确定度（最细一对，保守）：T %.3e ℃、C %.3e kg/kg；表面列 C %.3e"
          % (grid["uncertainty"]["p=1"]["T_C_full_window_max"],
             grid["uncertainty"]["p=1"]["C_kgkg_full_window_max"],
             grid["uncertainty"]["p=1"]["C_surface_col_max"]))
    print("[收敛] Δt_I：p_T=%s  p_C=%s" % (dtI["orders"]["p_T_full"],
                                         dtI["orders"]["p_C_full"]))
    for lab in DTI:
        lv = dtI["reference"]["levels"][lab]
        print("        Δt=1/%-4s T_max %.3e ℃  C_max %.3e kg/kg  达标=%s"
              % (lab, lv["T_max"], lv["C_max"], lv["meets_acceptance"]))
    print("[收敛] Δt_II（窗口 [14400,28800]）：p_T=%s p_C=%s"
          % (dtII["orders"]["p_T"], dtII["orders"]["p_C"]))
    for lab in DTII:
        lv = dtII["reference"][lab]
        print("        Δt=1/%-4s T_max %.3e  C_max %.3e  达标=%s  t_dry=%s"
              % (lab, lv["T_max"], lv["C_max"], lv["meets_acceptance"],
                 dtII["t_dry"][lab]))
    print("[收敛] 单段 vs 两段：跨接时刻 max|ΔT| %.3e、max|ΔC| %.3e → %s"
          % (svt["max_abs_diff_T"], svt["max_abs_diff_C"],
             "无伪影" if svt["meets_acceptance"] else "**有伪影，须退回单段或改连续过渡**"))


if __name__ == "__main__":
    main()
