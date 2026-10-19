# -*- coding: utf-8 -*-
"""
q3_from_q2.py —— 问题 3：**搭 Q2 的车**（不独立求解）

职责
----
1) 读 Q2 的 `C_max(t)` **逐内部步**轨迹 → 在括住判据的相邻两步之间插值定位 `t_dry`
2) 读 Q2 的**全时段**水分场（60 s 网格 × 21 列）→ 写 `results/result3.xlsx`
3) 生成论文表 5 的取数 CSV（每 6 h × r = 0, 0.5, 1, 1.5, 2 cm，末行「烘干结束时间」）
4) 落盘 `results/q3_t_dry.json`（`t_dry` + 括住区间 + 精度声明 + h 单位换算 + 表 5 全精度值）

口径（A 题 PDF 第 2 页原文 + `docs/Q2_建模文档.md` §11）
----
· 判据：药材**各处**水分浓度 < 0.15 kg/kg
      `C_max(t) ≡ max_{0≤r≤R} C(r,t)`，`t_dry ≡ inf{ t>0 : C_max(t) < 0.15 }`
  §11.1：`C_max` 连续且横截（|dC_max/dt| ≠ 0）→ `<` 与 `≤` 的 inf 相等
  → 操作上取**首个 `C_max ≤ 0.15` 的内部步**，再在括住区间内插值
· 定位精度：数值不确定度**主报实测网格项** `t_dry.uncertainty.grid_contribution_s`
  （从 `results/q2_error_budget.json` 读入、**继承自 Q2** —— Q3 不独立求解，无自己的网格研究）；
  **缩小 Δt 不改善 t_dry**，能改善的只有网格误差与模型误差
· 原写（**已废弃，勿照报**）：`∂t_dry/∂C_max = 9.536×10⁵ s per (kg/kg)` → **只能报 ±10 s 量级**（§11.3）
· Q2 求解跨度 = `t_dry` + 3 h 余量；硬上限 **14 d**。
  **未达标必须报「未达标」，不得静默返回末行**（§11.4）
· `result3.xlsx` 输出间隔 **60 s**（**不是** Q2 的 1 s），半径 0, 0.1, …, 2.0 cm（21 列）；
  sheet 名与 A1 表头**逐位照抄附件 3 的 `result3.xlsx` 模板**（单 sheet「Sheet1」）

产物（全部落 `ROOT/results/`，**绝不触碰 Q1/Q2 任何产物**）
----
  result3.xlsx                 提交版：t = 60, 120, …（60 s 网格）× 21 列，四位小数
  q3_table5_moisture.csv       论文表 5 取数（每 6 h × 5 列，末行「烘干结束时间」）
  q3_t_dry.json                t_dry + 括住区间 + 精度 + 表 5 全精度值 + 溯源
  run_manifest_q3.json         输入/脚本 SHA-256 + 配置 + 口径

用法
----
  python src/q3_from_q2.py --dry-run     # 只校验输入存在性与格式，**不写任何产物**
  python src/q3_from_q2.py               # 出产物（要求 Q2 真实结果已落盘）
  python src/q3_from_q2.py --res-dir <Q2 产物目录> --out-dir <输出目录>   # 自测用
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

for _s in (sys.stdout, sys.stderr):                 # Windows 重定向到文件/管道时默认 GBK 会崩
    if getattr(_s, "reconfigure", None):
        _s.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]

# ==================== 硬口径（题目 PDF + Q2 建模文档 §11） ====================
C_DRY = 0.15                                  # 判据 [kg/kg]
T_CAP_S = 14.0 * 86400.0                      # 硬安全上限 14 d（§11.4）
MARGIN_S = 10800.0                            # 达标后余量 3 h（§11.4）
DT_OUT_S = 60.0                               # result3 输出间隔（题目 PDF 第 2 页）
R_OUT_CM = np.arange(0.0, 2.0 + 1e-9, 0.1)    # 21 个交付半径 [cm]
R_TAB5_CM = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
T_TAB5_S = 6.0 * 3600.0                       # 论文表 5 行间隔 6 h
# §11.3「修订前口径（留痕，勿再用）」的名义值（基于 |dC_max/dt| = 1.0486417213573925e-06）
DOC_FIELD_TO_TIME_S_PER_KGKG = 953600.0       # ∂t_dry/∂C_max = 1/|dC_max/dt|（§7.4 / §11.3）
DOC_T_DRY_UNCERTAINTY_S = 10.0                # §11.3 修订前报的「±10 s 量级」（**已废弃**，勿照报）
ASSUMED_FIELD_ERROR_KGKG = 1e-5               # §11.3 假定的场值误差（**非本次实测**）
# ⚠️ 上述 `DOC_*` **只作对照留痕**（落 `deprecated['uncertainty.doc']`），**不是** t_dry 的精度口径：
#    t_dry 的**主报**数值不确定度 = **实测网格项** `grid_contribution_s`（从
#    `results/q2_error_budget.json` 读入、继承自 Q2 —— Q3 不独立求解）；缩小 Δt 不改善 t_dry。
#    `sensitivity_from_slope()` 在斜率不可用时退回的本常量，同样只落 `deprecated` 块
#    ⇒ **`DOC_*` 与 `deprecated` 块均不得再取作 t_dry 的不确定度**。
# 原写（**已废弃，勿照报**）：「上述 `DOC_*` 是建模文档的估计。**实测的跨越点斜率可能差几倍**
#    （Q2 生产算例实测 |dC_max/dt| = 2.9369e-07，比 §11.3 的 1.0486e-06 小 3.57 倍 → 实际敏感度
#    3.405e6 s per (kg/kg)，**同场值误差下 t_dry 的不确定度是 ±34 s 而不是 ±9.5 s**）。
#    故本脚本**一律用实测斜率**算不确定度，`DOC_*` 只作对照留痕。」
#    —— 其中 ±34.0492570758918 s 是**灵敏度换算**（现 `t_dry.uncertainty.field_error_at_1e-5_kgkg_s`），
#    2026-09-11 已按三层口径降格为「仅灵敏度说明，非主报项」，**勿再当 t_dry 的不确定度**。
HEADER_A1 = "时间\\到药材中心的距离"           # 附件 3 模板逐位照抄
SHEET_NAME_RESULT3 = "Sheet1"                 # 附件 3 result3.xlsx 的唯一 sheet 名
SEED = 42                                     # Q3 无随机数；对齐 Q1/Q2 留痕

# ==================== Q2 产物接口（候选名，按优先级自上而下） ====================
# ⚠️ 这些是**约定/候选**，不是既成事实：脚本对每个候选逐一核验，
#    找不到或跨度不足时**明确报错并列出「我期待什么 / 实际看到什么」**，绝不静默跳过。
TRAJ_CANDIDATES = ("q2_cmax_trajectory.npz",)
TRAJ_KEYS = ("t_s", "C_max")
SUMMARY_CANDIDATES = ("results2.json",)
FIELD_CANDIDATES = (
    "q2_field_60s.npz",           # 首选：Q2 若为 Q3 增补「全程 60 s 场」输出
    "q3_field_source.npz",
    "q2_field_full_60s.npz",
    "q2_result2_field_full.npz",  # Q2 现有全精度场（**仅 0–10800 s，跨度不足会报错**）
)
FIELD_KEYSETS = (("t_s", "C_kgkg", "r_cm"), ("t_s", "C", "r_cm"), ("t_s", "C_kgkg"))

# Q1/Q2 产物名单：**只读**。本脚本的输出名若与其中任何一个相同 → 立即停止（硬约束）
Q12_OWNED = frozenset({
    "result1.xlsx", "result1_caseB.xlsx", "result1_t0_row.xlsx",
    "result2.xlsx", "result2_t0_row.xlsx",
    "results.json", "results2.json",
    "run_manifest.json", "run_manifest_q2.json",
    "q2_tables34.xlsx",
})
Q3_OUTPUT_NAMES = ("result3.xlsx", "q3_table5_moisture.csv",
                   "q3_t_dry.json", "run_manifest_q3.json")


class Q3InputError(RuntimeError):
    """输入缺失/格式不符/口径冲突 —— 一律**明确报错**，不猜、不静默降级。"""


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fwd(p) -> str:
    """路径一律正斜杠落盘（项目约定）。"""
    return str(p).replace("\\", "/")


def rel(p) -> str:
    """**落盘记录用**的路径写法：项目内一律记相对 ROOT 的 `results/...` 形式。

    为什么要它（2026-09-11 交前终审）：`ROOT = Path(__file__).resolve().parents[1]`
    是**绝对**路径，直接 `fwd()` 落盘会把本机用户名与盘符
    （形如 `<盘符>:/<用户目录>/.../<项目目录>/results/...`）写进交付件 ——
    支撑材料含身份信息属一票否决，且换台机器/换个用户名后这些记录全部指不到文件。
    改为项目相对路径后：换机可读、可复现，且仍能唯一定位到文件。

    只影响**记录值**，不影响任何**读文件**的路径（那些仍用各自的 Path 对象）：
      * 项目内 → 相对 ROOT 的正斜杠形式，如 `results/q3_t_dry.json`
      * 项目外 → 原样回退为绝对路径（不伪造相对关系，避免记录失真）
    """
    q = Path(p)
    try:
        return fwd(q.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return fwd(q)


def _fail(title: str, expect, seen_lines) -> "Q3InputError":
    lines = ["[Q3-INPUT-FAIL] " + title,
             "  我期待：" + expect,
             "  我实际看到："]
    lines += ["    - " + s for s in seen_lines]
    return Q3InputError("\n".join(lines))


# --------------------------------------------------------------------------- #
# 1) 定位 t_dry：逐内部步轨迹 + 相邻两步线性插值（Hermite 给定位误差）
# --------------------------------------------------------------------------- #
def locate_t_dry(t_s, c_max, level: float = C_DRY, t_cap: float = T_CAP_S) -> dict:
    """在**逐内部步**轨迹上定位 `C_max ≤ level` 的首个下穿时刻。

    算法与 `src/run_q2.py::locate_crossing` **同构**（本脚本是独立复现，用于交叉核对
    Q2 自己落盘的 `results2.json → t_dry`；两者不一致即报错停手，不挑一个信）。

    返回 dict。未下穿时 `crossed=False`，并区分
    「轨迹已到 t_cap 仍未达标」与「轨迹本身在 t_cap 之前就结束了」。
    """
    t_s = np.asarray(t_s, dtype=float)
    c_max = np.asarray(c_max, dtype=float)
    if t_s.ndim != 1 or c_max.ndim != 1 or t_s.size != c_max.size:
        raise Q3InputError(
            "[Q3-INPUT-FAIL] C_max 轨迹的 t_s / C_max 形状不符\n"
            "  我期待：两个等长一维数组\n"
            "  我实际看到：t_s%s, C_max%s" % (t_s.shape, c_max.shape))
    if t_s.size < 2:
        raise Q3InputError(
            "[Q3-INPUT-FAIL] C_max 轨迹点数不足\n"
            "  我期待：≥ 2 个**逐内部步**采样（60 s 采样不合格，见 §11.2）\n"
            "  我实际看到：%d 个点" % t_s.size)
    if np.any(np.diff(t_s) <= 0):
        raise Q3InputError(
            "[Q3-INPUT-FAIL] C_max 轨迹的 t_s 非严格递增\n"
            "  我期待：严格单调递增的内部步时间轴\n"
            "  我实际看到：min(diff)=%r（%d 处非正）"
            % (float(np.diff(t_s).min()), int(np.count_nonzero(np.diff(t_s) <= 0))))

    out = {"level": float(level), "crossed": False, "n_steps": int(t_s.size),
           "t_first_s": float(t_s[0]), "t_last_s": float(t_s[-1]),
           "dt_min_s": float(np.diff(t_s).min()), "dt_max_s": float(np.diff(t_s).max())}

    idx = np.flatnonzero(c_max <= level)
    if idx.size == 0:
        out.update(c_max_last=float(c_max[-1]), c_max_min=float(c_max.min()),
                   reached_t_cap=bool(t_s[-1] >= t_cap - 1e-9),
                   t_cap_s=float(t_cap))
        return out

    n = int(idx[0])
    out.update(crossed=True, n_star=n, first_le=True,
               t_at_cross_s=float(t_s[n]), c_max_at_cross=float(c_max[n]))
    if n == 0:
        out.update(t_before_s=None, c_max_before=None, t_linear_s=float(t_s[n]),
                   slope_s=None, t_hermite_s=None, localization_error_s=None)
        return out

    t0, t1 = float(t_s[n - 1]), float(t_s[n])
    c0, c1 = float(c_max[n - 1]), float(c_max[n])
    out.update(t_before_s=t0, c_max_before=c0, bracket_width_s=t1 - t0)
    if c1 == c0:
        out.update(t_linear_s=t1, t_hermite_s=None, localization_error_s=None,
                   slope_s=0.0)
        return out

    # (a) 线性插值（标称值）
    w = (level - c0) / (c1 - c0)
    t_lin = t0 + w * (t1 - t0)

    # (b) 三次 Hermite（端点导数取相邻步差商）→ 一阶插值的偏差估计
    def slope_at(i):
        if i <= 0:
            return (c_max[1] - c_max[0]) / (t_s[1] - t_s[0])
        if i >= c_max.size - 1:
            return (c_max[i] - c_max[i - 1]) / (t_s[i] - t_s[i - 1])
        return (c_max[i + 1] - c_max[i - 1]) / (t_s[i + 1] - t_s[i - 1])

    m0, m1 = float(slope_at(n - 1)), float(slope_at(n))
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

    out.update(t_linear_s=float(t_lin), t_hermite_s=float(t_herm),
               slope_s=float((c1 - c0) / h),
               slope_at_cross_s=float(0.5 * (m0 + m1)),
               localization_error_s=float(abs(t_herm - t_lin)))
    return out


def sensitivity_from_slope(slope, field_err: float = ASSUMED_FIELD_ERROR_KGKG) -> dict:
    """由**实测**跨越点斜率给出 `t_dry` 的精度口径（§11.3 的口径，但用实测敏感度）。

    `|slope|` 若不可用（None / 0）→ 退回 §11.3 的名义值与 ±%.0f s。
    """
    out = {"assumed_field_error_kgkg": field_err,
           "doc": {"factor_s_per_kgkg": DOC_FIELD_TO_TIME_S_PER_KGKG,
                   "t_dry_uncertainty_s": DOC_T_DRY_UNCERTAINTY_S,
                   "note": "建模文档 §11.3 名义值（基于 |dC_max/dt| = 1.0486417213573925e-06），"
                           "仅作对照，不用作结论"}}
    if slope is None or float(slope) == 0.0:
        out.update(factor_s_per_kgkg=None, t_dry_uncertainty_s=DOC_T_DRY_UNCERTAINTY_S,
                   t_dry_uncertainty_h=DOC_T_DRY_UNCERTAINTY_S / 3600.0,
                   measured=False, ratio_vs_doc=None)
        out["statement"] = ("跨越点斜率不可用 → 退回 §11.3 名义口径 ±%.0f s（≈ %.4f h）"
                            % (DOC_T_DRY_UNCERTAINTY_S, DOC_T_DRY_UNCERTAINTY_S / 3600.0))
        return out
    abs_slope = abs(float(slope))
    factor = 1.0 / abs_slope
    unc_s = field_err * factor
    out.update(abs_dCmax_dt_kgkg_per_s=abs_slope, factor_s_per_kgkg=factor,
               t_dry_uncertainty_s=unc_s, t_dry_uncertainty_h=unc_s / 3600.0,
               measured=True,
               ratio_vs_doc=DOC_FIELD_TO_TIME_S_PER_KGKG / factor)
    out["statement"] = (
        "由**实测**跨越点 |dC_max/dt| = %.6e kg/(kg·s) → ∂t_dry/∂C_max = %.6e s per (kg/kg)"
        "（§11.3 名义值 %.6e，实测为其 %.3f 倍）；场值误差 %.0e kg/kg → t_dry 不确定度 "
        "±%.1f s（≈ ±%.4f h）。**缩小 Δt 不再改善**（§11.3）。"
        % (abs_slope, factor, DOC_FIELD_TO_TIME_S_PER_KGKG,
           factor / DOC_FIELD_TO_TIME_S_PER_KGKG, field_err, unc_s, unc_s / 3600.0))
    return out


# --------------------------------------------------------------------------- #
# 1b) `t_dry.uncertainty` 分层（与 `src/run_q2.py::build_t_dry_uncertainty` **逐名同构**）
# --------------------------------------------------------------------------- #
# 历史口径：旧 `uncertainty` 块把**灵敏度换算**（1/|dC_max/dt| × 假定场值误差 1e-5 kg/kg）
# 直接命名为 `t_dry_uncertainty_s` 并配 statement「场值误差 1e-05 kg/kg → t_dry 不确定度
# ±34.0 s」—— **无任何「这是灵敏度、不是不确定度」的限定** ⇒ 会被当作 t_dry 的数值
# 不确定度报出，比实测网格项小 4.1 倍（**低估**）。2026-09-11 跨问口径一批改已把产物
# JSON 改写成分层结构；本段是它的**生成端** —— 任何人重跑本脚本都不得退回旧口径。
# 三个口径必须分开登记（键名与 Q2 逐名相同）：
#   ① 实测**网格项** = |t_dry(N=200) − t_dry(N=400)|            → **主报项**
#   ② 灵敏度换算 = 1/|dC_max/dt| × **假定**场值误差 1e-5 kg/kg  → 只是灵敏度说明
#   ③ 模型/数据（驱动平台情景）                                  → 以**区间**表述
# ⚠️ Q3 **不独立求解**（读 Q2 的场与网格）⇒ ①②③ 都没有自己的加密研究，只能与 Q2 同源、
#    逐位继承（「为什么继承」写在 grid_contribution_note 里）。
# ⚠️ 顶层 `field_to_time_factor_s_per_kgkg`（已废弃）**必须保留**：历史读取路径按名取，
#    只可标注废弃、不可删；完整留痕在 `deprecated` 块。
T_DRY_MAIN_COMPONENT = "grid_contribution_s"
_T_DRY_EB = "q2_error_budget.json"
_T_DRY_FTF_DEPRECATED = DOC_FIELD_TO_TIME_S_PER_KGKG       # 953600.0（§11.3 设计期估计）
_T_DRY_REFILL = "先跑 src/verify_q2_error_budget.py"


def _eb_get(d, *keys):
    """安全取 `q2_error_budget.json` 的嵌套键：任一层缺 → `None`（不抛异常、不编数值）。"""
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def load_measured_t_dry_budget(res_dir: Path, t_dry_now: float, slope_now):
    """读 `results/q2_error_budget.json` 里**实测**的 t_dry 贡献项（Q3 算不出的部分）。

    ⚠️ 为什么必须「读入」而不是「自算 / 硬编码」：网格项与 Δt 项都需要**多档网格 /
    时间步的加密研究**（各 3~4 次完整求解），该研究由 `src/verify_q2_error_budget.py`
    产出、时序在 Q2 生产跑之后；Q3 不独立求解 ⇒ 硬编码会与重跑后的新解脱钩
    （这正是本次要修的根因）。

    守卫（fail-closed，**宁可缺也不编**）：文件缺失 / 缺字段 / 其中 `t_dry_s` 或
    `amplification_measured_s_per_kgkg` 与本轮结果不一致（说明预算过期）→
    返回 `(None, 原因)`；调用方须把主报项写 `None` 并写明「待回填」，
    **不得填任何看似自算的数**。回填路径：`python src/verify_q2_error_budget.py`
    后重跑本脚本。
    """
    p = Path(res_dir) / _T_DRY_EB
    if not p.is_file():
        return None, "缺 results/%s（%s）" % (_T_DRY_EB, _T_DRY_REFILL)
    try:
        eb = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:                                        # noqa: BLE001
        return None, "results/%s 解析失败：%r" % (_T_DRY_EB, e)
    t_eb = _eb_get(eb, "t_dry_uncertainty", "t_dry_s")
    if t_eb is None:
        return None, "results/%s 缺 t_dry_uncertainty.t_dry_s" % _T_DRY_EB
    if abs(float(t_eb) - float(t_dry_now)) > 1e-9 * max(1.0, abs(float(t_dry_now))):
        return None, ("预算过期：其中 t_dry_s = %s，本轮 t_dry = %s（两者必须同源；"
                      "请重跑 src/verify_q2_error_budget.py）" % (t_eb, t_dry_now))
    amp_eb = _eb_get(eb, "t_dry_uncertainty", "amplification_measured_s_per_kgkg")
    if amp_eb is not None and slope_now:
        amp_now = 1.0 / abs(float(slope_now))
        if abs(float(amp_eb) - amp_now) > 1e-9 * max(1.0, abs(amp_now)):
            return None, ("预算过期：其中 amplification_measured_s_per_kgkg = %s，本轮 "
                          "1/|slope_at_cross| = %s" % (amp_eb, amp_now))
    return eb, None


def build_t_dry_uncertainty(t_dry_s, loc_err, slope, res_dir, legacy) -> dict:
    """构造 `q3_t_dry.json → t_dry.uncertainty`（18 键，与 `results2.json` **逐名同构**）。

    `legacy` = 旧口径块（`sensitivity_from_slope()` 的输出 + `localization_only_s`）；
    **原键逐位迁入 `deprecated`，一个都不删**。

    主报项 `grid_contribution_s` **读入** `results/q2_error_budget.json`（见
    `load_measured_t_dry_budget`），读不到就写 `None` + 「待回填」，**不伪造**。
    """
    sens = (1.0 / abs(float(slope))) if slope else None
    fe = (sens * ASSUMED_FIELD_ERROR_KGKG) if sens is not None else None
    eb, why = load_measured_t_dry_budget(res_dir, t_dry_s, slope)
    sgm = _eb_get(eb, "t_dry_uncertainty", "contributions", "space_grid_tail_measured")
    tsd = _eb_get(eb, "t_dry_uncertainty", "contributions", "time_step_dt_II")
    grid = _eb_get(sgm, "abs_diff_N200N400_s")
    dts = _eb_get(tsd, "implied_t_s")
    scen = _eb_get(eb, "t_dry_uncertainty", "contributions",
                   "driver_scenario", "interval_h")
    amp = _eb_get(eb, "t_dry_uncertainty", "amplification_measured_s_per_kgkg")
    ratio_doc = legacy.get("ratio_vs_doc") if isinstance(legacy, dict) else None
    t_s, t_h = float(t_dry_s), float(t_dry_s) / 3600.0

    # ---- 推荐表述（论文取数入口）：主报项与两个「说明项」必须分句说清 ----
    if grid is not None and fe is not None:
        rec = ("t_dry = %s s = %s h（未舍入值）。"
               "**数值**不确定度**主报 ±%s s（≈ ±1.4×10² s = %s h，实测网格项，"
               "**继承自 Q2**）**；±%s s 为**灵敏度说明**（= 1/|dCmax_dt_at_crossing| × "
               "**假定**场值误差 1e-5 kg/kg），**不得单独当作 t_dry 的不确定度**"
               "（会比网格项小 %s 倍）；"
               % (t_s, t_h, grid, float(grid) / 3600.0, fe, float(grid) / float(fe)))
        if scen:
            rec += ("**模型/数据**不确定度另以情景区间 [%s, %s] h 表述。"
                    % (scen[0], scen[1]))
        rec += ("不得声称 4 位小数有效（docs/Q2_建模文档.md §11.3；本文件与 results2.json "
                "→ t_dry.uncertainty 同口径）。")
    else:
        rec = ("t_dry = %s s = %s h（未舍入值）。**数值**不确定度的**主报项 = 实测网格项**"
               "（`grid_contribution_s` = |t_dry(N=200) − t_dry(N=400)|）**待回填**（%s）—— "
               "Q3 **不独立求解**、复用 Q2 的场与网格，网格项只能与 Q2 同源继承；"
               "**回填前不得报任何 t_dry 不确定度数**。" % (t_s, t_h, why))
        if sens is not None:
            rec += ("本次自算的灵敏度换算：∂t_dry/∂C_max = 1/|dCmax_dt_at_crossing| = %s "
                    "s per (kg/kg)（**只是灵敏度说明，不是不确定度**）。" % sens)
        rec += ("不得声称 4 位小数有效（docs/Q2_建模文档.md §11.3；本文件与 results2.json "
                "→ t_dry.uncertainty 同口径）。")

    # ---- ① 网格项 note：把「为什么继承」写清 ----
    if grid is not None and sgm is not None:
        grid_note = ("**为什么继承（不是本问自算）**：Q3 **不独立求解** —— "
                     "`src/q3_from_q2.py` 直接读 Q2 的水分场首达时刻（本文件 meta.script = "
                     "\"src/q3_from_q2.py\"，inputs_sha256 含 results/results2.json），"
                     "`results/q3_t_dry.json → t_dry.value_s` 与 `results/results2.json → "
                     "t_dry.value_s` **逐位相同**（均为 %s）⇒ Q3 复用 Q2 的场与网格，"
                     "**没有自己的网格收敛研究**，网格项只能与 Q2 同源、逐位继承。"
                     "**读入来源（不硬编码）**：results/%s → t_dry_uncertainty."
                     "contributions.space_grid_tail_measured.abs_diff_N200N400_s"
                     "（= conservative_uncertainty_s = |t_dry_dr_0_01_cm − t_dry_dr_0_005_cm| "
                     "= |%s − %s| = %s，同块 t_dry_dr_0_02_cm = %s）；该档**只改 Δr**"
                     "（Δt_I=1/16、Δt_II=1/2 与灵敏度口径逐项相同），与 Q3 的生产配置同为 "
                     "N=200。"
                     % (t_s, _T_DRY_EB, sgm.get("t_dry_dr_0_01_cm"),
                        sgm.get("t_dry_dr_0_005_cm"), grid,
                        sgm.get("t_dry_dr_0_02_cm")))
    else:
        grid_note = ("**待回填**：%s。网格项（= |t_dry(N=200) − t_dry(N=400)|，同 Δt_I=1/16、"
                     "Δt_II=1/2，只改 Δr）须读 results/%s 现算 —— **不填数、不估算**。"
                     "回填：%s，再重跑本脚本。" % (why, _T_DRY_EB, _T_DRY_REFILL))

    # ---- ② 灵敏度项 / ③ 模型数据项 ----
    if grid is not None and fe is not None:
        fe_note = ("= field_error_sensitivity_s_per_kgkg × 1e-5 kg/kg（**假定**场值误差，"
                   "不是实测误差）—— **仅灵敏度说明，非主报项**；单用它当 t_dry 的不确定度"
                   "会比实测网格项小 %s 倍。**本键承载的正是原 "
                   "uncertainty.t_dry_uncertainty_s = %s（旧主报值）**，按 Q2 的处置降格"
                   "（对照 results2.json → t_dry.uncertainty.field_error_at_1e-5_kgkg_s，"
                   "逐位同值）。" % (float(grid) / float(fe), fe))
        doc_note = ("✅ **无待裁决项**：docs/Q2_建模文档.md §11.3 已于 2026-09-11 按实测网格项"
                    "更正 —— 推论 1 现写「本算例：t_dry = 57.11 h ± %s s（≈ ±0.04 h，实测"
                    "网格项）」，并明确 ±%s s 只能作灵敏度说明、不得单独当作不确定度"
                    "（其第 5 条待确认项同此）。故本块与文档同口径。**论文取数**须以 "
                    "results2.json → t_dry.uncertainty.grid_contribution_s 与本文件 → "
                    "t_dry.uncertainty.grid_contribution_s 为准（两者逐位同值）。"
                    % (grid, fe))
        dep_why = ("原 uncertainty 块把**灵敏度换算**（%s s = 1/|dC_max/dt| × 假定场值误差 "
                   "1e-5 kg/kg）直接命名为 t_dry_uncertainty_s 并配 statement『场值误差 "
                   "1e-05 kg/kg → t_dry 不确定度 ±34.0 s』，**无任何「这是灵敏度、不是"
                   "不确定度」的限定** ⇒ 会被当作 t_dry 的数值不确定度报出，比实测网格项 "
                   "%s s 小 %s 倍（**低估 4.1 倍**）。2026-09-11 跨问口径预检确认该根因已在 "
                   "Q2/Q3/Q4 复制 3 次；本块按 Q2 的分层结构改写，原键值**全部保留在本 "
                   "deprecated 块内，未删除**。"
                   % (fe, grid, float(grid) / float(fe)))
    else:
        fe_note = ("**仅灵敏度说明，非主报项**：= 1/|dCmax_dt_at_crossing| × **假定**场值误差 "
                   "1e-5 kg/kg（本脚本自算）。主报项（实测网格项）待回填前，**不得**用本项"
                   "代替 t_dry 的不确定度。对照 results2.json → t_dry.uncertainty."
                   "field_error_at_1e-5_kgkg_s，逐位同值。")
        doc_note = ("⚠️ 本块主报项**待回填**（%s）：docs/Q2_建模文档.md §11.3 已按实测网格项"
                    "更正，主报项须与 Q2 同源继承后重跑本脚本。" % why)
        dep_why = ("原 uncertainty 块把**灵敏度换算**直接命名为 t_dry_uncertainty_s，"
                   "**无任何「这是灵敏度、不是不确定度」的限定** ⇒ 会被当作 t_dry 的数值"
                   "不确定度报出。2026-09-11 跨问口径预检确认该根因已在 Q2/Q3/Q4 复制 3 次；"
                   "本块按 Q2 的分层结构改写，原键值**全部保留在本 deprecated 块内，未删除**。"
                   "（主报网格项待回填：%s）" % why)

    # 旧（灵敏度）键值**逐位迁移**，一个都不删
    dep_keys = ("assumed_field_error_kgkg", "abs_dCmax_dt_kgkg_per_s", "factor_s_per_kgkg",
                "t_dry_uncertainty_s", "t_dry_uncertainty_h", "measured", "ratio_vs_doc",
                "localization_only_s", "doc", "statement")
    deprecated = {"uncertainty." + k: legacy.get(k) for k in dep_keys}
    deprecated["why"] = dep_why

    return {
        "main_reported_component": T_DRY_MAIN_COMPONENT,
        "recommended_statement": rec,
        "grid_contribution_s": grid,
        "grid_contribution_note": grid_note,
        "time_step_contribution_s": dts,
        "time_step_contribution_note": (
            "**同样继承自 Q2**（Q3 与 Q2 共用时间步与场，理由同 grid_contribution_note）。"
            "来源 results/%s → t_dry_uncertainty.contributions.time_step_dt_II.implied_t_s"
            "（= Δt_II 1/8 vs 1/16 的 C_max 轨迹最大偏差 %r × 实测放大系数 %r）；"
            "比网格项小两个量级 ⇒ **缩小 Δt 不改善 t_dry**（Q2 建模文档 §11.3 推论 2）。"
            % (_T_DRY_EB, _eb_get(tsd, "dC_max_window_max"), amp)
            if dts is not None else
            "**待回填**：%s（Δt_II 加密研究的实测值 × 实测放大系数）" % why),
        "localization_only_s": loc_err,
        "localization_only_note": ("Q3 自己的定位误差（线性 vs 三次 Hermite）—— 与本文件 "
                                   "t_dry.localization_error_s 同值，亦与 results2.json → "
                                   "t_dry.uncertainty.localization_only_s 逐位同值；"
                                   "比网格项小 10 个量级。"),
        "field_error_sensitivity_s_per_kgkg": sens,
        "field_error_sensitivity_note": (
            "= 1/|t_dry.dCmax_dt_at_crossing|（实测跨越点斜率 %r kg/(kg·s)）= **灵敏度换算"
            "系数，不是不确定度本身**。Q3 与 Q2 同场 ⇒ 与 results2.json → t_dry.uncertainty."
            "field_error_sensitivity_s_per_kgkg 逐位同值，亦与 results/%s → "
            "t_dry_uncertainty.amplification_measured_s_per_kgkg 逐位同值。"
            "**原写在本文件 uncertainty.factor_s_per_kgkg，已迁入 deprecated**。"
            % (abs(float(slope)), _T_DRY_EB) if slope else
            "**待回填**：跨越点斜率不可用。"),
        "field_error_at_1e-5_kgkg_s": fe,
        "field_error_at_1e-5_kgkg_note": fe_note,
        "model_data_uncertainty_interval_h": scen,
        "model_data_uncertainty_note": (
            "驱动平台情景（E0/E1/E2 的 4×4 外推口径）之差，**继承自 Q2**。来源 results/%s → "
            "t_dry_uncertainty.contributions.driver_scenario.interval_h（逐位同值）；"
            "**与上面的数值不确定度不是同一个东西**，须分开表述（Q2 建模文档 §14.3 第 16 条）。"
            % _T_DRY_EB if scen is not None else "**待回填**：%s" % why),
        "doc_consistency_note": doc_note,
        "field_to_time_factor_s_per_kgkg": _T_DRY_FTF_DEPRECATED,
        "field_to_time_factor_s_per_kgkg_status": (
            "**已废弃（勿用）** —— docs/Q2_建模文档.md §11.3『修订前口径（留痕，勿再用）』的"
            "设计期估计：LDF 单点冻结 D(0.15)（基于 |dC_max/dt| = 1.0486417213573925e-06）；"
            "实测斜率是它的 %s 倍 ⇒ **低估 %.3f 倍**。原写在本文件 uncertainty.doc."
            "factor_s_per_kgkg，迁移留痕见 deprecated。**不得作为 t_dry 的精度口径**。"
            % (ratio_doc, 1.0 / ratio_doc) if ratio_doc else
            "**已废弃（勿用）** —— §11.3 的设计期估计（LDF 单点冻结 D(0.15)）；"
            "迁移留痕见 deprecated。**不得作为 t_dry 的精度口径**。"),
        "deprecated": deprecated,
    }


def monotonicity_check(t_s, c_max, n_star, tol: float = 1e-12) -> dict:
    """§11.2 单调性检查：跨越判据之前 `C_max` 应非增。

    容差 `tol`：浮点舍入会给出 ~1e-16 量级的正增，不是物理回升。
    """
    d = np.diff(c_max[:n_star + 1]) if n_star is not None else np.diff(c_max)
    n_any = int(np.count_nonzero(d > 0))
    n_sig = int(np.count_nonzero(d > tol))
    k = int(np.argmax(d)) if d.size else 0
    return {"n_rise_steps_any_sign": n_any, "n_rise_steps_over_tol": n_sig,
            "monotone_tol": tol, "max_rise": float(d.max()) if d.size else 0.0,
            "t_at_max_rise_s": float(t_s[k + 1]) if d.size else None,
            "verdict": ("非增（容差 %g 以上无正增步）" % tol) if n_sig == 0 else
                       ("**出现 %d 个超容差正增步** —— 先查边界驱动方向与 Picard 冻结，"
                        "再考虑是否属附件 1 数据摆动的真实响应（§11.2）" % n_sig)}


# --------------------------------------------------------------------------- #
# 2) 读 Q2 产物（容错 + 明确报错）
# --------------------------------------------------------------------------- #
def _load_npz(path: Path, keysets, kind: str):
    z = np.load(path, allow_pickle=False)
    for ks in keysets:
        if all(k in z.files for k in ks):
            return {k: z[k] for k in ks}, list(ks), list(z.files)
    raise _fail("%s 的键名不符（%s）" % (kind, fwd(path)),
                "键 %s 之一" % " 或 ".join(str(list(k)) for k in keysets),
                ["实际键：" + repr(list(z.files))])


def scan_not_dry_evidence(res_dir: Path, max_lines: int = 6):
    """扫 Q2 的运行日志，找「14 d 未达标而中止」的痕迹。

    `run_q2.py` 在 `status == not_dry_by_cap` 时于**落任何产物之前** `SystemExit`，
    于是 `q2_cmax_trajectory.npz` / `results2.json` 都不存在。
    此时若只报「文件缺失」会误导 —— 真实原因是未达标（§11.4 要求明说）。
    """
    keys = ("not_dry_by_cap", "未达标", "t_cap", "硬上限")
    hits = []
    cands = sorted(res_dir.glob("*.log")) + sorted(res_dir.glob("*.txt")) \
        + sorted(res_dir.glob("*.out"))
    for p in cands:
        try:
            for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
                if any(k in line for k in keys):
                    hits.append("%s: %s" % (p.name, line.strip()[:180]))
                    break
        except OSError:
            continue
        if len(hits) >= max_lines:
            break
    return hits


def read_cmax_trajectory(res_dir: Path, explicit=None):
    """读 Q2 的 `C_max(t)` **逐内部步**轨迹。"""
    cands = [Path(explicit)] if explicit else [res_dir / n for n in TRAJ_CANDIDATES]
    seen = []
    for p in cands:
        if not p.exists():
            seen.append("缺失：%s" % fwd(p))
            continue
        d, ks, allk = _load_npz(p, (TRAJ_KEYS,), "C_max 轨迹")
        seen.append("命中：%s（键 %s，%d 点）" % (fwd(p), ks, d["t_s"].size))
        return {"path": p, "t_s": np.asarray(d["t_s"], float),
                "c_max": np.asarray(d["C_max"], float), "keys": ks, "seen": seen}
    ev = scan_not_dry_evidence(res_dir)
    extra = ["检索目录：%s" % fwd(res_dir)]
    if ev:
        extra += ["⚠️ 发现「14 d 未达标」痕迹（Q2 在落产物**之前**就中止了，"
                  "故文件根本不存在）—— 应报「未达标」而非「文件缺失」（§11.4）："] + \
                 ["    " + h for h in ev]
    raise _fail("找不到 Q2 的 C_max 逐内部步轨迹（**Q3 的必需输入**）",
                "以下任一文件（含键 %s）：%s" % (list(TRAJ_KEYS),
                                                " / ".join(fwd(res_dir / n)
                                                           for n in TRAJ_CANDIDATES)),
                seen + extra)


def read_q2_summary(res_dir: Path, explicit=None):
    """读 Q2 的 `results2.json → t_dry`（Q2 自己算的权威值，用于交叉核对）。"""
    cands = [Path(explicit)] if explicit else [res_dir / n for n in SUMMARY_CANDIDATES]
    seen = []
    for p in cands:
        if not p.exists():
            seen.append("缺失：%s" % fwd(p))
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:                                    # noqa: BLE001
            seen.append("读取失败：%s（%s）" % (fwd(p), e))
            continue
        top = list(d.keys())
        if "t_dry" not in d:
            seen.append("缺键 t_dry：%s（顶层键 %s）" % (fwd(p), top))
            continue
        td = d["t_dry"]
        if not isinstance(td, dict):
            seen.append("t_dry 不是对象：%s（类型 %s）" % (fwd(p), type(td).__name__))
            continue
        if "value_s" not in td:
            seen.append("t_dry 缺键 value_s：%s（t_dry 键 %s）"
                        % (fwd(p), list(td.keys())))
            continue
        seen.append("命中：%s（t_dry.value_s=%r）" % (fwd(p), td["value_s"]))
        return {"path": p, "t_dry": td, "status": d.get("meta", {}).get("config", {})
                .get("status"), "seen": seen}
    raise _fail("找不到 Q2 的 `results2.json → t_dry`（用于交叉核对定位值）",
                "以下任一文件，且含 `t_dry.value_s`：%s"
                % " / ".join(fwd(res_dir / n) for n in SUMMARY_CANDIDATES),
                seen + ["检索目录：%s" % fwd(res_dir)])


def _load_field(p: Path):
    """读全时段水分场。支持 npz（多组候选键）与 csv（首列 t_s，其余 21 列）。"""
    if p.suffix.lower() == ".npz":
        d, ks, allk = _load_npz(p, FIELD_KEYSETS, "水分场")
        t_s = np.asarray(d["t_s"], float)
        C = np.asarray(d["C_kgkg"] if "C_kgkg" in d else d["C"], float)
        if "r_cm" in d:
            r_cm = np.asarray(d["r_cm"], float)
        else:
            if C.shape[1] != R_OUT_CM.size:
                raise _fail("水分场无 r_cm 且列数 ≠ 21（%s）" % fwd(p),
                            "键 r_cm，或 %d 列" % R_OUT_CM.size,
                            ["实际列数：%d（键 %s）" % (C.shape[1], allk)])
            r_cm = R_OUT_CM.copy()
        return {"t_s": t_s, "C": C, "r_cm": r_cm, "keys": ks}
    if p.suffix.lower() == ".csv":
        raw = np.genfromtxt(p, delimiter=",", skip_header=1)
        if raw.ndim != 2 or raw.shape[1] < 2:
            raise _fail("水分场 CSV 结构不符（%s）" % fwd(p),
                        "首行表头 + 首列 t_s + 其余列半径", ["实际形状：%r" % (raw.shape,)])
        return {"t_s": raw[:, 0], "C": raw[:, 1:], "r_cm": R_OUT_CM.copy()}
    raise _fail("水分场文件类型不支持（%s）" % fwd(p), ".npz 或 .csv",
                ["实际后缀：%s" % p.suffix])


def resolve_field(res_dir: Path, t_dry_s: float, explicit=None):
    """找出**覆盖 t_dry** 的 60 s 抽样的来源。逐个候选核验，不静默跳过。"""
    cands = [Path(explicit)] if explicit else [res_dir / n for n in FIELD_CANDIDATES]
    seen = []
    for p in cands:
        if not p.exists():
            seen.append("缺失：%s" % fwd(p))
            continue
        try:
            d = _load_field(p)
        except Q3InputError as e:
            seen.append("读取失败：%s" % str(e).replace("\n", " | "))
            continue
        t_s, C, r_cm = d["t_s"], d["C"], d["r_cm"]
        if C.ndim != 2 or C.shape[0] != t_s.size:
            seen.append("形状不符：%s（t_s %d 点，C %r）" % (fwd(p), t_s.size, C.shape))
            continue
        if r_cm.size != R_OUT_CM.size or float(np.abs(r_cm - R_OUT_CM).max()) > 1e-9:
            seen.append("半径轴不符：%s（%d 列；前 3 列 %s）"
                        % (fwd(p), r_cm.size, np.round(r_cm[:3], 4).tolist()))
            continue
        if np.any(np.diff(t_s) <= 0):
            seen.append("时间轴非严格递增：%s" % fwd(p))
            continue
        if t_s[-1] < t_dry_s - 1e-9:
            seen.append("**跨度不足**：%s 只到 %.1f s（%.3f h），"
                        "而 t_dry = %.1f s（%.3f h）→ 缺 %.3f h"
                        % (fwd(p), t_s[-1], t_s[-1] / 3600.0,
                           t_dry_s, t_dry_s / 3600.0, (t_dry_s - t_s[-1]) / 3600.0))
            continue
        seen.append("命中：%s（%d 步 × %d 列，t=%.1f…%.1f s，键 %s）"
                    % (fwd(p), t_s.size, C.shape[1], t_s[0], t_s[-1], d.get("keys")))
        return {"path": p, "t_s": t_s, "C": C, "r_cm": r_cm, "seen": seen}
    raise _fail("找不到**覆盖 t_dry 的**全时段水分场（result3.xlsx 的必要输入）",
                "Q2 需提供 60 s 网格（或更细）× 21 列、且 t 跨度 ≥ t_dry = %.1f s 的水分场；"
                "候选名：%s" % (t_dry_s, " / ".join(FIELD_CANDIDATES)),
                seen + ["检索目录：%s" % fwd(res_dir),
                        "提示：Q2 现有 `q2_result2_field_full.npz` 的快照窗只覆盖 0–10800 s"
                        "（见 src/run_q2.py 的 snap_until_s），**不足以支撑 Q3 全程**。"
                        "需 Q2 增补全时段场输出（或改 60 s 网格快照）后重跑。"])


# --------------------------------------------------------------------------- #
# 3) 抽样与插值
# --------------------------------------------------------------------------- #
def sample_on_grid(t_s, C, targets, tol: float = 1e-6):
    """把 targets（秒）对齐到 t_s 网格上取行。要求**精确命中**（不插值）。"""
    idx = np.searchsorted(t_s, targets)
    idx = np.clip(idx, 0, t_s.size - 1)
    left = np.clip(idx - 1, 0, t_s.size - 1)
    idx = np.where(np.abs(t_s[left] - targets) <= np.abs(t_s[idx] - targets), left, idx)
    err = np.abs(t_s[idx] - targets)
    if float(err.max()) > tol:
        k = int(np.argmax(err))
        raise _fail("输出时间网格与水分场时间轴不对齐（无法精确抽样）",
                    "目标时刻能逐个落在场的时间轴上（|Δt| ≤ %g s）" % tol,
                    ["最大偏差 %.6f s 出现在目标 t=%.1f s（最近场时刻 %.6f s）"
                     % (err.max(), targets[k], t_s[idx[k]]),
                     "场时间轴：%.6f … %.6f s，共 %d 点" % (t_s[0], t_s[-1], t_s.size)])
    return idx, float(err.max())


def interp_linear_columns(t_s, C, t_q):
    """在时间方向对**每一列**做线性插值（用于「烘干结束时间」行，t_dry 通常不在网格上）。"""
    i = int(np.clip(np.searchsorted(t_s, t_q) - 1, 0, t_s.size - 2))
    i = max(i, 0)
    t0, t1 = float(t_s[i]), float(t_s[i + 1])
    if not (t0 - 1e-12 <= t_q <= t1 + 1e-12):
        raise _fail("烘干结束时刻落在水分场时间跨度之外",
                    "t_q 位于 [t_s[0], t_s[-1]] 内",
                    ["t_q = %.6f s；最近括住区间 [%.6f, %.6f] s" % (t_q, t0, t1)])
    w = 0.0 if t1 == t0 else (t_q - t0) / (t1 - t0)
    return (1.0 - w) * C[i] + w * C[i + 1], (t0, t1, w)


# --------------------------------------------------------------------------- #
# 4) 写产物
# --------------------------------------------------------------------------- #
def _guard_outputs(out_dir: Path):
    for n in Q3_OUTPUT_NAMES:
        if n in Q12_OWNED:
            raise Q3InputError("[FATAL] 输出名 %r 与 Q1/Q2 产物同名 —— "
                               "违反「不得覆盖 Q1/Q2 任何产物」" % n)
    out_dir.mkdir(parents=True, exist_ok=True)


def write_result3_xlsx(path: Path, times_s, mat, sheet_name=SHEET_NAME_RESULT3):
    """按附件 3 `result3.xlsx` 模板写：单 sheet、A1 表头、A 列时间（int，秒）、四位小数。"""
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = sheet_name
    ws.append([HEADER_A1] + [float(round(rc, 1)) for rc in R_OUT_CM])
    for k, t in enumerate(times_s):
        ws.append([int(round(t))] + [float(round(v, 4)) for v in mat[k]])
    wb.save(path)


def write_table5_csv(path: Path, rows_h, labels, mat, unit="kg/kg"):
    """论文表 5 取数（末行「烘干结束时间」），编码与 Q2 的表 CSV 一致（utf-8-sig）。"""
    lines = ["时间/h," + ",".join("r=%g cm" % r for r in R_TAB5_CM) + "  [单位 %s]" % unit]
    for i, lab in enumerate(labels):
        first = lab if isinstance(lab, str) else "%g" % rows_h[i]
        lines.append(first + "," + ",".join("%.4f" % v for v in mat[i]))
    path.write_text("\n".join(lines), encoding="utf-8-sig")


# --------------------------------------------------------------------------- #
# 5) 主流程
# --------------------------------------------------------------------------- #
def run(res_dir: Path, out_dir: Path, *, traj_path=None, summary_path=None,
        field_path=None, dry_run=False, end_policy="t_dry", verbose=True) -> dict:
    """读 Q2 产物 → 定位 t_dry → 出 result3.xlsx / 表 5 CSV / q3_t_dry.json。"""
    t_start = time.time()
    res_dir, out_dir = Path(res_dir), Path(out_dir)
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)

    # ---- (a) t_dry：Q2 自报值 + 本脚本独立复现，两者必须一致 ----
    traj = read_cmax_trajectory(res_dir, traj_path)
    loc = locate_t_dry(traj["t_s"], traj["c_max"], C_DRY, T_CAP_S)

    if not loc["crossed"]:
        why = ("轨迹已到硬上限 t_cap = 14 d 仍未达标" if loc["reached_t_cap"]
               else "轨迹在 t_cap 之前就结束了（Q2 自身未跑满 14 d）")
        raise Q3InputError(
            "[Q3-NOT-DRY] **在该口径下 14 d 内未达标** —— 按 §11.4 必须报「未达标」，"
            "不得静默返回末行。\n"
            "  判据：C_max(t) < %g kg/kg（各路）\n"
            "  轨迹：%s，%d 个内部步，t ∈ [%.1f, %.1f] s（%.3f h）\n"
            "  末点 C_max = %r，全程最小 C_max = %r\n"
            "  结论：%s"
            % (C_DRY, fwd(traj["path"]), loc["n_steps"], loc["t_first_s"],
               loc["t_last_s"], loc["t_last_s"] / 3600.0,
               loc["c_max_last"], loc["c_max_min"], why))

    mono = monotonicity_check(traj["t_s"], traj["c_max"], loc["n_star"])
    t_dry = float(loc["t_linear_s"])
    t_dry_herm = loc["t_hermite_s"]
    loc_err = loc["localization_error_s"]
    slope = loc["slope_at_cross_s"] if "slope_at_cross_s" in loc else loc["slope_s"]
    unc = sensitivity_from_slope(slope)      # 旧（灵敏度）口径 —— 原键值**全部**留痕进 deprecated
    unc_layer = build_t_dry_uncertainty(     # 分层口径：主报实测网格项（继承自 Q2）
        t_dry, loc_err, slope, res_dir, dict(unc, localization_only_s=loc_err))
    log("[t_dry] 首个 C_max ≤ %g 的内部步 n*=%d：t=%.6f s（C_max=%.12f）"
        % (C_DRY, loc["n_star"], loc["t_at_cross_s"], loc["c_max_at_cross"]))
    log("[t_dry] 括住区间 [%.1f, %.1f] s（宽 %.3f s）；线性插值 t_dry = %.6f s = %.6f h"
        % (loc["t_before_s"], loc["t_at_cross_s"], loc["bracket_width_s"],
           t_dry, t_dry / 3600.0))
    log("[t_dry] 定位误差（线性 vs Hermite）= %.3e s（仅插值定位，不含场值误差）"
        % (loc_err if loc_err is not None else float("nan")))
    # 灵敏度换算**不是**不确定度（旧 statement 的措辞已废弃）—— 打日志时明确区分，
    # 免 stdout 被当成精度口径引用。
    log("[unc  ] 灵敏度换算（**不是** t_dry 的数值不确定度）：" + unc["statement"])
    if unc_layer["grid_contribution_s"] is not None:
        _ratio = (float(unc_layer["grid_contribution_s"])
                  / float(unc_layer["field_error_at_1e-5_kgkg_s"]))
        log("[unc* ] 数值不确定度主报项 = 实测网格项 grid_contribution_s = %s s"
            "（**继承自 Q2**，读入 results/%s）；网格项是灵敏度项的 %s 倍 ⇒ 单用灵敏度项"
            "当不确定度会低估 %s 倍"
            % (unc_layer["grid_contribution_s"], _T_DRY_EB, _ratio, _ratio))
    else:
        log("[unc* ] 数值不确定度主报项 = 实测网格项 **待回填**（读不到 results/%s）"
            % _T_DRY_EB)
    log("[mono ] " + mono["verdict"])

    # Q2 自报的 t_dry：交叉核对（不一致 → 停手报告）
    q2_sum = None
    try:
        q2_sum = read_q2_summary(res_dir, summary_path)
    except Q3InputError as e:
        log("[warn ] 未取到 Q2 自报 t_dry，改用本脚本定位值：\n" + str(e))
    agree = None
    if q2_sum is not None:
        q2_val = float(q2_sum["t_dry"]["value_s"])
        agree = abs(q2_val - t_dry)
        log("[cross] Q2 results2.json t_dry=%.6f s vs 本脚本复现 %.6f s，差 %.3e s"
            % (q2_val, t_dry, agree))
        if agree > 1e-3:
            raise Q3InputError(
                "[Q3-CROSSCHECK-FAIL] Q2 自报 t_dry 与本脚本独立复现值不一致 —— 停手报告\n"
                "  期待：两值一致（|Δ| ≤ 1e-3 s）\n"
                "  实际：Q2 results2.json = %.9f s；本脚本复现 = %.9f s；Δ = %.3e s\n"
                "  来源：%s / %s\n"
                "  处置：先查 Q2 轨迹文件是否被覆盖、results2.json 是否与本轨迹同一算例，"
                "不得挑一个信。" % (q2_val, t_dry, agree, fwd(q2_sum["path"]),
                                    fwd(traj["path"])))
        if q2_sum["status"] == "not_dry_by_cap":
            raise Q3InputError("[Q3-NOT-DRY] Q2 自报 status=not_dry_by_cap —— 未达标（§11.4）")

    t_end = (float(q2_sum["t_dry"].get("t_end_s", t_dry + MARGIN_S))
             if q2_sum is not None else t_dry + MARGIN_S)

    # ---- (b) 全时段水分场 ----
    need_last = t_dry if end_policy == "t_dry" else max(t_dry, t_end)
    field = resolve_field(res_dir, need_last, field_path)
    t_f, C_f = field["t_s"], field["C"]
    log("[field] %s：%d 步 × %d 列，t=%.1f…%.1f s"
        % (fwd(field["path"]), t_f.size, C_f.shape[1], t_f[0], t_f[-1]))

    # ---- (c) result3.xlsx：60 s 网格，行到 floor(t_last/60)·60 ----
    t_last_out = float(np.floor(need_last / DT_OUT_S + 1e-9) * DT_OUT_S)
    times_out = np.arange(DT_OUT_S, t_last_out + 1e-9, DT_OUT_S)
    idx, align_err = sample_on_grid(t_f, C_f, times_out)
    mat_out = C_f[idx]
    i_tab5 = np.array([int(np.argmin(np.abs(R_OUT_CM - r))) for r in R_TAB5_CM])
    if float(np.abs(R_OUT_CM[i_tab5] - R_TAB5_CM).max()) > 1e-9:
        raise Q3InputError("[FATAL] 表 5 半径未落在交付半径轴上")

    # ---- (d) 论文表 5：每 6 h × 5 列 + 末行「烘干结束时间」 ----
    t_dry_h = t_dry / 3600.0
    n_full = int(np.floor(t_dry_h / 6.0 + 1e-12))
    tab_times_h = [6.0 * (k + 1) for k in range(n_full)]
    tab_t = np.array([th * 3600.0 for th in tab_times_h])
    if tab_t.size:
        idx5, _ = sample_on_grid(t_f, C_f, tab_t)
        mat5 = C_f[idx5][:, i_tab5]
    else:
        mat5 = np.zeros((0, R_TAB5_CM.size))
    row_end, bracket_used = interp_linear_columns(t_f, C_f, t_dry)
    mat5 = np.vstack([mat5, row_end[i_tab5][None, :]]) if mat5.size else \
        row_end[i_tab5][None, :]
    labels = ["%g" % h for h in tab_times_h] + ["烘干结束时间"]

    # 「烘干结束时间」行中心列超额的时间当量 → 与**主报**网格项（非灵敏度项）比口径。
    # 2026-09-11 跨问口径一批改已把产物里的本判据主报项改为实测网格项；**原写（±34.0 s
    # 灵敏度口径）逐字保留为「留痕·原写」** —— 重跑不得退回旧文本。
    _exc_t = float((row_end[0] - C_DRY) * float(unc["factor_s_per_kgkg"] or 0.0))
    _v_old = ("中心列读数 %.4f，超出判据 %+.3e kg/kg = %+.3f s 时间当量 —— 在 ±%.1f s "
              "精度口径内，属 60 s 网格线性插值的弦曲率效应，非判据违反"
              % (row_end[0], row_end[0] - C_DRY, _exc_t,
                 float(unc["t_dry_uncertainty_s"])))
    if unc_layer["grid_contribution_s"] is not None:
        _v_new = ("中心列读数 %.4f，超出判据 %+.3e kg/kg = %+.3f s 时间当量 —— 在**主报**"
                  "网格项 ±%s s（亦在灵敏度口径 ±%s s）内，属 60 s 网格线性插值的弦曲率"
                  "效应，非判据违反"
                  % (row_end[0], row_end[0] - C_DRY, _exc_t,
                     unc_layer["grid_contribution_s"],
                     unc_layer["field_error_at_1e-5_kgkg_s"]))
        _verdict = (_v_new + "。【留痕·原写】" + _v_old
                    if abs(_exc_t) <= float(unc_layer["grid_contribution_s"]) else
                    "**超出主报网格项 ±%s s 精度口径** —— 需查网格粗细与插值方式"
                    "。【留痕·原写】%s"
                    % (unc_layer["grid_contribution_s"], _v_old))
    else:
        _verdict = _v_old

    summary = {
        "meta": {
            "question": "Q3 烘干时间（搭 Q2 的车，不独立求解）",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "seed": SEED, "python": platform.python_version(), "numpy": np.__version__,
            "script": "src/q3_from_q2.py",
            "script_sha256": sha256(Path(__file__)) if Path(__file__).exists() else None,
            "wall_s": None, "dry_run": bool(dry_run),
            "end_policy": end_policy,
            "config": {"c_dry_kgkg": C_DRY, "dt_out_s": DT_OUT_S,
                       "r_out_cm": R_OUT_CM.tolist(),
                       "r_tab5_cm": R_TAB5_CM.tolist(),
                       "tab5_interval_h": 6.0, "margin_s": MARGIN_S,
                       "t_cap_s": T_CAP_S, "sheet_name": SHEET_NAME_RESULT3,
                       "header_a1": HEADER_A1},
        },
        "inputs_sha256": {rel(p): sha256(p) for p in
                          [traj["path"], field["path"]]
                          + ([q2_sum["path"]] if q2_sum is not None else [])},
        "criterion": {
            "statement": "药材各处水分浓度 < %g kg/kg；C_max(t) = max_r C(r,t)，"
                         "t_dry = inf{t>0 : C_max(t) < %g}" % (C_DRY, C_DRY),
            "relation_lt_le": "C_max 连续且横截 → `<` 与 `≤` 的 inf 相等；"
                              "操作上取首个 C_max ≤ %g 的内部步（Q2 建模文档 §11.1）" % C_DRY,
            "reference": "docs/Q2_建模文档.md §11.1 (Q2-26)(Q2-27)",
        },
        "t_dry": {
            "value_s": t_dry, "value_h": t_dry_h, "value_d": t_dry / 86400.0,
            "value_h_reported": "%.2f" % t_dry_h,
            "bracket": {"t_before_s": loc["t_before_s"], "C_max_before": loc["c_max_before"],
                        "t_after_s": loc["t_at_cross_s"], "C_max_after": loc["c_max_at_cross"],
                        "width_s": loc["bracket_width_s"]},
            "first_le_step": {"n_star": loc["n_star"], "t_s": loc["t_at_cross_s"],
                              "C_max": loc["c_max_at_cross"]},
            "interpolation": "线性（标称）+ 三次 Hermite（定位误差估计）",
            "t_dry_hermite_s": t_dry_herm,
            "localization_error_s": loc_err,
            "dCmax_dt_at_crossing": slope,
            # 分层口径（与 results2.json **逐名同构**）：主报项 = 实测网格项（Q3 不独立
            # 求解 ⇒ **继承自 Q2**，读入 q2_error_budget.json，**不硬编码**）；灵敏度换算
            # 降格为说明项；旧键值**全部**迁入 deprecated（不删除）。
            "uncertainty": unc_layer,
            "t_end_s": t_end, "margin_s": MARGIN_S,
            "status": (q2_sum["status"] if q2_sum is not None else None),
            "source": {"trajectory": rel(traj["path"]),
                       "summary_json": rel(q2_sum["path"]) if q2_sum else None,
                       "crosscheck_abs_diff_s": agree,
                       "independent_recompute": True},
            "monotonicity_before_crossing": mono,
        },
        "result3": {
            "path": rel(out_dir / "result3.xlsx"), "sheet": SHEET_NAME_RESULT3,
            "n_rows": int(times_out.size), "n_cols": int(mat_out.shape[1]),
            "t_first_s": float(times_out[0]) if times_out.size else None,
            "t_last_s": float(times_out[-1]) if times_out.size else None,
            "dt_s": DT_OUT_S, "round_decimals": 4,
            "time_axis_source": "Q2 水分场按 60 s 精确抽样（对齐偏差 %.3e s）" % align_err,
            "k_readme": "result3 是**抽样**产物：60 s 是 Q2 输出步长的整数倍，"
                        "半径 0…2.0 cm 即 21 列交付网格",
        },
        "table5": {
            "unit": "kg/kg", "r_cm": R_TAB5_CM.tolist(),
            "times_h": tab_times_h, "labels": labels,
            "values_full_precision": mat5.tolist(),
            "t_dry_h": t_dry_h,
            "last_row": "烘干结束时间",
            "last_row_time_bracket_s": bracket_used,
            "last_row_note": "「烘干结束时间」行的浓度由水分场在 t_dry 处**线性插值**得到"
                             "（t_dry 一般不在 60 s 网格上）",
            "csv_path": rel(out_dir / "q3_table5_moisture.csv"),
        },
        "checks": {
            "t_dry_crosscheck_ok": bool(agree is not None and agree <= 1e-3),
            "monotone_before_crossing": bool(mono["n_rise_steps_over_tol"] == 0),
            "field_covers_t_dry": bool(t_f[-1] >= t_dry - 1e-9),
            "grid_alignment_max_err_s": align_err,
            # ⚠️ 「烘干结束时间」行由 60 s 网格上的**线性插值**给出；C_max(t) 对 t 是凸的，
            #    弦在曲线上方 → 中心列可读出**略高于** 0.15 的值（O(h²) 曲率效应，
            #    与物理误差无关）。此处把它换算成**时间当量**，与实测斜率的精度口径对齐。
            "row_at_t_dry_center_kgkg": float(row_end[0]),
            "row_at_t_dry_center_excess_kgkg": float(row_end[0] - C_DRY),
            "row_at_t_dry_center_excess_time_s":
                float((row_end[0] - C_DRY) * float(unc["factor_s_per_kgkg"] or 0.0)),
            "row_at_t_dry_center_verdict": _verdict,
        },
    }

    if dry_run:
        log("[dry-run] 输入校验全部通过；未写任何产物。")
        summary["meta"]["wall_s"] = time.time() - t_start
        return summary

    _guard_outputs(out_dir)
    write_result3_xlsx(out_dir / "result3.xlsx", times_out, mat_out)
    write_table5_csv(out_dir / "q3_table5_moisture.csv", tab_times_h, labels, mat5)
    (out_dir / "q3_t_dry.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "seed": SEED, "generated_at": summary["meta"]["generated_at"],
        "script": "src/q3_from_q2.py", "script_sha256": summary["meta"]["script_sha256"],
        "inputs_sha256": summary["inputs_sha256"],
        "config": summary["meta"]["config"], "end_policy": end_policy,
        "t_dry_s": t_dry, "t_dry_h": t_dry_h,
        "uncertainty_s": unc_layer["grid_contribution_s"],
        "uncertainty_source": (
            "继承 Q2 实测网格项（Q3 不独立求解，复用 Q2 的场与网格；见 uncertainty."
            "grid_contribution_note）" if unc_layer["grid_contribution_s"] is not None
            else "**待回填**：主报实测网格项须先跑 src/verify_q2_error_budget.py"
                 "（见 uncertainty.grid_contribution_note）"),
        "uncertainty": unc_layer,
        "outputs": [rel(out_dir / n) for n in
                    ("result3.xlsx", "q3_table5_moisture.csv", "q3_t_dry.json")],
        "does_not_touch": sorted(Q12_OWNED),
        # 原 `uncertainty_s` = 灵敏度换算 / `uncertainty_source` = "实测跨越点斜率"
        # **原样留痕，未删除**（完整留痕另见 uncertainty.deprecated）
        "uncertainty_legacy_s": unc.get("t_dry_uncertainty_s"),
        "uncertainty_legacy_note": (
            "原 uncertainty_s = %s、uncertainty_source = \"实测跨越点斜率\" —— 那是"
            "**灵敏度换算**（= 1/|dCmax_dt_at_crossing| × **假定**场值误差 1e-05 kg/kg），"
            "**不是 t_dry 的数值不确定度**（比实测网格项小 %s 倍）。原值未删除，完整留痕见 "
            "uncertainty.deprecated。"
            % (unc.get("t_dry_uncertainty_s"),
               float(unc_layer["grid_contribution_s"])
               / float(unc_layer["field_error_at_1e-5_kgkg_s"]))
            if unc_layer["grid_contribution_s"] is not None
            else "原 uncertainty_s = %s（**灵敏度换算，不是 t_dry 的数值不确定度**；"
                 "主报网格项待回填）。原值未删除，完整留痕见 uncertainty.deprecated。"
                 % unc.get("t_dry_uncertainty_s")),
    }
    (out_dir / "run_manifest_q3.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["meta"]["wall_s"] = time.time() - t_start
    log("[out  ] %s" % fwd(out_dir))
    for n in ("result3.xlsx", "q3_table5_moisture.csv", "q3_t_dry.json",
              "run_manifest_q3.json"):
        log("        %-24s %8d B" % (n, (out_dir / n).stat().st_size))
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Q3：从 Q2 产物定位 t_dry 并生成 result3.xlsx / 论文表 5（不独立求解）")
    ap.add_argument("--res-dir", default=str(ROOT / "results"),
                    help="Q2 产物目录（默认 ROOT/results）")
    ap.add_argument("--out-dir", default=None, help="输出目录（默认同 --res-dir）")
    ap.add_argument("--dry-run", action="store_true",
                    help="只校验输入存在性与格式，**不写任何产物**")
    ap.add_argument("--end-policy", choices=("t_dry", "t_end"), default="t_dry",
                    help="result3 末行时刻：t_dry（默认）或 t_dry+3h 余量")
    ap.add_argument("--traj", default=None, help="显式指定 C_max 轨迹 npz")
    ap.add_argument("--summary", default=None, help="显式指定 results2.json")
    ap.add_argument("--field", default=None, help="显式指定全时段水分场 npz/csv")
    args = ap.parse_args(argv)
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.res_dir)
    try:
        s = run(Path(args.res_dir), out_dir, traj_path=args.traj,
                summary_path=args.summary, field_path=args.field,
                dry_run=args.dry_run, end_policy=args.end_policy)
    except Q3InputError as e:
        print("\n" + str(e), file=sys.stderr, flush=True)
        return 2
    u = s["t_dry"]["uncertainty"]
    print("\n[done ] t_dry = %.6f s = %.6f h（**数值**不确定度主报 = 实测网格项 %s s，"
          "**继承自 Q2**；灵敏度换算项不是不确定度）；result3 %d 行 × %d 列"
          % (s["t_dry"]["value_s"], s["t_dry"]["value_h"],
             ("±%s" % u["grid_contribution_s"]) if u["grid_contribution_s"] is not None
             else "**待回填**",
             s["result3"]["n_rows"], s["result3"]["n_cols"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
