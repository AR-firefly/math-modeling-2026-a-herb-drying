# -*- coding: utf-8 -*-
"""
q2_parallel.py —— Q2 检验算例的**并行调度**辅助（subprocess，一算例一进程）

为什么用 subprocess 而不是 multiprocessing：
  · Windows 默认 spawn，worker 必须可导入；`pull` 一个独立脚本最简单、最不易出错；
  · 每个算例是**确定性**的（无随机、无跨进程共享状态）→ 并行不改变任何数值；
  · 每个子进程只写自己的 `<case>.npz/.json`，无写冲突。

用法
----
    from q2_parallel import run_cases
    run_cases(["dr_100", "dr_200"], workers=6)
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

if getattr(sys.stdout, "reconfigure", None):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "src" / "q2_case_runner.py"
PARTS = ROOT / "results" / "q2_parts"
LOGS = PARTS / "_logs"


def case_done(cid, outdir=PARTS):
    return (outdir / ("%s.json" % cid)).is_file() and (outdir / ("%s.npz" % cid)).is_file()


def _run_one(cid, outdir, force):
    if not force and case_done(cid, outdir):
        return cid, True, 0.0, "cached"
    LOGS.mkdir(parents=True, exist_ok=True)
    log = LOGS / ("%s.log" % cid)
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as fh:
        p = subprocess.run([sys.executable, str(RUNNER), cid,
                            "--outdir", str(outdir)],
                           stdout=fh, stderr=subprocess.STDOUT, cwd=str(ROOT))
    dt = time.time() - t0
    ok = p.returncode == 0 and case_done(cid, outdir)
    return cid, ok, dt, ("ok" if ok else "FAILED(rc=%d, 见 %s)" % (p.returncode, log))


def run_cases(cases, workers=6, outdir=PARTS, force=False, verbose=True):
    """并行跑一批算例；返回 {case: dict(ok, wall_s, note)}。串行/并行的**数值结果完全一致**。"""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    res = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_run_one, c, outdir, force): c for c in cases}
        for f in futs:
            cid, ok, dt, note = f.result()
            res[cid] = dict(ok=ok, wall_s=dt, note=note)
            if verbose:
                print("[par] %-24s %-6s %7.1f s  %s"
                      % (cid, "OK" if ok else "FAIL", dt, note), flush=True)
    if verbose:
        print("[par] 共 %d 个算例，墙钟 %.1f s（workers=%d）"
              % (len(cases), time.time() - t0, workers), flush=True)
    return res


def load_part(cid, outdir=PARTS):
    """读回一个算例的 (npz, meta)。"""
    import numpy as np
    z = np.load(Path(outdir) / ("%s.npz" % cid))
    meta = json.loads((Path(outdir) / ("%s.json" % cid)).read_text(encoding="utf-8"))
    return z, meta


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[write] %s" % path, flush=True)


# ============================================================================
# 收敛阶 / 不确定度工具（Q1 同式：p = ln(e_a/e_b)/ln(r)）
# ============================================================================
def order(err_a, err_b, ratio):
    import math
    if err_a <= 0 or err_b <= 0:
        return float("nan")
    return math.log(err_a / err_b) / math.log(ratio)


def richardson(fine, coarse, p, ratio=2.0):
    """Richardson 外推（fine/coarse 为两个网格上的同一点数值）。"""
    return fine + (fine - coarse) / (ratio ** p - 1.0)
