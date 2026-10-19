# 验证套件（Q1–Q4）

Q1–Q4 的模型验证入口都集中在 `src/`，求解器仍留在各自的 `Q1/`–`Q4/` 目录；`Q1/data_prep.py` 是数据预处理，仍在 `Q1/`：

- `Q1/`–`Q4/`：提交版求解器与运行入口。
- `src/`：Q1–Q4 的验证入口，以及 Q2/Q3 验证所需的开发期依赖。

Q1/Q4 的验证脚本原本在各自目录，后来统一移到 `src/`；`independent_check_q1_temp.py` 也一起移入。移动时只补了 `sys.path`，让它们还能找到 `Q1/solver_q1.py`、`Q4/solver_q4.py`、`Q4/run_q4.py` 等求解器模块；求解器本身没有搬。

## 文件说明

| 文件                           | 覆盖    | 作用                                                         |
| ------------------------------ | ------- | ------------------------------------------------------------ |
| `convergence_q1.py`            | Q1      | 网格与时间步收敛                                             |
| `verification_q1.py`           | Q1      | Q1 综合验证                                                  |
| `window_error_q1.py`           | Q1      | 窗口内逐点误差核对                                           |
| `independent_check_q1_temp.py` | Q1      | Bessel 级数 + Duhamel 卷积独立温度校验；`run_q1.py` 也会调用 |
| `verify_q2_compare.py`         | Q2      | 对比验证：退化回归、能量/水分形式、物性口径等                |
| `verify_q2_convergence.py`     | Q2      | 网格无关性与时间步收敛                                       |
| `verify_q2_error_budget.py`    | Q2      | 误差预算：离散/模型/数据/参数四类归因                        |
| `verify_q2_sensitivity.py`     | Q2      | 单参数与组合灵敏度扫描                                       |
| `q2_parallel.py`               | Q2 依赖 | 验证算例的 subprocess 并行调度                               |
| `q2_case_runner.py`            | Q2 依赖 | 单个验证算例的执行入口                                       |
| `solver_q2.py`                 | Q2 依赖 | 开发期 Q2 求解器                                             |
| `q3_selftest.py`               | Q3      | 合成数据自测：插值定位、边界情况、产物格式                   |
| `q3_from_q2.py`                | Q3 依赖 | Q3 生产逻辑：读 Q2 轨迹定位烘干终点并写 result3              |
| `verify_q4_baseline.py`        | Q4      | 基线对比                                                     |
| `verify_q4_compare.py`         | Q4      | 对比验证                                                     |
| `verify_q4_convergence.py`     | Q4      | 网格与时间收敛                                               |
| `verify_q4_sensitivity.py`     | Q4      | 灵敏度扫描                                                   |

`independent_check_q1_temp.py` 也移到了 `src/`。`Q1/run_q1.py` 通过 `sys.path.insert(0, ROOT / "src")` 继续引用它；`run_q1.py` 的 manifest 路径也已同步改成 `src/...`。

## 运行

按 Q1 → Q2 → Q3 → Q4 顺序。

### Q1

```bash
python src/convergence_q1.py
python src/verification_q1.py
python src/window_error_q1.py
python src/independent_check_q1_temp.py
```

这些脚本会重跑数值解，耗时较长；先看脚本顶部的参数和输出说明。

### Q2

```bash
python src/verify_q2_compare.py --help
python src/verify_q2_convergence.py --help
python src/verify_q2_error_budget.py --help
python src/verify_q2_sensitivity.py --help
```

注意：Q2 的完整复跑依赖开发期的 `results/q2_parts/` 中间缓存（完整目录约 719MB，未纳入仓库）和当时的输入数据版本。仓库保留的是冻结的验证结果 JSON，因此这些脚本适合审阅和追溯，不保证在当前精简仓库里一键跑完全部算例。`verify_q2_error_budget.py` 在不带匹配缓存的情况下无法完成插值对照项，实测会因 `interp_step` 数据版本不匹配而中止；这种情况下应直接查看冻结的 `results/q2_error_budget.json`。

### Q3

```bash
python src/q3_selftest.py
```

它会写到 `results/_q3_selftest_out/`（已在 `.gitignore` 中忽略），不覆盖 `results/` 根下的任何产物。当前版本 54 项检查全部通过。

### Q4

```bash
python src/verify_q4_baseline.py
python src/verify_q4_compare.py
python src/verify_q4_convergence.py
python src/verify_q4_sensitivity.py
```

这些脚本会重跑数值解，耗时较长。

## 冻结的验证结果

仓库根的 `results/`：

- `q2_compare.json`
- `q2_convergence.json`
- `q2_dt_convergence.json`
- `q2_error_budget.json`
- `q2_sensitivity.json`
- `q2_scenarios.json`
- `q3_t_dry.json`
- `data_check.json`

## 与提交版的关系

- `src/solver_q2.py` 是开发期求解器，`Q2/problem2.py` + `Q2/utils.py` 是后来扁平化的提交版；两者同源，不建议交叉修改。
- `q3_from_q2.py` 是 Q3 的生产逻辑，提交版的 `Q3/problem3.py` 是它的整理版本。
- Q2/Q3 的独立验证脚本没有随提交包收录，只留在开发工作目录；仓库从开发目录补入了这些脚本和冻结结果，因此 Q2/Q3 的验证入口在 `src/`，而不是各自的 `Q2/`、`Q3/` 目录。
- `q3_selftest.py` 已按新的分层不确定度口径更新：主报项用 `grid_contribution_s`，灵敏度用 `field_error_sensitivity_s_per_kgkg`；当合成算例没有同源 `q2_error_budget.json` 时，阈值比较会明确跳过，而不是误报失败。
