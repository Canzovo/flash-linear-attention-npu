# RecurrentKda 输入输出双缓冲测试报告

## 1. 结论

- 最终仅新增 State 输入双缓冲与跨任务首 tile 预取，保留既有输出双缓冲。
- README 方式 A 构建与 wheel 安装通过。
- `bash run_atk.sh accuracy`：8/8 PASS。
- `bash run_atk.sh performance`：连续两轮 8/8 SUCCESS。
- 第二轮总 Device 耗时由 1322.0956 us 降至 1278.7665 us，提升 3.2773%。

## 2. 代码范围

| 文件 | 内容 |
|---|---|
| `op_kernel/recurrent_kda.h` | legacy State 双槽及下一 head 首 tile 预取 |
| `op_kernel/arch35/recurrent_kda.h` | arch35 State 双槽及下一 task 首 tile 预取 |
| `op_host/recurrent_kda_tiling_processor.h` | State 输入双槽 UB 预算 |
| `docs/输入输出双缓冲设计.md` | 最终设计 |
| `docs/输入输出双缓冲开发问题记录.md` | 问题、截图与解决办法 |

未修改公开 API、tiling data、ATK 用例、数学顺序或 state 语义。

## 3. 构建与设备

| 项目 | 结果 |
|---|---|
| 方式 / SOC | README 方式 A / ascend950 |
| wheel | `flash_linear_attention_npu-26.7.0.dev0-950.x86_64-py3-none-any.whl` |
| SHA256 | `c7b024c3157a27183f189d5a8e4a0507aaf5020cd7501915711c8215c0577527` |
| 构建 / 安装退出码 | 0 / 0 |
| 可用卡 | 物理卡 1；`ASCEND_RT_VISIBLE_DEVICES=1` |

## 4. 精度

| 测试 | 结果 |
|---|---|
| `bash run_atk.sh accuracy` | `verified accuracy: 8/8 PASS` |
| 退出码 | 0 |

## 5. 性能

下表取稳定性复跑的第二轮 Device profiler 结果：

| case | B | 模式 | 基线 (us) | 优化后 (us) | 提升 |
|---:|---:|---|---:|---:|---:|
| 0 | 1 | base | 13.0586 | 12.7367 | +2.47% |
| 1 | 1 | cb_mtp | 13.2890 | 13.0845 | +1.54% |
| 2 | 4 | base | 31.9610 | 32.7078 | -2.34% |
| 3 | 4 | cb_mtp | 31.9865 | 33.0077 | -3.19% |
| 4 | 16 | base | 110.3073 | 114.4449 | -3.75% |
| 5 | 16 | cb_mtp | 110.6434 | 114.9100 | -3.86% |
| 6 | 64 | base | 505.8273 | 478.7474 | +5.35% |
| 7 | 64 | cb_mtp | 505.0225 | 479.1275 | +5.13% |

正数表示更快。B4/B16 有局部回退，但 B1、B64 提升，总耗时连续两轮下降：

| 轮次 | 总耗时 (us) | 相对基线 |
|---|---:|---:|
| 基线 | 1322.0956 | - |
| 最终实现第 1 轮 | 1278.9600 | +3.2627% |
| 稳定性复跑 | 1278.7665 | +3.2773% |

两轮均为 `verified performance: 8/8 SUCCESS`、退出码 0。未使用 Python
wall time 形成性能结论。

## 6. 验收

| 验收项 | 结果 |
|---|---|
| 设计文档 | PASS |
| 输入/输出双缓冲范围 | PASS |
| README 方式 A 构建 | PASS |
| accuracy 8/8 | PASS |
| performance 8/8 | PASS |
| 相对报告基线整体提升 | PASS，稳定约 3.27% |
| 问题记录、截图、根因和解决办法 | PASS |
| `git diff --check` / UTF-8 | PASS |
