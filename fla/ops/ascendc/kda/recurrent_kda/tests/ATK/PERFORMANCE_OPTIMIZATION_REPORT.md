# RecurrentKda 输入输出双缓冲性能优化测试报告

## 1. 最终结论

最终交付仅保留输入/输出双缓冲实现。方式A构建成功，完整 recurrent_kda pytest
27/27通过，ATK accuracy 8/8通过，ATK performance 8/8执行成功。

最终8-case Device总耗时为1275.7635 us，相对 `TEST_REPORT.md` 基线
1322.0956 us降低46.3321 us，整体提升3.5044%。性能数据来自 ATK
`performance_device` profiler，不使用 Python wall time。

## 2. 修改范围

- State输入队列采用双缓冲；
- 当前任务计算期间预取下一有效 State tile；
- 保留 State/Attention 输出双缓冲；
- 维持原有 Q/K/V/Gate/Beta 调度和多 token 语义；
- 未保留后续按 head 压缩 Beta、K=128 融合归约、vStep=128 等候选优化。

## 3. 构建与测试环境

- 构建方式：README 方式A，目标 `ascend950`；
- 最终 wheel SHA256：
  `5257c52d8544498f81307f44acfc4694dcd226689e44d73bb6794c9747769abd`；
- NPU测试限定为单张可用卡；
- ATK固定随机种子与8个既有 case 保持不变。

## 4. 精度与回归测试

| 测试项 | 结果 |
|---|---|
| README 方式A wheel构建/安装 | PASS |
| `bash run_atk.sh accuracy` | 8/8 PASS |
| 完整 `pytest -q tests/operators/recurrent_kda` | 27 passed |
| 原契约测试（cu_seqlens、state prefetch顺序） | PASS |

完整 pytest 在候选优化上发现的2项多 token 精度回归及二分过程见
[开发问题记录](../../docs/输入输出双缓冲开发问题记录.md)。

## 5. 性能结果

| case | B | 模式 | 基线 (us) | 最终 (us) | 变化 |
|---:|---:|---|---:|---:|---:|
| 0 | 1 | base | 13.0586 | 12.9304 | +0.9817% |
| 1 | 1 | cb_mtp | 13.2890 | 13.3572 | -0.5132% |
| 2 | 4 | base | 31.9610 | 32.5129 | -1.7268% |
| 3 | 4 | cb_mtp | 31.9865 | 33.2418 | -3.9245% |
| 4 | 16 | base | 110.3073 | 114.8615 | -4.1286% |
| 5 | 16 | cb_mtp | 110.6434 | 115.0845 | -4.0139% |
| 6 | 64 | base | 505.8273 | 474.3590 | +6.2212% |
| 7 | 64 | cb_mtp | 505.0225 | 479.4162 | +5.0703% |
| **合计** |  |  | **1322.0956** | **1275.7635** | **+3.5044%** |

“+”表示加速，“-”表示退化。收益主要来自 B=64：State搬运量较大时，双缓冲
更能覆盖 MTE2/MTE3 延迟；小/中 B 中流水建立与同步开销占比更高，存在波动退化。

## 6. 十轮实验摘要

| 轮次 | 方案 | 固定ATK | 最终状态 |
|---:|---|---|---|
| 1 | 按head压缩Beta搬运 | 精度/性能通过 | 完整pytest失败，撤销 |
| 2 | 删除Gate同步 | 性能退化 | 当轮回退 |
| 3 | K=128 MatVec融合归约 | 性能提升 | 依赖第1轮，撤销 |
| 4 | 衰减与归约融合 | 性能提升 | 依赖第1轮，撤销 |
| 5 | delta与状态更新融合 | 性能提升 | 依赖第1轮，撤销 |
| 6 | State仅预填1 tile | 性能退化 | 当轮回退 |
| 7 | vStep提高到128 | 性能提升 | 依赖第1轮，撤销 |
| 8 | Beta Vector化 | 性能提升 | 依赖第1轮，撤销 |
| 9 | Beta sigmoid融入MicroAPI | 性能提升 | 依赖第1轮，撤销 |
| 10 | BlockDim 56改48 | 性能退化 | 当轮回退 |

完整逐轮数据和 profile 指标见
[性能优化迭代记录](PERFORMANCE_OPTIMIZATION_ITERATIONS.md)。

## 7. 问题与风险结论

- 第4轮期间发生服务器重启和恢复期507033；触发原因未知，不归因于 kernel。
- 固定 ATK 仅覆盖单 token decode，不能替代多 token 完整回归。
- `aiv_vec_ratio` 用于定位瓶颈，不能单独作为保留门槛；最终以精度、完整回归和
  Device耗时共同判定。
- 最终源码未保留存在多 token 语义风险的候选优化。
