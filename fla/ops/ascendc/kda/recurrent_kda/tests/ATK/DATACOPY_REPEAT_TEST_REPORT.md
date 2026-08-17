# RecurrentKDA DataCopy 搬运优化测试报告

## 1. 结论

本轮对 `op_kernel/arch35/recurrent_kda.h` 的全部 `DataCopy` 做了分类审计，并验证了 generic repeat 与 K=128 dual load/store 两类候选。

最终保留 `DecayMatVecReduce128` 和 `ProcessDeltaKQReduce128` 的 FP32 dual load/store：两轮 ATK profiler 总耗时均值从 `993.9492 us` 降至 `975.9634 us`，提升 `1.81%`。四项精度门禁全部通过。

`MatVecMul` LocalTensor repeat 候选已实现、完整编译和测量，但因当前 K=128 公共路径不可达且长 case 波动退化而回退，不纳入最终代码。

## 2. 最终修改

- `DecayMatVecReduce128`
  - gate、vec、state 的两次 64-lane load 合并为 `DIST_DINTLV_B32` dual load；
  - state 的两次 store 合并为 `DIST_INTLV_B32` dual store。
- `ProcessDeltaKQReduce128`
  - k、q、state 使用相同的 dual load；
  - state 使用 dual store。
- 偶/奇列分发后仍执行相同的逐列点运算；两个乘积寄存器相加后归约，数学上仍为 128 列总和；匹配的 interleave store 恢复原 state 布局。

未修改 host tiling、接口、UB 容量、同步、测试数据或精度阈值。

## 3. DataCopy 审计结论

| 类别 | 处理结论 | 原因 |
|---|---|---|
| GM/UB `DataCopyPad`、`DataCopyExtParams`、LocalTensor count copy | 不修改 | 已使用 blockCount/blockLen/stride 表达批量 MTE 搬运 |
| generic `MatVecMul` | 候选实现后回退 | repeat 可去掉逐行 RegTensor DataCopy，但当前 K=128 公共路径不可达，实测无可归因收益 |
| `DecayMatVecReduce128` | 保留 dual load/store | K=128 热路径，成对 FP32 load/store 可合并 |
| `ProcessDeltaKQReduce128` | 保留 dual load/store | K=128 热路径，成对 FP32 load/store 可合并 |
| `ProcessKQ` | 不改循环 | 每行从紧凑标量数组广播不同值，4-byte 行距不能用现有 block repeat 直接表达 |
| `ReduceSum64/128/VF` | 不改循环 | 每行需产生独立标量归约；DataCopy 只是 RAW 依赖链的一部分，且当前 K=128 热路径绕过 generic dispatch |

## 4. 构建结果

| 项目 | 结果 |
|---|---|
| 环境预检 `python scripts/check_npu_env.py --build-only` | PASS |
| README 方式 A，`FLA_NPU_SOC=ascend950` | PASS |
| 最终 wheel SHA256 | `57e9bf20f4dc3690873d7dded18883824286f5427477b776e82411764de88d22` |

## 5. 精度结果

| 测试 | 结果 |
|---|---|
| `bash run_atk.sh accuracy` | 8/8 PASS，执行失败 0 |
| `python fla/ops/ascendc/kda/recurrent_kda/tests/pta/test_accuracy.py` | PASS |
| `python torch_custom/fla_npu/test/test_npu_recurrent_kda.py` | 5 组场景 PASS |
| `FLA_NPU_RUN_OPERATOR_TESTS=1 FLA_NPU_RUN_LEGACY_TESTS=1 FLA_NPU_RUN_LARGE_SHAPE_TESTS=1 pytest -q tests/operators/recurrent_kda` | 27 passed |

PTA 覆盖 raw/safe gate、TND/BSND、FP16/BF16/FP32 gate/beta、V-first/K-first、连续与非连续 state、原位/非原位、470-slot state pool，以及两类预期 host 拦截。

## 6. 候选筛选

性能统一采用 ATK `NPU AVG Profiler Time`，不用 Python wall time。表中总耗时为 8 个固定 case 的 profiler 时间之和。

| 版本 | 轮次 | 总耗时均值 (us) | 相对开工前基线 |
|---|---:|---:|---:|
| 开工前当前源码 | 2 | 993.9492 | 基线 |
| 仅 Decay dual DataCopy | 2 | 984.9814 | 提升 0.90% |
| Decay + Delta dual DataCopy | 2 | 975.9634 | 提升 1.81% |
| 上述版本 + generic MatVecMul repeat | 3 | 986.8152 | 提升 0.72%，回退 |

`MatVecMul` 候选的 repeat 写法本身通过 arch35 CCE 完整构建，但当前固定 K=128 case 不进入该函数。加入后 case 7 连续退化，故不以不可达代码的理论收益替代实测选择。

## 7. 最终性能明细

| case | 开工前均值 (us) | 最终轮次 1 (us) | 最终轮次 2 (us) | 最终均值 (us) | 变化 |
|---:|---:|---:|---:|---:|---:|
| 0 | 10.9897 | 10.9265 | 11.0011 | 10.9638 | 提升 0.24% |
| 1 | 11.1284 | 11.1181 | 11.2074 | 11.1628 | 退化 0.31% |
| 2 | 24.6965 | 24.7597 | 25.0638 | 24.9118 | 退化 0.87% |
| 3 | 25.3355 | 25.6786 | 25.8265 | 25.7526 | 退化 1.65% |
| 4 | 84.8090 | 82.9378 | 82.8590 | 82.8984 | 提升 2.25% |
| 5 | 83.8213 | 83.6552 | 83.2271 | 83.4412 | 提升 0.45% |
| 6 | 377.3281 | 363.9409 | 370.3662 | 367.1536 | 提升 2.70% |
| 7 | 375.8408 | 366.4669 | 372.8920 | 369.6795 | 提升 1.64% |
| 合计 | 993.9492 | 969.4837 | 982.4431 | 975.9634 | 提升 1.81% |

两轮 performance 均为 8/8 SUCCESS。收益主要来自 B=16/64 的长 case；小 case 的约 0.3%～1.7% 波动接近 profiler 抖动范围，不作为独立优化结论。

## 8. 验收结论

- 精度：通过；
- 构建：通过；
- 性能：通过，最终两轮总耗时均低于开工前对应两轮，总均值提升 `1.81%`；
- 修改范围：仅 arch35 kernel 的两处 K=128 搬运合并，符合最小修改原则；
- 未提交、未推送。
