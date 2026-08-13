# RecurrentKda 性能优化迭代记录

## 1. 验证口径

- 固定执行 `bash run_atk.sh accuracy` 和 `bash run_atk.sh performance`。
- performance 覆盖 8 个固定 case；以 8 case Device profiler 时延总和比较上一版。
- 收益较小时至少复跑一轮；两轮均提升后才保留代码并提交。
- `profile` 覆盖 base case 0/2/4/6，用于观察 `kernel_details.csv` 的 AIV 指标。
- 单次 `aiv_vec_ratio` 受核负载采样影响，只作为选点线索，不替代完整 performance 结论。
- 每轮只提交确认提升的代码和记录，不 push；无提升实验回退后只在最终总结中记录。

## 2. 第0轮：0574a72 基线

### 2.1 Performance

`verified performance: 8/8 SUCCESS`。

| case | B | 模式 | Device 时延 (us) |
|---:|---:|---|---:|
| 0 | 1 | base | 13.0094 |
| 1 | 1 | cb_mtp | 13.5050 |
| 2 | 4 | base | 32.5244 |
| 3 | 4 | cb_mtp | 33.1626 |
| 4 | 16 | base | 114.2544 |
| 5 | 16 | cb_mtp | 114.6966 |
| 6 | 64 | base | 478.6105 |
| 7 | 64 | cb_mtp | 477.6303 |
| **合计** | - | - | **1277.3932** |

### 2.2 Profile

实际 `Block Num=56`，与早期理论文档按72核估算不同。

| case | Duration (us) | aiv_vec_ratio | aiv_scalar_ratio | aiv_mte2_ratio | aiv_mte3_ratio |
|---:|---:|---:|---:|---:|---:|
| 0 | 13.628 | 0.508 | 0.534 | 0.196 | 0.051 |
| 2 | 33.074 | 0.681 | 0.477 | 0.264 | 0.068 |
| 4 | 115.010 | 0.330 | 0.201 | 0.125 | 0.033 |
| 6 | 476.039 | 0.716 | 0.423 | 0.457 | 0.106 |

## 3. 第1轮：按当前 head 局部搬运 Beta

### 3.1 设计与实现

原 arch35 热路径为每个 `(sequence, head)` 任务读取完整 `[L, NV]` Beta，
但实际只消费当前 head 的 `L` 个标量。本轮改为：

1. 每个 token 只从 GM 读取当前 head 的一个 Beta；
2. 用 `DataCopyPad` 将每个标量填充为32B UB 行；
3. FP32 每行8元素，BF16/FP16每行16元素；
4. `LoadBeta` 按 token 本地偏移和 dtype 对齐步长读取；
5. 保留原有队列、V→S 同步、sigmoid 和数学顺序。

FP32 ATK 路径的每 token/head Beta GM 指令有效范围由384B降至32B。

### 3.2 构建与精度

- README 方式A ascend950 wheel：PASS。
- Wheel SHA256：`9c87a01af45e2a1657b09ae08243b794da49d6ccf75629510f2eea6fccea0791`。
- `bash run_atk.sh accuracy`：`verified accuracy: 8/8 PASS`。

### 3.3 Performance

两次独立运行均为 `verified performance: 8/8 SUCCESS`。

| case | 第0轮 (us) | 第1轮首次 (us) | 第1轮复跑 (us) |
|---:|---:|---:|---:|
| 0 | 13.0094 | 12.6934 | 13.1421 |
| 1 | 13.5050 | 13.2562 | 13.1822 |
| 2 | 32.5244 | 32.6349 | 32.4333 |
| 3 | 33.1626 | 33.0445 | 32.8570 |
| 4 | 114.2544 | 113.9545 | 114.0110 |
| 5 | 114.6966 | 114.6835 | 114.7423 |
| 6 | 478.6105 | 476.6274 | 477.4259 |
| 7 | 477.6303 | 477.7137 | 477.3595 |
| **合计** | **1277.3932** | **1274.6081** | **1275.1533** |

- 首次相对第0轮提升：`0.2180%`。
- 复跑相对第0轮提升：`0.1754%`。
- 结论：两轮总时延均下降，保留并提交。

### 3.4 Profile

| case | Duration (us) | aiv_vec_ratio | aiv_scalar_ratio | aiv_mte2_ratio | aiv_mte3_ratio |
|---:|---:|---:|---:|---:|---:|
| 0 | 14.398 | 0.499 | 0.555 | 0.205 | 0.050 |
| 2 | 33.289 | 0.130 | 0.095 | 0.049 | 0.013 |
| 4 | 117.992 | 0.740 | 0.461 | 0.299 | 0.076 |
| 6 | 473.464 | 0.715 | 0.426 | 0.460 | 0.107 |

B4/B16 的单次 ratio 与接近不变的 kernel 时延不一致，说明该列受本次单样本和
不同核负载影响明显。后续轮次继续采集，但以完整8-case performance 的稳定复跑为准。
