# recurrent_kda ATK 精度与性能测试报告

## 结论

- ATK 精度测试 8/8 通过；ATK 性能测试 8/8 执行成功，失败用例为 0。
- 固定随机数种子为 `20260811`，未重新编译算子。
- 精度范围：最大相对误差 `6.097561e-3`～`7.751938e-3`，平均相对误差不超过 `1.356033e-6`，RMSE 不超过 `1.783712e-7`。
- 性能范围：设备耗时 `13.0586`～`505.8273 us`，MFU `0.283671%`～`0.546794%`，MBU `28.422053%`～`54.393037%`。

## 测试范围与方法

- 公共基础属性：`layout=BSND`、`B∈{1,4,16,64}`、`S=1`、`H=96`、`K=V=128`。
- 算子属性：`output_final_state=False`、`inplace_final_state=True`、`use_qk_l2norm_in_kernel=True`、`use_gate_in_kernel=True`、`use_beta_sigmoid_in_kernel=True`、`allow_neg_eigval=False`、`safe_gate=True`、`lower_bound=-5.0`、`state_v_first=True`。
- `base`：`ssm_state_indices=None` 且 `num_accepted_tokens=None`。
- `cb_mtp`：continuous batching 与 MTP 组合，`ssm_state_indices` 为逆序状态索引，`num_accepted_tokens` 全为 1。四个 batch 各覆盖 `base` 与 `cb_mtp`，共 8 个 case。
- CPU 标杆使用 `tests/reference/recurrent_kda_reference.py`；NPU 使用已安装的 `fla_npu.ops.ascendc.recurrent_kda`。
- 精度判定为 `torch.allclose(rtol=0.02, atol=0.01)`；相对误差分母为 `max(abs(CPU), 1e-6)`。
- 性能使用 ATK `performance_device`：20 次预热、20 次 profiler 采样，取最后 10 次统计。耗时为 ATK profiler 的 Device 平均耗时，不使用 Python wall time。

## 精度结果

| case | B | 模式 | 最大相对误差 | 平均相对误差 | RMSE | 结果 |
|---:|---:|---|---:|---:|---:|---|
| 0 | 1 | base | 6.097560748e-03 | 1.356032385e-06 | 6.882814318e-08 | PASS |
| 1 | 1 | cb_mtp | 6.097560748e-03 | 1.356032385e-06 | 6.882814318e-08 | PASS |
| 2 | 4 | base | 6.849315017e-03 | 4.113170462e-07 | 3.441696705e-08 | PASS |
| 3 | 4 | cb_mtp | 7.575757802e-03 | 6.999530910e-07 | 3.871530296e-08 | PASS |
| 4 | 16 | base | 7.692307699e-03 | 6.630966141e-07 | 7.814242053e-08 | PASS |
| 5 | 16 | cb_mtp | 7.692307699e-03 | 9.263224001e-07 | 9.149257352e-08 | PASS |
| 6 | 64 | base | 7.751937956e-03 | 7.707031386e-07 | 1.783711525e-07 | PASS |
| 7 | 64 | cb_mtp | 7.751937956e-03 | 7.104688962e-07 | 9.311273175e-08 | PASS |

## 性能结果

| case | B | 模式 | Device 耗时 (us) | MFU (%) | MBU (%) |
|---:|---:|---|---:|---:|---:|
| 0 | 1 | base | 13.0586 | 0.288676 | 28.923483 |
| 1 | 1 | cb_mtp | 13.2890 | 0.283671 | 28.422053 |
| 2 | 4 | base | 31.9610 | 0.471789 | 46.999467 |
| 3 | 4 | cb_mtp | 31.9865 | 0.471412 | 46.962057 |
| 4 | 16 | base | 110.3073 | 0.546794 | 54.393037 |
| 5 | 16 | cb_mtp | 110.6434 | 0.545133 | 54.227875 |
| 6 | 64 | base | 505.8273 | 0.476964 | 47.429519 |
| 7 | 64 | cb_mtp | 505.0225 | 0.477724 | 47.505161 |

## MFU / MBU 口径

- 逻辑 FLOPs 下界：`B × H × (7 × K × V + 2 × V + 7 × K)`；包含状态衰减、两次状态向量乘、delta、外积更新、Q/K 归一化和 query 缩放，不计 exp/sigmoid 等超越函数。
- 逻辑读流量为所有输入张量各读一次；逻辑写流量为输出加原地状态写回。该口径是算法流量模型，不是 HBM 硬件计数器实测字节数。
- `MFU = (逻辑 FLOPs / Device 耗时) / 295 TFLOPS × 100%`。
- `MBU = (逻辑读写字节 / Device 耗时) / 1600 GB/s × 100%`。
- ATK 对 `Ascend950PR` 未命中专用峰值表，本次按其日志明确回退的默认峰值 `295 TFLOPS` 与 `1600 GB/s` 计算；因此 MFU/MBU 应按该统一口径解读。

## 交付与原始证据
- 固定 ATK 用例：`all_recurrent_kda.json`（8 个 case，运行时不再重新生成）

- ATK 精度 XLSX：`results/atk_accuracy_report.xlsx`
- ATK 性能 XLSX：`results/atk_performance_report.xlsx`
- 精度逐 case 指标：`results/accuracy_metrics.jsonl`
- 性能逐 case 指标：`results/performance_metrics.jsonl`
- 执行入口：`bash run_atk.sh accuracy`、`bash run_atk.sh performance` 或 `bash run_atk.sh all`。
