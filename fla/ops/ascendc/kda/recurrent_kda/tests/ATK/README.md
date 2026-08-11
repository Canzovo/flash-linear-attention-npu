# recurrent_kda ATK 接入

该目录固定验证四个 decode batch，每个 batch 包含两个模式，共 8 个 case：

- `base`：`ssm_state_indices=None`、`num_accepted_tokens=None`。
- `cb_mtp`：continuous batching 的 `ssm_state_indices` 非空，同时传入 MTP 的
  `num_accepted_tokens`（该参数按接口约束依赖 `ssm_state_indices`）。

所有 case 使用固定随机种子 `20260811`。输入从 CPU 侧按固定顺序生成后再搬到
NPU，保证 CPU reference 与 NPU 收到完全相同的数据。算子属性与 benchmark 保持一致。

运行命令：

```bash
bash run_atk.sh accuracy
bash run_atk.sh performance
bash run_atk.sh all
```

脚本在 ATK 命令结束后调用 `verify_results.py`，校验 8 个 case、失败用例、
精度结果和性能 custom data，避免 ATK 内部失败但命令返回 0 时产生误判。最终报告见
`TEST_REPORT.md`，稳定副本和逐 case 数据位于 `results/`。

性能任务使用 ATK `performance_device --enable_custom_data`。MFU/MBU 分别对应
ATK 报告中的 `calc_utilization(%)` 和 `mem_utilization(%)`。FLOPs 使用 decode
recurrent 主路径的下界模型，GM 搬运量统计全部输入读取、输出写回与 state 原位写回。
指数和 sigmoid 等超越函数不计入 FLOPs，因而 MFU 是保守估计。

