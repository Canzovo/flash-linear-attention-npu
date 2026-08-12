# recurrent_kda CPU/NPU/GPU ATK

本目录提供 recurrent_kda decode 的 CPU、NPU、GPU 双标杆精度与性能脚本。

## 测试矩阵

- 固定随机种子：`20260811`
- batch：`1`、`4`、`16`、`64`
- `base`：`ssm_state_indices=None`，`num_accepted_tokens=None`
- `cb_mtp`：同时启用 continuous batching 与 MTP
- 共 8 个 case，顺序与相邻 `ATK/all_recurrent_kda.json` 一致

基础 shape、dtype 与算子属性保持为：

- `S=1`，`H=96`，`K=128`，`V=128`
- `q/k/v/initial_state` 为 BF16
- `g/beta/A_log/dt_bias` 为 FP32
- `layout=BSND`、`state_v_first=True`
- kernel 内启用 Q/K L2Norm、gate 与 beta sigmoid

## 精度拓扑

`./run_atk.sh accuracy` 使用以下三路输出：

1. NPU DUT：`fla_npu.ops.ascendc.recurrent_kda`
2. GPU control：标准 FLA Triton `fused_recurrent_kda_fwd`
3. CPU benchmark：仓内 canonical PyTorch reference

CPU benchmark 保持原始输入 dtype，不做额外升精度。三路输出在
accuracy 任务中统一转为 FP32，再由 ATK 原生
`cv_fused_double_benchmark` 按下列比例阈值判定：

- 最大相对误差比：5
- 平均相对误差比：1.5
- 均方根误差比：1.5

CPU reference 来自仓内
`tests/reference/recurrent_kda_reference.py`。GPU callable 默认是：

```text
fla.ops.kda.fused_recurrent:fused_recurrent_kda_fwd
```

如 GPU 环境的 FLA 版本使用其他入口，可通过
`RECURRENT_KDA_GPU_CALLABLE=<module>:<callable>` 覆盖。NPU 与 GPU
不在同一主机时，需要在 `nodes_accuracy.yaml` 中为 GPU 节点补充 ATK
server 的 `host`、`port` 和 `output_path`。

## 执行

在具备对应后端和 ATK server 的环境执行：

```bash
./run_atk.sh accuracy
./run_atk.sh performance_npu
./run_atk.sh performance_gpu
./run_atk.sh performance_cpu
./run_atk.sh performance
./run_atk.sh all
```

`performance` 依次运行 NPU、GPU 和 CPU 的全部 8 个 case。
默认性能参数为 `5,5,5`（预热 5 次、采集 5 次、取后 5 次平均），可覆盖：

```bash
PERFORMANCE_DATA=1,1,1 ./run_atk.sh performance_gpu
```

固定 case 文件 `all_recurrent_kda.json` 已随目录交付；日常执行不会调用
`atk case` 重新生成。仅在测试矩阵发生变化时，才使用 YAML 和 generator
重新生成并更新固定 JSON。

当前提交只提供代码资产，尚未在 GPU 环境执行精度或性能验证。
