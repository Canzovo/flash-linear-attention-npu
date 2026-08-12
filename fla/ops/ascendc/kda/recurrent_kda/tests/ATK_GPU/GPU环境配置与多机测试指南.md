# recurrent_kda ATK_GPU 环境配置与多机测试指南

本文说明如何准备一台新的 NVIDIA GPU 环境，并由另一台 NPU 机器作为
ATK 控制端，完成 recurrent_kda 的 CPU、NPU、GPU 三路精度测试以及三种
后端的性能测试。

## 1. 推荐拓扑

```text
NPU 控制端
├── NPU DUT：fla_npu.ops.ascendc.recurrent_kda
├── CPU benchmark：tests/reference/recurrent_kda_reference.py
└── ATK client ──HTTP──> GPU ATK server
                         └── GPU control：FLA Triton fused_recurrent_kda_fwd
```

CPU benchmark 留在 NPU 控制端执行。GPU 环境只负责 GPU control 和 GPU
性能测试，不需要安装 CANN 或 `torch_npu`。

必须满足以下条件：

- NPU 控制端可以访问 GPU 环境的 ATK 服务端口，默认是 `9090`。
- 两端使用同一份 ATK wheel 或同一源码快照。本目录按 ATK `0.3.28`
  的接口编写，不建议两端混用不同版本。
- 两端都有同一版本的 NPU 仓代码，至少应完整包含 `ATK_GPU` 目录和
  `tests/reference/recurrent_kda_reference.py`。
- GPU 环境安装的是完整 CUDA 版 FLA，而不是本 NPU 仓中只有 AscendC
  路径的同名 `fla` 包。
- 两端使用本目录随代码交付的固定 `all_recurrent_kda.json`，不要日常调用
  `atk case` 重新生成。

本文命令和 YAML 中的 `<...>` 都是占位符，执行前必须替换成新环境的实际
值，不能原样复制执行。

## 2. GPU 驱动和 Python 环境

先确认 GPU 驱动正常：

```bash
nvidia-smi
```

建议创建独立的 Python 3.11 环境。当前 FLA 要求 Python 3.10 或更高版本；
独立环境也可以避免 CUDA 版 Triton、ATK 和 NPU 版 Python 包相互覆盖。

```bash
conda create -n atk_recurrent_kda_gpu python=3.11 -y
conda activate atk_recurrent_kda_gpu
python -m pip install --upgrade pip setuptools wheel
```

### 2.1 安装 CUDA 版 PyTorch 和 FLA

优先选定一个经过确认的 FLA commit，再从源码安装并保持该 commit 不变：

```bash
git clone https://github.com/fla-org/flash-linear-attention.git /path/to/fla-gpu
cd /path/to/fla-gpu
git checkout <PINNED_FLA_COMMIT>
python -m pip install -e '.[cuda]'
```

如果 GPU 机器已经有一套可用且满足 FLA 要求的 CUDA PyTorch，可先安装
匹配驱动的官方 PyTorch wheel，再执行上面的可编辑安装。不要为了追随
`main` 自动更新工作环境；应记录并固定实际验证过的 FLA commit。

也可以直接安装发布包：

```bash
python -m pip install 'flash-linear-attention[cuda]'
```

但发布包必须确实包含本测试需要的低层入口及参数：

```text
fla.ops.kda.fused_recurrent:fused_recurrent_kda_fwd
ssm_state_indices
num_accepted_tokens
state_v_first
inplace_final_state
use_qk_l2norm_in_kernel
use_gate_in_kernel
use_beta_sigmoid_in_kernel
allow_neg_eigval
lower_bound
```

官方安装说明：
<https://github.com/fla-org/flash-linear-attention/blob/main/INSTALL.md>

### 2.2 安装与 NPU 控制端一致的 ATK

推荐在一端构建 wheel，然后把同一个 wheel 安装到 NPU 和 GPU 环境：

```bash
cd /path/to/ATK-dev
python setup.py sdist bdist_wheel
python -m pip install /path/to/atk-0.3.28-*.whl
atk --version
```

如果使用源码直接安装，两端也必须来自同一份源码：

```bash
cd /path/to/ATK-dev
python -m pip install .
atk --version
```

## 3. 在 GPU 环境部署测试代码

把包含 `ATK_GPU` 修改的同一代码版本拉到 GPU 环境：

```bash
git clone <NPU_REPOSITORY_URL> /path/to/flash-linear-attention-npu
cd /path/to/flash-linear-attention-npu
git checkout <ATK_GPU_COMMIT>
```

设置便于后续复用的路径：

```bash
export NPU_REPO=/path/to/flash-linear-attention-npu
export ATK_GPU_DIR="${NPU_REPO}/fla/ops/ascendc/kda/recurrent_kda/tests/ATK_GPU"
export GPU_ATK_OUTPUT=/path/to/atk-gpu-output
mkdir -p "${GPU_ATK_OUTPUT}"
```

不要在 GPU 环境中执行本 NPU 仓的包安装命令，以免其中的 `fla` 抢先
覆盖完整 CUDA FLA。GPU 服务进程的 `PYTHONPATH` 也不要包含
`NPU_REPO`。执行器会根据自身路径定位 CPU reference，无需把仓根目录
加入 `PYTHONPATH`。

如果 CUDA FLA 使用源码可编辑安装，可以显式把它放在搜索路径最前面：

```bash
export FLA_GPU_SRC=/path/to/fla-gpu
export PYTHONPATH="${FLA_GPU_SRC}${PYTHONPATH:+:${PYTHONPATH}}"
```

## 4. 启动前检查

在 GPU 环境执行以下检查：

```bash
python - <<'PY'
import inspect
import torch
import triton
import fla
from fla.ops.kda.fused_recurrent import fused_recurrent_kda_fwd

required = {
    "ssm_state_indices",
    "num_accepted_tokens",
    "state_v_first",
    "inplace_final_state",
    "use_qk_l2norm_in_kernel",
    "use_gate_in_kernel",
    "use_beta_sigmoid_in_kernel",
    "allow_neg_eigval",
    "lower_bound",
}
parameters = set(inspect.signature(fused_recurrent_kda_fwd).parameters)
missing = sorted(required - parameters)

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("triton:", triton.__version__)
print("fla path:", fla.__file__)
print("CUDA available:", torch.cuda.is_available())
print("BF16 supported:", torch.cuda.is_bf16_supported())
print("GPU:", torch.cuda.get_device_name(0))
print("callable:", inspect.getfile(fused_recurrent_kda_fwd))
print("missing parameters:", missing)

assert torch.cuda.is_available()
assert torch.cuda.is_bf16_supported()
assert not missing
PY
```

检查结果中：

- `fla path` 和 `callable` 应指向完整 CUDA FLA 安装或固定的 FLA 源码。
- `CUDA available`、`BF16 supported` 都应为 `True`。
- `missing parameters` 应为 `[]`。

如果 callable 的模块路径不同，启动 GPU ATK server 前设置：

```bash
export RECURRENT_KDA_GPU_CALLABLE='<python_module>:<callable>'
```

默认值是：

```bash
export RECURRENT_KDA_GPU_CALLABLE='fla.ops.kda.fused_recurrent:fused_recurrent_kda_fwd'
```

## 5. 启动 GPU ATK server

ATK 控制端的 `-p executor_recurrent_kda_gpu.py` 只负责控制端插件加载，
不会自动替代服务端插件加载。GPU server 必须再通过 `--plugin_path`
显式加载同一执行器。

如果只使用一张物理 GPU，例如物理卡 3，对进程暴露该卡后，ATK 中仍使用
逻辑设备号 0：

```bash
export CUDA_VISIBLE_DEVICES=3
```

在 `tmux` 或其他可持续终端中启动服务：

```bash
conda activate atk_recurrent_kda_gpu
cd "${ATK_GPU_DIR}"

atk server \
  --name recurrent_kda_gpu \
  --host <GPU_BIND_IP> \
  --port 9090 \
  --devices 0 \
  --output_path "${GPU_ATK_OUTPUT}" \
  --plugin_path "${ATK_GPU_DIR}/executor_recurrent_kda_gpu.py" \
  --timeout 2000
```

启动命令必须在设置好 `CUDA_VISIBLE_DEVICES`、`PYTHONPATH` 和
`RECURRENT_KDA_GPU_CALLABLE` 后执行，因为 ATK worker 会继承 server
进程的环境变量。

安全注意：ATK server 提供任务执行和文件传输接口，不应暴露到公网或不可信
网络。优先绑定受信任的内网地址，并用防火墙只允许 NPU 控制端访问端口
`9090`。如果必须绑定 `0.0.0.0`，务必在外层限制访问源。

### 容器中的 GPU 环境

如果 GPU 环境位于容器内，还需要把端口发布到 NPU 控制端可达的宿主机地址，
并把代码、FLA 源码和输出目录挂载到容器。节点 YAML 中填写宿主机可达地址，
不要填写容器内的 `127.0.0.1` 或只在 Docker bridge 内可见的地址。例如：

```text
容器 9090 端口 -> GPU 宿主机 9090 端口 -> 仅允许 NPU 控制端访问
```

## 6. 从 NPU 控制端检查连通性

在 NPU 控制端执行：

```bash
curl --fail --show-error \
  'http://<GPU_SERVICE_IP>:9090/api/server/get_server_args'
```

返回 JSON 中应能看到 server 名称、设备列表 `[0]` 和 GPU 侧输出目录。
如果这里不通，先处理路由、容器端口映射或防火墙，不要直接运行全量 ATK。

## 7. 修改 NPU 控制端节点配置

只需在 NPU 控制端修改节点 YAML。`host` 必须是 NPU 控制端实际可达的
GPU 服务地址，`output_path` 必须是 GPU 环境内部可写的路径，并与 server
启动参数一致。

### 7.1 三路精度配置

将 `nodes_accuracy.yaml` 配置为：

```yaml
nodes:
  - name: npu_dut
    backend: npu
    devices: [0]
    task: [accuracy]
    is_compare: true

  - name: gpu_control
    backend: gpu
    host: <GPU_SERVICE_IP>
    port: 9090
    devices: [0]
    output_path: <GPU_ATK_OUTPUT_IN_GPU_ENV>
    download_transport: http
    task: [accuracy]
    is_compare: true

  - name: cpu_benchmark
    backend: cpu
    task: [accuracy]
    is_compare: false
```

该拓扑中 CPU 是 `cv_fused_double_benchmark` 的 canonical benchmark，
NPU 和 GPU 都是待比较输出。`run_atk.sh` 已固定使用 `--bm_device cpu`，
不要改成 GPU benchmark。

`download_transport: http` 会通过同一个 ATK 服务端口回传精度输出，不要求
配置 GPU 机器的 SSH 密钥。当前输出数据量较小，推荐保持该设置。

### 7.2 GPU 性能配置

将 `nodes_performance_gpu.yaml` 配置为：

```yaml
nodes:
  - name: gpu_performance
    backend: gpu
    host: <GPU_SERVICE_IP>
    port: 9090
    devices: [0]
    output_path: <GPU_ATK_OUTPUT_IN_GPU_ENV>
    download_transport: http
    task: [performance_device]
```

`nodes_performance_npu.yaml` 和 `nodes_performance_cpu.yaml` 保持本地节点配置
即可。

## 8. 分步执行

所有 ATK client 命令都在 NPU 控制端的 `ATK_GPU` 目录执行。

### 8.1 单 case 精度烟测

先只跑 case 0，确认 NPU、CPU、GPU 三路执行和远程输出回传正常：

```bash
atk task \
  -c all_recurrent_kda.json \
  -n nodes_accuracy.yaml \
  --task accuracy \
  --bm_device cpu \
  -p executor_recurrent_kda_gpu.py \
  -s 0 \
  -e 1 \
  -sp \
  -to 2000
```

### 8.2 单 case GPU 性能烟测

首次执行可能触发 Triton JIT 编译，因此保留较大的任务超时。烟测可使用较小
的性能参数：

```bash
atk task \
  -c all_recurrent_kda.json \
  -n nodes_performance_gpu.yaml \
  --task performance_device \
  --performance_data 1,1,1 \
  --enable_custom_data \
  -p executor_recurrent_kda_gpu.py \
  -s 0 \
  -e 1 \
  -sp \
  -to 2000
```

### 8.3 全量精度和性能

烟测通过后执行固定的 8 个 case：

```bash
./run_atk.sh accuracy
./run_atk.sh performance_gpu
./run_atk.sh performance_npu
./run_atk.sh performance_cpu
```

也可以一次执行全部任务：

```bash
./run_atk.sh all
```

默认性能参数为 `5,5,5`（预热 5 次、采集 5 次、取后 5 次平均）。执行器通过 `--enable_custom_data` 提供计算量和
访存量。测试前应确认 GPU 没有其他高负载任务，且不要把 Triton 首次 JIT
编译耗时当作算子耗时。

已知限制：当前 ATK `0.3.28` 仅在 NPU backend 中消费
`export_custom_data()` 并结合 Ascend 硬件峰值计算利用率。GPU 和 CPU
backend 会接受 `--enable_custom_data`，但原生 `performance_device`
报告不会因此自动生成 MFU、MBU；GPU 性能测试仍会使用 CUDA profiler
统计设备耗时。若验收要求 GPU MFU/MBU，还需要另行扩展 ATK GPU backend
或增加后处理，并为实际 GPU 型号配置 BF16 峰值算力和显存峰值带宽。这个
问题不能通过安装依赖或修改节点 YAML 解决。

固定随机种子 `20260811` 已写入交付 JSON 和执行器。日常执行不要调用
`atk case`，否则可能改变 case 内容或顺序。

## 9. 结果检查

至少检查以下内容：

- 精度任务共 8 个 case，三路后端都没有 `ERROR`。
- NPU 和 GPU 都按 `cv_fused_double_benchmark` 与 CPU benchmark 比较。
- 精度报告中最大相对误差比、平均相对误差比、均方根误差比均达到 YAML
  标准。
- NPU、GPU、CPU 性能报告各包含 8 个 case 的耗时。
- NPU 报告按当前 ATK 能力检查 MFU、MBU；GPU/CPU 不把原生报告缺少 MFU、MBU 判定为环境失败。
- NPU 控制端汇总报告与 GPU server 日志中的 case 数量一致。

建议记录以下版本信息并随测试报告保存：

```bash
git rev-parse HEAD
atk --version
python -c 'import torch; print(torch.__version__, torch.version.cuda)'
python -c 'import triton; print(triton.__version__)'
python -c 'import fla; print(fla.__file__)'
git -C /path/to/fla-gpu rev-parse HEAD
```

## 10. 常见问题

### `ModuleNotFoundError: fla.ops.kda`

GPU 环境加载了本 NPU 仓中的不完整 `fla`，或者没有安装完整 CUDA FLA。
检查 `fla.__file__`，移除 `PYTHONPATH` 中的 NPU 仓根目录，并重启 ATK
server。

### `configured GPU target is not callable`

当前 FLA 的模块或函数名与默认值不同。先用 Python 直接导入目标函数，再通过
`RECURRENT_KDA_GPU_CALLABLE=<module>:<callable>` 配置，之后重启 server。

### callable 缺少 `ssm_state_indices` 或 `num_accepted_tokens`

FLA 版本过旧，不支持本目录的 continuous batching 与 MTP case。切换到包含
这些接口语义的固定 commit，不能通过删除参数绕过测试。

### `CUDA available: False` 或 BF16 不支持

检查容器是否正确传入 GPU、驱动是否可见，以及 PyTorch wheel 是否为匹配
驱动的 CUDA 版本。ATK server 必须从修复后的同一 Conda 环境重新启动。

### NPU 侧提示连接拒绝或超时

依次检查 GPU server 进程、`9090` 监听地址、容器端口发布、防火墙和节点
YAML 的 `host`。先让 `/api/server/get_server_args` 的 `curl` 检查通过。

### GPU 侧找不到自定义执行器注册名

确认 server 启动命令包含
`--plugin_path "${ATK_GPU_DIR}/executor_recurrent_kda_gpu.py"`。仅在 NPU
client 命令中传 `-p` 不够；修改插件或环境变量后也必须重启 server。

### 精度输入不一致

不要修改固定 JSON 中的 seed，不要重新生成 case。执行器在每个后端都使用
CPU Generator 和固定 seed 构造相同输入，再搬运到目标设备。还应确认两端
的 `executor_recurrent_kda_gpu.py` 文件来自同一 commit。

### GPU 性能首轮异常偏慢

首次运行通常包含 Triton JIT 编译。先完成单 case 烟测，使 kernel 编译缓存
就绪，再运行默认 `5,5,5` 的正式性能任务；同时检查是否有其他 GPU 进程
竞争资源。
