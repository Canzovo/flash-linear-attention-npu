# ChunkKdaFwd API

## Python 主入口

```python
from fla_npu.ops.ascendc import chunk_kda_fwd

outputs = chunk_kda_fwd(
    q, k, v, g, beta, scale, chunk_size,
    layout="BSND",
    initial_state=None,
    output_final_state=False,
    cu_seqlens=None,
    chunk_indices=None,
    safe_gate=False,
    lower_bound=None,
    use_gate_in_kernel=False,
    A_log=None,
    dt_bias=None,
    disable_recompute=False,
    return_intermediate_states=False,
    state_v_first=False,
)
```

返回：

```text
(attn_out, final_state, gk, Aqk, Akk, w, u, qg, kg, v_new, h, initial_state)
```

可选输出在 Python 层返回 `None`。`Aqk/Akk` 始终存在；其余保留策略见算子 README。

## aclnn

```cpp
aclnnStatus aclnnChunkKdaFwdGetWorkspaceSize(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *g,
    const aclTensor *beta,
    const aclTensor *aLogOptional,
    const aclTensor *dtBiasOptional,
    const aclTensor *initialStateOptional,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    const char *layout,
    double scale,
    int64_t chunkSize,
    bool outputFinalState,
    bool safeGate,
    double lowerBound,
    bool useGateInKernel,
    bool disableRecompute,
    bool returnIntermediateStates,
    bool stateVFirst,
    const aclTensor *attnOut,
    const aclTensor *finalStateOut,
    const aclTensor *gkOut,
    const aclTensor *aqkOut,
    const aclTensor *akkOut,
    const aclTensor *wOut,
    const aclTensor *uOut,
    const aclTensor *qgOut,
    const aclTensor *kgOut,
    const aclTensor *vNewOut,
    const aclTensor *hOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnChunkKdaFwd(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

aclnn 原型使用 `OPTIONAL_INPUT/OPTIONAL_OUTPUT` 对应的空指针表达可选项：

- `finalStateOut` 仅在 `outputFinalState=true` 时必需。
- `gkOut` 在 `useGateInKernel=false` 或 `disableRecompute=true` 时必需。
- `wOut/uOut/qgOut/kgOut/vNewOut/hOut` 在 `disableRecompute=true` 时必需。
- `hOut` 在 `returnIntermediateStates=true` 时必需。
- `aqkOut/akkOut` 始终必需。

## 输入与输出布局

`layout` 只解释 q/k/v/g/beta 输入。输出固定为：

- `attnOut`: BSND 或 TND。
- `finalStateOut`: `[N,H_v,K,V]` 或 `stateVFirst=true` 时 `[N,H_v,V,K]`。
- 所有反向中间量：BNSD/NTD；`hOut` 的末两维服从 `stateVFirst`。

完整 Shape 表见 [KDA 模型符号表](../../README.md#model-shape-symbols)。

## Gate 语义

```text
useGateInKernel=false:
    gate = g
useGateInKernel=true, safeGate=false:
    gate = -exp(A_log) * softplus(g + dt_bias)
useGateInKernel=true, safeGate=true:
    gate = lowerBound * sigmoid(exp(A_log) * (g + dt_bias))
gk = chunk_local_cumsum(gate) / ln(2)
```

`safeGate` 的 true/false 都支持；`useGateInKernel=false` 时仍支持 `safeGate=true` 的后续稳定计算路径。

## 示例

```python
import torch
from fla_npu.ops.ascendc import chunk_kda_fwd

B, T, H, K, V = 1, 128, 4, 128, 128
q = torch.randn(B, T, H, K, device="npu", dtype=torch.float16)
k = torch.randn_like(q)
v = torch.randn(B, T, H, V, device="npu", dtype=torch.float16)
g = -torch.rand(B, T, H, K, device="npu", dtype=torch.float32) * 0.01
beta = torch.rand(B, T, H, device="npu", dtype=torch.float32)

attn_out, final_state, *_ = chunk_kda_fwd(
    q, k, v, g, beta, K ** -0.5, 64,
    layout="BSND",
    output_final_state=True,
    safe_gate=True,
)
assert attn_out.shape == (B, T, H, V)
assert final_state.shape == (B, H, K, V)
```

## 调用途径

| 路径 | 入口 |
| --- | --- |
| 稳定 Python | `fla_npu.ops.ascendc.chunk_kda_fwd` |
| aclnn | `aclnnChunkKdaFwdGetWorkspaceSize/aclnnChunkKdaFwd` |
| legacy | 显式加载后的 `torch.ops.npu.npu_chunk_kda_fwd` |
| 受限直调样例 | `torch.ops.ascend_ops.chunk_kda_fwd_direct` |

直调样例仅覆盖 dense BNSD、K=128、V=128/256，并保留“调用方传入已累计 gk”的低层测试接口；
公开顶层语义以稳定 Python/aclnn 接口为准。
