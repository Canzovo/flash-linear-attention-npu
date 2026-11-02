# Recurrent Gated Delta Rule 接口

## Python 入口

```python
from fla_npu.ops.ascendc import recurrent_gated_delta_rule

out = recurrent_gated_delta_rule(
    query, key, value, state,
    beta=beta,
    scale=1.0,
    actual_seq_lengths=actual_seq_lengths,
    ssm_state_indices=ssm_state_indices,
    num_accepted_tokens=None,
    g=None,
    gk=None,
    use_qk_l2norm_in_kernel=False,
    use_gate_in_kernel=False,
    A_log=None,
    dt_bias=None,
    use_beta_sigmoid_in_kernel=False,
    allow_neg_eigval=False,
)
```

返回连续的 BF16 attention 输出 `out`，并按 checkpoint 索引原地更新 `state`。State 保留调用方的存储、storage offset 和 stride，不复制整个 State pool。

本次完整融合能力适用于 A2/A3。A5 继续使用历史实现及其原有约束，不因增加 V2 符号而声明支持本次融合能力。

## 输入、输出与属性

所有非空 Tensor 位于同一 NPU，逻辑格式为 ND。`K=V=128`，`T>0`，`B>=1`，`Cstate>0`，`HK>0`，`HV>0`，`HV/HK` 是 `[1,16]` 内的整数。A2/A3 通用实现不增加 head 数上限。

| 参数 | 逻辑 shape / dtype | 语义 |
| --- | --- | --- |
| query、key | `[T,HK,128]` / BF16 | token-major 的 Q/K 逻辑视图 |
| value | `[T,HV,128]` / BF16 | V 逻辑视图 |
| state | `[Cstate,HV,128,128]` / BF16 或 FP32 | v-first State pool，最后两轴为 V、K |
| beta | `[T,HV]` / BF16 或 FP32 | 每 token、value head 的更新系数或 sigmoid 输入 |
| actual_seq_lengths | `[B+1]` / INT32 | `[prefix,L_0,...,L_(B-1)]`，不是累计长度数组 |
| ssm_state_indices | `[T]` 或 `[B,Ckpt]` / INT32 | 初始读槽与逐 token checkpoint 写槽 |
| num_accepted_tokens | 可选 `[B]` / INT32 | 每个非空请求的初始 checkpoint 下标加一，缺省为 1 |
| g | 可选 `[T,HV]` / BF16 或 FP32 | log decay 或 raw gate 输入 |
| gk | 保留参数 | V2 拒绝非空 gk；历史 ACLNN 入口保留原 gk 功能 |
| A_log | 可选 `[HV]` / FP32 | raw gate 的 head 参数，raw gate 开启时必需 |
| dt_bias | 可选 `[HV]` / FP32 | raw gate 的 head bias，缺省为 0 |
| scale | 有限实数，默认 1.0 | Q 的缩放；Python bool 不作为合法 scale |
| use_qk_l2norm_in_kernel | bool，默认 False | 在 FP32 中对 Q、K 分别做 L2 norm |
| use_gate_in_kernel | bool，默认 False | 用 g、A_log、dt_bias 生成 decay |
| use_beta_sigmoid_in_kernel | bool，默认 False | 对 beta 应用 sigmoid |
| allow_neg_eigval | bool，默认 False | 仅 sigmoid 开启时将 sigmoid 结果乘 2 |
| out | `[T,HV,128]` / BF16 | 有效 token 的 attention 输出 |

State、beta、g 的 BF16/FP32 选择相互独立，共 8 种 dtype 模板组合。g 缺省时内部选择 FP32 gate 模板，但不读取 g。

## 视图与别名

- Q/K/V 每个 stride 为正，最后一维 stride 为 1；token stride `sT>=128`，且 `(sT-128)*2<=UINT32_MAX`。支持混合 QKV backing storage、token/head 转置、padding、offset 和满足条件的只读重叠视图。
- State 每个 stride 为正，K 轴 stride 为 1，V 轴 stride `sV>=128`，且 `(sV-128)*sizeof(StateType)<=UINT32_MAX`。支持 row/head/slot padding、head/slot 置换和 offset。
- State 可写逻辑视图不得自重叠，不得与任何只读输入或 out 产生写读冲突；out 也不得与输入冲突。Q/K/V 之间的只读别名允许。
- Q/K/V 和 State 直接使用原始 descriptor 的逻辑首地址及 stride，不能通过连续化、整池克隆或拼接改变此契约。beta/g、head 参数和 metadata 可以连续化。
- Tensor 的可达地址、storage 边界、累计位置与字节数使用 64 位校验。DMA 的窄字段先以 64 位验证，再转换为对应字段类型。

## 序列与 checkpoint

要求 `prefix>=0`、所有 `L_b>=0`，且 `prefix+sum(L_b)=T`。metadata 的每个元素为 INT32，累计 token 位置与寻址为 64 位。请求 b 的首 token 为 `start_b=prefix+sum(L_j,j<b)`。

| indices 模式 | token r 写槽 | 初始读槽 | 非空请求 accepted 范围 |
| --- | --- | --- | --- |
| `[T]` | `indices[start_b+r]` | `indices[start_b+accepted_b-1]` | `[1,L_b]` |
| `[B,Ckpt]` | `indices[b,r]` | `indices[b,accepted_b-1]` | `[1,Ckpt]`，允许 accepted 大于 L_b |

二维表要求 `Ckpt>0` 且 `Ckpt>=max(L_b)`，与配套 CPU 标杆的正表宽约束一致；即使所有请求为空，也不接受 `[B,0]`。包括未使用部分在内的全部索引均位于 `[0,Cstate)`。

同请求可以重复写槽，最终值来自最后一次写入。不同请求的写集合不得相交，一个请求的初始读槽也不得被其他请求写入。同一 State slot 的不同 head 由各自任务负责。

空请求跳过，不读取 accepted 或 State。允许所有请求为空且所有 token 都属于 prefix。out 的 prefix 区域数值未指定；精度仅比较有效 token，State 需比较完整 pool，包括未写槽保持不变。

## 数学语义

中间计算、Q/K norm 和跨 token 递推均为 FP32。令 `hk=hv//(HV/HK)`：

```text
q, k, v = FP32(query[t,hk]), FP32(key[t,hk]), FP32(value[t,hv])
if use_qk_l2norm_in_kernel:
    q *= rsqrt(sum(q*q) + 1e-6)
    k *= rsqrt(sum(k*k) + 1e-6)
q *= scale
b = FP32(beta[t,hv])
if use_beta_sigmoid_in_kernel:
    b = sigmoid(b) * (2 if allow_neg_eigval else 1)
alpha = 1                                  # g 缺省
alpha = exp(FP32(g[t,hv]))                  # 有 g，raw gate 关闭
alpha = exp(-exp(A_log[hv])*softplus(g[t,hv]+bias)) # raw gate 开启
Sbar = alpha * S
delta = b * (v - Sbar @ k)
S = Sbar + delta[:,None] * k[None,:]
out[t,hv] = BF16(S @ q)
state[write_slot,hv] = cast_state_dtype(S)
```

归一化后的 Q/K 不重新量化。BF16 checkpoint 仅在写回时按 RNE 舍入，下一 token 和下一窗口继续使用未量化的 FP32 State，不回读 checkpoint。sigmoid 使用正负分支，softplus 保留负输入的小尾部；FP32 本身的非有限行为按 CPU 标杆语义比较，不通过缩小输入范围规避。

## ACLNN 与错误边界

历史 `aclnnRecurrentGatedDeltaRuleGetWorkspaceSize` / `aclnnRecurrentGatedDeltaRule` 原型和 ABI 保持不变。V2 新增 `aclnnRecurrentGatedDeltaRuleV2GetWorkspaceSize` / `aclnnRecurrentGatedDeltaRuleV2`，仅 V2 增加 A_log、dt_bias 和四个 bool 属性。完整声明见 [ACLNN header](../op_host/op_api/aclnn_recurrent_gated_delta_rule.h)。

两段式调用先获取 executor/workspace 大小，再提交对应 launch；旧入口与 V2 的 executor 不混用。Python Stable ABI 使用 V2。历史入口继续接受历史 gk，不把新增参数追加到旧 ABI。

Python 入口在提交计算前校验 descriptor、属性、metadata 值域和跨请求 State 所有权；metadata 原地修改后必须重新验证。直接 ACLNN 校验 descriptor 与属性，直接 C 调用方负责设备 metadata 的值域与 State 所有权。非法 shape、dtype、stride、slot、bool、scale、别名或 raw gate 缺少 g/A_log 时应在提交前报错，并保持 State 不变。

Host 参数拒绝通过 `op_common/log/log.h` 的结构化 Error Message 上报，调用方可用 `aclGetRecentErrMsg()` 获取错误码、参数名、实际值和期望条件；不依赖普通错误日志是否开启。V1/V2 保留原返回状态：必选 tensor 为空或 dtype 错误为 `161002`，workspaceSize/executor 指针为空为 `161001`；公开函数签名不变。V1 的 shape/tiling 错误仍按原阶段返回，当前直接接口检验中的 inference/tiling 失败为 `561103`，具体原因由结构化消息保留；不将 V2 的新增功能约束加到 V1。

| Error Message | 问题 |
| --- | --- |
| `EZ0004` | 必选 tensor、workspaceSize 或 executor 指针为空 |
| `EZ0006` | 不支持的输入 format |
| `EZ0007` | dtype 不匹配，附实际和支持的类型 |
| `EZ0008` / `EZ0009` | shape 不匹配，附期望 shape 或原因 |
| `EZ0011` / `EZ0013` | rank 或维度不匹配 |
| `EZ0024` / `EZ0027` | 属性、维度值或组合关系无效 |
| `EZ0028` | stride 不符合计算或写回要求 |
| `EZ0037` | raw gate 依赖、V2 gk、storage 越界、写别名等约束不满足 |
| `EZ9999` | executor、内部 view/输出分配或启动器失败 |

这些编号为结构化错误目录编号，与 ACLNN 返回状态分开核验。V2 descriptor 检查在 DFX phase-1 缓存查询之前执行，缓存命中也不能绕过 storage 和别名约束。直接接口检验脚本见 [check_error_messages.py](../../../../../../../tests/atk/recurrent_gated_delta_rule/scripts/check_error_messages.py)。

测试覆盖、执行方法和结果见 [ATK README](../../../../../../../tests/atk/recurrent_gated_delta_rule/README.md)。
