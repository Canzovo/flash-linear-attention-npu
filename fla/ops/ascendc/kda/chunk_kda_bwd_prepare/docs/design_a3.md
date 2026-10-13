# KernelA / chunk_kda_bwd_prepare：A3（Ascend910_93）迁移设计

更新日期：2026-10-12。状态：已完成文档静态审查与容量核算，尚未进行 A3 编译、精度测试或 profiling。

本文按用户指定，以 `KernelA设计文档.md` 和当前 `../op_kernel/arch35/` A5 源码为标杆。保持一个 head 由一个 AIV 完整处理、head generation 奇偶分流、4-head 工作项、A→Q→D 公式顺序，以及各资源按最后消费者完成后复用的原则。用户的明确标杆要求优先于仓库 R01 的半片映射；本文不采用同一 head 横切到两个 AIV 的方案。

源码与原文档不一致的项目在第一部分列明：L1 驻留份数、dAqk_raw dtype 和 state 索引以当前源码为准；A3 的 Q/D 直接写 GM 沿用原设计文档的输出方式，替代无法继续使用的片上 hand-off 路径。数学公式、输入输出 dtype、dense/varlen 和 state_v_first 语义保持统一新接口约定。

**本方案每 AIV 的自有 UB payload 为 80 KiB，按 192 KiB 预算有 112 KiB payload 余量；每 AIC 使用 L1 288 KiB、L0A/L0B 各 64 KiB、L0C 64 KiB；dAqk_raw GM 中转使用每个实际启动 AIC block 64 KiB user workspace。** 80 KiB 不包含 ND2NZ API 可能要求的临时 UB；如果目标 CANN 要求为 GM→L1 ND2NZ 预留 8 KiB，且该临时区与 AIV UB 共享预算，则对应峰值按 88 KiB、可用余量按 104 KiB 重新核算；如果由 AIC 或独立硬件区承担，则从对应资源单列扣除，AIV 的 112 KiB 仍只表示自有 payload 余量。原文档的每 AIV 64 KiB 和当前源码的每 AIV 128 KiB 同样都能放入 192 KiB，但都必须把平台 API 的额外临时区单独计入最终验收。

## 第零部分：chunk_kda_bwd_prepare 接口设计

本节定义 chunk_kda_bwd_prepare 的统一接口、数学契约、布局属性和 Device 参数。接口只有一组签名，支持 BNSD、BSND、NTD 和 TND 四种 token 输入布局；A3 先实现四种布局，后续 A5 按同一接口和布局契约适配。layout 描述 aqk、v_new、d_o 的物理存储，h 保持 chunk-major 布局，公开输出统一使用 head-major 布局。接口能力由 Host 按 SoC 校验，平台差异只存在于 tiling、搬运和 kernel 实现。

### 命名与数学契约

```text
算子目录 / 编译名：chunk_kda_bwd_prepare
OpDef：             ChunkKdaBwdPrepare
L0：                l0op::ChunkKdaBwdPrepare
aclnn：             aclnnChunkKdaBwdPrepareGetWorkspaceSize
                   aclnnChunkKdaBwdPrepare
Python：            chunk_kda_bwd_prepare

prepare := (dAqk, dv, dq_raw)
```

输出的 dv 和 dq_raw 是后续 Kernel 继续消费的部分梯度，不表示完整反向计算的最终结果。

```text
dAqk_raw[C,C] = d_o[C,V] @ v_new[C,V]^T

dAqk[r,c] = dAqk_raw[r,c] * scale,  c <= r
           = 0,                         c > r

dv[C,V] = Aqk[C,C]^T @ d_o[C,V]

H_right[V,K] = h_chunk^T, stateVFirst=false，h_chunk 物理布局为 [K,V]
             = h_chunk,   stateVFirst=true， h_chunk 物理布局为 [V,K]

dq_raw[C,K] = d_o[C,V] @ H_right[V,K]
```

h_chunk 表示当前 batch/chunk/head 的入口 state，按 StateOffset 定位；上式 c 只表示矩阵列。尾 chunk 的实际公式以 1.4 的有效长度 m 为准。

### Torch 接口

```python
from collections.abc import Sequence
import torch

def chunk_kda_bwd_prepare(
    aqk: torch.Tensor,
    v_new: torch.Tensor,
    d_o: torch.Tensor,
    h: torch.Tensor,
    *,
    scale: float,
    chunk_size: int = 64,
    state_v_first: bool = False,
    layout: str,
    cu_seqlens: Sequence[int] | None = None,
    chunk_indices: Sequence[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ...
```

layout 显式且必选，取值只接受 BNSD、BSND、NTD、TND；nullptr、空字符串和未知字符串均返回参数错误。先检查 cu_seqlens 与 chunk_indices 必须同时为空或同时有效，两者均为 Host 整数 sequence；BNSD/BSND 只能用于 metadata 为空的 dense 调用，NTD/TND 只能用于 metadata 成对有效的 packed 调用。不能根据 shape 猜测 layout，也不提供 layout 默认值。三份 token 输入必须使用同一个 layout，rank 和语义维度须满足下表。

| layout | aqk | v_new / d_o | h（state_v_first=false） | h（state_v_first=true） |
| --- | --- | --- | --- | --- |
| BNSD | [B,NV,T,64] | [B,NV,T,128] | [B,NT,NV,K,V] | [B,NT,NV,V,K] |
| BSND | [B,T,NV,64] | [B,T,NV,128] | 同 BNSD | 同 BNSD |
| NTD | [NV,T,64] | [NV,T,128] | [totalChunks,NV,K,V] | [totalChunks,NV,V,K] |
| TND | [T,NV,64] | [T,NV,128] | 同 NTD | 同 NTD |

其中 C=64、K=V=128、NT=ceil(T/64)。BNSD/BSND 是 rank-4 dense，NTD/TND 是 rank-3 packed；packed 的 T 是所有序列 token 总数，cu_seqlens 和 chunk_indices 切分 canonical chunk。dense 的 h chunk 维为 NT，packed 的 h chunk 维为 totalChunks。

aqk、v_new、d_o、h 均为 BF16；dAqk、dq_raw 为 FP32，dv 为 BF16。四份输入和三份输出均须物理连续，三份 token 输入的语义 B/NV/T 一致。B、NV、T 必须为正；packed 内可包含空序列，但不接受总 token 数为零的调用。cu_seqlens 含 2..1025 个整数，首项为 0、末项为 T、单调不减；chunk_indices 是长度为 2*totalChunks 的扁平整数数组，严格按 sequence-major 的 (seq,localChunk) 顺序列出全部有效 chunk，不允许重排、重复或遗漏。上述规则是 A3 及后续 A5 共用的新输入契约。

输出固定为 head-major，不随输入 layout 改变：

```text
dense:
    dAqkOut [B,NV,T,64]
    dvOut   [B,NV,T,128]
    dqRawOut[B,NV,T,128]

packed:
    dAqkOut [NV,T,64]
    dvOut   [NV,T,128]
    dqRawOut[NV,T,128]
```

因此 BSND/TND 输入不能继续用“输出 shape 与输入 shape 完全相同”表达输出契约；Host 必须按 layout 解析语义维度，再校验固定的 BNSD/NTD 输出 shape。三份 token 输入须为物理连续 tensor，不能隐式插入整张量 transpose。

### L2 aclnn 接口

原 aclnn 符号直接扩展 `layout` 参数，接口只有一组，不保留旧 C ABI 或第二套符号。`layout` 位于两个 metadata 参数之后、`scale` 之前；第二阶段执行第一阶段返回的 executor。

```cpp
aclnnStatus aclnnChunkKdaBwdPrepareGetWorkspaceSize(
    const aclTensor *aqk, const aclTensor *vNew, const aclTensor *dO,
    const aclTensor *h, const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional, const char *layout,
    double scale, int64_t chunkSize, bool stateVFirst,
    const aclTensor *dAqkOut, const aclTensor *dvOut,
    const aclTensor *dqRawOut, uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnChunkKdaBwdPrepare(
    void *workspace, uint64_t workspaceSize,
    aclOpExecutor *executor, aclrtStream stream);
```

Host 先验证 layout、metadata 成对性、rank、语义 shape、连续性和 canonical chunk 内容，再构造 executor。layout 不通过 tensor 或 GM 传递；它作为 OpDef string attr 写入 tiling。

### L0 接口

L0 只提供带 layout 的统一签名。

```cpp
namespace l0op {
const std::array<const aclTensor *, 3> ChunkKdaBwdPrepare(
    const aclTensor *aqk, const aclTensor *vNew, const aclTensor *dO,
    const aclTensor *h, const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional, const char *layout,
    double scale, int64_t chunkSize, bool stateVFirst,
    const aclTensor *dAqkOut, const aclTensor *dvOut,
    const aclTensor *dqRawOut, aclOpExecutor *executor);
}
```

`cuSeqlensOptional` 和 `chunkIndicesOptional` 仍转换为设备侧 INT64 Tensor；layout 通过 OpDef string attr 传递，不作为 Device GM 地址。

### Device 入口与属性

Device 入口地址顺序不变：

```cpp
chunk_kda_bwd_prepare(
    GM_ADDR aqk, GM_ADDR vNew, GM_ADDR dO, GM_ADDR h,
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
    GM_ADDR dAqk, GM_ADDR dv, GM_ADDR dqRaw,
    GM_ADDR workspace, GM_ADDR tiling);
```

解析后的 layout、scale、chunkSize 和 stateVFirst 由 Host 写入对应平台的 tiling data。本次 A3 tiling key 的模板参数为 `INPUT_LAYOUT` 和 `IS_VARIABLE`；`IS_VARIABLE` 对应 tiling data 的 isVariable，0 表示 dense，1 表示 packed。`INPUT_LAYOUT` 与 `IS_VARIABLE` 存在约束，只声明以下四个合法组合，A3 共生成 **4 个 tiling key**：

| INPUT_LAYOUT | IS_VARIABLE | metadata |
| --- | --- | --- |
| BNSD | 0 | 两项均为空 |
| BSND | 0 | 两项均为空 |
| NTD | 1 | 两项均有效 |
| TND | 1 | 两项均有效 |

使用 `ASCENDC_TPL_ARGS_DECL` / `ASCENDC_TPL_SEL` 声明上述组合；Host 先校验 layout、metadata 和 rank 一致，再用 `GET_TPL_TILING_KEY(...)` 生成 key。不声明 BNSD/BSND 的 packed 分支或 NTD/TND 的 dense 分支；固定 BF16 输入、C=64、K=V=128 和动态 B/T/NV 不增加 key。Kernel 按合法组合实例化搬运路径；不得通过 shape 猜测布局，也不得新增手写 bit 编码或连续数字 tiling key。

stateVFirst 不纳入 tiling key，也不作为上述 tiling key 宏的模板参数。Host 将其写入 tiling data；每个 AIC 在进入 work/head 主循环前读取一次，选择该 layout 下的 Cube<true> 或 Cube<false> 模板实例。两种实例均需编译，stateVFirst 只改变 h 的最后两个维度的数学视图，不在每个 head 或 GEMM 内重复判断。每个 layout key 均支持 stateVFirst=false/true；AIV 不读取或处理该属性。

后续 A5 使用同一组四种 layout/IS_VARIABLE 合法组合和模板宏分发；A5 的 arch35 资源、hand-off 和同步可以不同，但不得另设接口或 key 分支。SoC 根据平台信息和编译目标选择，不作为四个 layout key 的额外乘数。


## 第一部分：算子的 Stage

### 1.1 A5 标杆事实与迁移边界

| 项目                      | KernelA 原设计文档                                      | 当前 A5 源码                                            | 本 A3 方案                             |
| ------------------------- | ------------------------------------------------------- | ------------------------------------------------------- | -------------------------------------- |
| AIV 分工                  | generation 偶数整 head 给 AIV0，奇数整 head 给 AIV1     | 相同                                                    | 保持                                   |
| 4-head 工作项             | 改变多核任务粒度                                        | 相同                                                    | 保持，不改成半片任务                   |
| L1 owner                  | 两份，每份 72 KiB，共 144 KiB                           | 四份，每份 72 KiB，共 288 KiB                           | 以源码为准，四份                       |
| **A / dAqk_raw handoff** | FP32 L0C→BF16 UB                                       | FP32 L0C→FP32 UB                                       | FP32 L0C→FP32 GM→FP32 UB             |
| **Q / dq_raw**      | FixPipe 直接写 FP32 GM                                  | FixPipe→FP32 UB→AIV MTE3→GM                          | 按原文档，FixPipe 直接写 FP32 GM       |
| **D / dv**          | FixPipe 直接写 BF16 GM                                  | FixPipe→BF16 UB→AIV MTE3→GM                          | 按原文档，FixPipe 直接写 BF16 GM       |
| **UB / AIV**        | BF16 dAqk_raw 双槽 + FP32 dAqk 双槽 + mask，共 64 KiB      | FP32 dAqk_raw/dAqk 双槽 + mask + Q/D，共 128 KiB           | FP32 dAqk_raw/dAqk 双槽 + mask，共 80 KiB |
| h 索引                    | dense`[B,HV,NT,K,V]`，packed `[HV,totalChunks,K,V]` | dense`[B,NT,HV,K,V]`，packed `[totalChunks,HV,K,V]` | 使用源码的 chunk-major 索引      |

state_v_first=true 时交换 h 的最后两个语义维度。标杆存在两种 dAqk_raw dtype，本文明确选择源码的 FP32 路径，不增加 BF16 handoff 舍入；原文档的 BF16 handoff 可作为后续独立精度/带宽候选，不混入本方案；dAqk 指 AIV 完成下三角 mask 和 scale 后的结果。

源码证据：

- [Vector Process](../op_kernel/arch35/chunk_kda_bwd_prepare_vector.h)：`aivIdx=generation&1`，非目标 AIV `continue`；目标 AIV 按完整 `chunk.validRows` 计算并写回。
- [Cube Process](../op_kernel/arch35/chunk_kda_bwd_prepare_cube.h)：`aivIdx=headStep&1`、`aivSlot=(headStep>>1)&1`；`L1_HEAD_SLOT_COUNT=4`，每 owner 72 KiB；A/Q 共享 dO 的 L0A tile。
- [Common](../op_kernel/arch35/chunk_kda_bwd_prepare_common.h)：dAqk_raw FP32、A UB 80 KiB、Q 32 KiB、D 16 KiB，以及 chunk-major `StateOffset`。

A3 需要适配的内容是 L0C→GM 输出路径、AIV 的 GM→UB 搬入、GM workspace、跨核同步和 arch22 支持的 Vector/Cube API。源码中的 RegBase/MicroAPI、mode 0x4/+16、编译架构判断和直接 UB 投递不能仅改 SoC 名称后沿用。

### 1.2 符号、分核与整 head 映射

单位采用 KiB，`1 KiB=1024 B`；公式省略 batch、chunk、head 索引。

| 符号       | 含义                                                      |
| ---------- | --------------------------------------------------------- |
| C          | chunk 长度，64                                            |
| K,V        | 特征维度，均为 128                                        |
| m          | 当前 chunk 有效行数，1..64                                |
| HV         | value head 数，即原文档的 NV                              |
| CG         | 一个工作项最多四个连续 head，固定为 4                     |
| Nbase      | dense：`B*ceil(T/64)`；varlen：canonical totalChunks    |
| NW         | `ceil(HV/4)`                                            |
| Nwork      | `Nbase*NW`                                              |
| P          | `min(平台可用AIC数,Nwork)`，实际启动的逻辑 AIC block 数 |
| r          | 工作项内有效 head 编号，0..activeHeads-1                  |
| generation | 本 block 按实际执行顺序递增的 head 代次，跨工作项连续     |
| aiv        | `generation&1`，此 head 唯一的数据处理 AIV              |
| slot       | `(generation>>1)&1`，此目标 AIV 自己的 ping/pong        |
| record     | `2*aiv+slot`，GM 中转的四个循环 record 之一             |

沿用当前源码 head-window-major 解码：

```text
for work = logicalBlock; work < Nwork; work += P:
    window = work / Nbase
    chunkTask = work % Nbase
    headBegin = window * 4
    activeHeads = min(4, HV-headBegin)
    for r = 0 .. activeHeads-1:
        hv = headBegin+r
        PipelineHead(chunkTask,hv,generation)
        generation++
```

Dense：`batch=chunkTask/NT`，`chunkIdx=chunkTask%NT`，`tokenStart=chunkIdx*64`，`m=min(64,T-tokenStart)`。Varlen：canonical pair 给出 `(seq,localChunk)`，`tokenStart=cu[seq]+localChunk*64`，`m=min(64,cu[seq+1]-tokenStart)`，stateIdx=chunkTask。输入按 hv 寻址，不引入新的 GVA 语义。

```text
generation 0 -> AIV0.slot0，完整 head
           1 -> AIV1.slot0，完整 head
           2 -> AIV0.slot1，完整 head
           3 -> AIV1.slot1，完整 head
           4 -> AIV0.slot0，完整 head
           5 -> AIV1.slot0，完整 head
```

generation 不是 hv，也不是 r：跨工作项必须连续；HV=1/2/3/5 时，不能每个工作项把 generation 归零。每个 head 的有效 m 行全部由一个 AIV 处理；两路 AIV 按 head 分工，没有行半片，也没有 8 KiB half mask。

空序列不生成 chunk 工作项；不存在对应的 generation 或同步。所有有效 head 都计算，包括短于 32 行的尾块；非目标 AIV 在 A3 仅执行同步确认，不搬运、计算或写回该 head。

分配示例：B=1、HV=96、T=8192 时，Nbase=128、NW=24、Nwork=3072。若实际平台提供24个可用AIC，则P=24，每block执行128个工作项、512个head。核数只是代入示例；一般每核执行floor或ceil(Nwork/P)个工作项，波数ceil(Nwork/P)，末波有效工作项数`((Nwork-1)%P)+1`。本迁移保持4-head标杆，不凭静态容量宣称它已经是A3最快配置。

### 1.3 Stage 总表

| Stage                           | 执行单元    | 公式 / 功能                   | 输出路径                       | 主要空间                             |
| ------------------------------- | ----------- | ----------------------------- | ------------------------------ | ------------------------------------ |
| Stage 0：Cube，生成三项部分梯度 | AIC         | A：dAqk_raw；Q：dq_raw；D：dv | A→GM scratch，Q/D→公开输出GM | L1 288 KiB；L0A/B各64 KiB；L0C64 KiB |
| Stage 1：Vector，mask和scale    | 唯一目标AIV | 对当前head完整有效m行处理     | GM dAqk_raw→UB→FP32 dAqkOut       | 每AIV UB80 KiB                       |

首个迁移实现保持同一条1 AIC : 2 AIV MIX kernel流水。两个Stage表示计算类型和依赖关系，不强制拆成两次launch。GM中转不增加额外计算Stage。

### 1.4 Stage 0：Cube，A→Q→D

**A：注意力系数的原始梯度。**

```text
dAqk_raw = d_o @ v_new^T
[m,128] @ [128,m] -> [m,m]
BF16输入，FP32累加
```

v_new物理RowMajor[m,128]，数学B使用ColumnMajor转置视图[128,m]；不实体转置输入。A3路径：

```text
FP32 L0C -> FixPipe NZ→ND -> GM dAqk_raw[block][aiv][slot][64][64]
          -> READY[record]
          -> 唯一目标AIV MTE2 -> FP32 UB dAqk_raw[slot]
```

每份GM record固定64列stride、16 KiB容量。完整块写64×64；尾块只定义前m行、前m列。消费者只搬入有效m×m区域到UB固定64列stride；其余列不视为已初始化，按1.5清零后再参与Vector乘法。

**Q：query的状态部分梯度。**

```text
dq_raw = d_o @ H_right
[m,128] @ [128,128] -> [m,128]
state_v_first=false: H_right=h^T，h物理RowMajor[K,V]
state_v_first=true:  H_right=h，  h物理RowMajor[V,K]
BF16输入，FP32累加，FixPipe直接写FP32 dqRawOut
```

AIC在主循环前读取tiling data中的stateVFirst一次，选择对应的Cube模板和B-layout；两种state_v_first共享同一个layout key，不在GM转置h。Q不再经UB绕行，不申请Q scratch。

**D：value的初始部分梯度。**

```text
dv = Aqk^T @ d_o
[m,m] @ [m,128] -> [m,128]
BF16输入，FP32累加，FixPipe round-to-nearest转BF16直接写dvOut
```

使用前向Aqk，不额外乘scale。Aqk物理RowMajor，数学A采用转置视图。A3支持FP32 L0C写FP32/BF16 GM；ND stride、valid M/N和CAST_RINT参数须按目标CANN实现。[Fixpipe L0C→GM](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/cube_compute_ISASI/cube_compute_store/Fixpipe_L0CToGM.md)

**尾块的数学尺寸与硬件尺寸。** A3 在 MMAD 的 M=1 时可能选择 GEMV 并按 ND 读取 L0A；不能将旧平台的分形 L0A 与 M=m 原样迁移。首个 A3 实现统一采用 `M_hw=Align16(m)`，A/Q/D 的 MMAD 参数分别为 `(M_hw,m,128)`、`(M_hw,128,128)`、`(M_hw,128,m)`，保持同一分形 GEMM 路径；因此本方案传给 MMAD 的 M 不应在 `m=1` 时仍为 1，但必须在目标 SDK 上确认封装没有按有效行数重新选择 GEMV。只从 GM 读取实际 m 行及 Aqk 前 m 列；L1→L0 搬运限制在已预留 tile 的分形边界内，额外结果行不写回。D 的归约长度仍为 m，不能直接改成 `M_hw`；若具体封装扩大 K，则必须先将两操作数的补齐归约区置零。BF16 非对齐 N/K 的无效分形元素按 MMAD 契约忽略。FixPipe 只写 A 的 m×m 及 Q/D 的 m×128，源 stride 按实际 L0C 分形 M pitch 填写，GM 目的 stride 仍为 64/128。该规则不增加 key 或超过完整 tile 容量，m=1 需单独验收。[Mmad：GEMV与非对齐维度](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0249.html)

三公式在每head按A→Q→D发射，数学上互不依赖。A FixPipe完成立即发布READY，目标AIV的mask/scale/MTE3可以与当前head的Q/D和后续head重叠。

保持源码的L1四owner预取：按相邻head对搬入A batch=dO+vNew、Q batch=h、D batch=Aqk，各batch分别发布MTE2_MTE1 ready；A batch到达即可进入A的MTE1，不等待其它batch。对于同一pair，leader消费A/Q/D的ready，follower由同一MTE1指令流的顺序保证到达性。

保持源码A/Q的L0A复用：A搬入dO后不释放该L0A槽；Q直接复用dO，仅加载h到新的L0B槽；Q的MMAD完成读取后才释放共享L0A。D加载自己的Aqk^T/dO。L1 owner在D最后一次MTE1完成后释放；L0A/B等MMAD读完，L0C等FixPipe读完，三者生命周期独立。

### 1.5 Stage 1：Vector，完整 head 的mask/scale

每个AIV启动时生成一次完整FP32[64,64] triScaleMask，16 KiB，驻留自身UB整个block生命周期：

```text
triScaleMask[row,col] = scale, col<=row
                            0, col>row
```

两路AIV各有一份完整mask，没有共享UB，也不是各存半份。mask与batch/chunk/head无关，只由固定C和本次scale决定；不写GM、不占workspace、不在各head之间重建。

目标AIV消费一个完整head：

```text
MTE2: GM dAqk_raw[:m,:m] -> UB dAqk_raw[:m,:m]，UB行stride固定64
safe[row,col] = dAqk_raw[row,col], col<m && col<=row
                           0, 其它位置
PIPE_V barrier                  // safe的复制/清零完成后才能原位乘
dAqk = safe * triScaleMask        // FP32逐元素乘法
MTE3: dAqk[:m,0:64] -> dAqkOut    // 只写真实token行
```

safe直接使用已分配的dAqk槽：复制dAqk_raw有效区域，显式清零尾列与上三角，再在dAqk槽中原位乘mask，不分配额外完整safe矩阵。A3采用 arch22 支持的 Vector API，不直接粘贴旧平台的 RegBase/__simd_vf__ 接口。

GM→UB使用非对齐DataCopyPad：blockCount=m，blockLen=m*4 B，srcStride=(64-m)*4 B，dstStride=(64-Align8(m))/8 个32 B数据块；isPad=true，leftPadding=0，rightPadding=Align8(m)-m 个FP32元素，paddingValue=0。这样两端行起点间距均为256 B，只读取GM有效列。rightPadding仅负责当前32 B块补齐，不能设成64-m去清整行；dstStride跳过区没有零值保证。Vector只向dAqk复制有效列，再显式清零其上三角和尾列，并以PipeBarrier<PIPE_V>()保护清零/复制→原位Mul的依赖。完整块无需非对齐补齐。[DataCopyPad GM→UB](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/memory_vector_compute/data_move/DataCopyPad_GMToUB.md)

本次A3实现对完整块和尾块统一先清上三角及尾列，再乘mask，避免无效位置的NaN/Inf经乘零传播。row>=m的行不读GM也不写回，不需要清空整槽。

A dAqk_raw GM record在目标AIV的MTE2真实完成后即可归还AIC；dAqk_raw UB等Vector最后读取后归还；dAqk UB等MTE3完成后归还。这三个条件不能合并成一个FREE。

公开输出仍为dAqk FP32、dq_raw FP32、dv BF16。A/Q保持FP32累加与写出；只有D最终舍入为BF16。A3 Host检查scale及其FP32转换仍有限。有限GEMM结果是数值分析与标杆值域的前提，Host不扫描tensor值或保证任意BF16输入都不溢出；若要求最终输出有限，还须约束dAqk_raw×scale的范围。尾列与上三角精确零单独验收。

### 1.6 数据搬运汇总

完整chunk、单head的逻辑GM搬运量，不含元数据/cache line放大：

| 数据     | 方向 / 次数         |     大小 | 复用与最后消费者        |
| -------- | ------------------- | -------: | ----------------------- |
| dO       | GM→L1一次          |   16 KiB | L1供A/Q/D；L0A供A/Q复用 |
| vNew     | GM→L1一次          |   16 KiB | A的MTE1                 |
| h        | GM→L1一次          |   32 KiB | Q的MTE1                 |
| Aqk      | GM→L1一次          |    8 KiB | D的MTE1                 |
| dAqk_raw | L0C→GM一次         |   16 KiB | 一个目标AIV的MTE2       |
| dAqk_raw | GM→UB一次          |   16 KiB | 一个目标AIV的Vector     |
| dAqk     | UB→GM一次          |   16 KiB | 一个目标AIV写完整head   |
| dq_raw   | L0C→GM一次         |   32 KiB | 后续Kernel              |
| dv       | L0C→GM一次         |   16 KiB | 后续Kernel              |
| mask     | 各AIV本地生成并复用 | 0 GM字节 | block结束前常驻         |

合计GM读88 KiB、写80 KiB、总168 KiB/head，四head为672 KiB。相比FP32 dAqk_raw片上直传，每head增加16 KiB写+16 KiB读；并非两个AIV各读一遍dAqk_raw。实际HBM流量还取决于L2命中。

原始输入和公开输出的cache hint按当前源码在目标A3上的支持情况移植；dAqk_raw workspace采用受同步保护的GM搬运，不通过scalar缓存轮询flag。L2 hint不替代跨核可见性协议，性能变化须单独profiling。

## 第二部分：A3片上内存、workspace与double buffer

### 2.1 硬件容量与规划口径

| 资源 |  A3预算 | 作用范围            | 本方案使用 |    剩余 |
| ---- | ------: | ------------------- | ---------: | ------: |
| L1   | 512 KiB | 每AIC               |    288 KiB | 224 KiB |
| L0A  |  64 KiB | 每AIC               |     64 KiB |       0 |
| L0B  |  64 KiB | 每AIC               |     64 KiB |       0 |
| L0C  | 128 KiB | 每AIC               |     64 KiB |  64 KiB |
| UB   | 192 KiB | 每AIV、独立地址空间 |     80 KiB 自有 payload | 112 KiB payload余量 |

UB预算采用用户提供的192 KiB。CANN 的 GM→L1 ND2NZ 文档对 Atlas inference 产品要求为 API 临时数据预留 8 KiB UB；该空间不是本算子的 payload，也不是 transpose workspace，归属 AIC、AIV 还是独立硬件区要按目标 SDK 确认。若它与 AIV UB 共享预算，资源验收按 80+8=88 KiB 峰值和 104 KiB 余量执行；若它由 AIC 或独立区承担，则在对应资源表中单列，不能把 112 KiB 或其它资源余量重复计算。[GM→L1 ND2NZ 临时 UB 约束](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/cube_compute_ISASI/cube_compute_load/DataCopy_GMToL1_ND2NZ.md) L1/L0表为910_93规划预算，开发时以目标平台GetCoreMemSize查询结果及资源封装可用上限逐项确认，不继承A5的L0C256 KiB或UB248 KiB。[GetCoreMemSize](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1/API/ascendcopapi/atlasascendc_api_07_1042.html)

下表地址是各资源基址起的绝对半开字节区间，数值以KiB表示。固定区起点至少512 B对齐，满足32 B数据块和L0C源64 B对齐；具体布局和尾块参数继续服从A3 API约束。剩余是payload规划余量，Resource/TPipe额外空间需要显式扣除，不得覆盖固定区，也不得被多个管理器重复分配。

### 2.2 L1：四份head owner，288 KiB

```text
一份owner = dO16 + vNew16 + h32 + Aqk8 = 72 KiB
当前源码四owner = 4*72 = 288 KiB
```

| L1区间 / KiB | 对象         | dtype / 完整shape |    大小 | 写入与最后消费          |
| ------------ | ------------ | ----------------- | ------: | ----------------------- |
| [0,72)       | owner0       | BF16，子区见下表  |  72 KiB | MTE2写；D末次MTE1后释放 |
| [72,144)     | owner1       | 同上              |  72 KiB | 同上                    |
| [144,216)    | owner2       | 同上              |  72 KiB | 同上                    |
| [216,288)    | owner3       | 同上              |  72 KiB | 同上                    |
| [288,512)    | 空闲，无数据 | 未分配            | 224 KiB | 其它分配须独立计数      |

owner r内部的固定绝对边界：

| 地址 / KiB      | tensor | 物理视图                            |   大小 | 最后消费者 |
| --------------- | ------ | ----------------------------------- | -----: | ---------- |
| [72r,72r+16)    | dO     | BF16 NZ，数学[64,128]              | 16 KiB | D的MTE1    |
| [72r+16,72r+32) | vNew   | BF16 NZ，数学[64,128]，转置view    | 16 KiB | A的MTE1    |
| [72r+32,72r+64) | h      | BF16 NZ，数学[K,V]或[V,K]          | 32 KiB | Q的MTE1    |
| [72r+64,72r+72) | Aqk    | BF16 NZ，数学[64,64]，转置view     |  8 KiB | D的MTE1    |

GM输入是RowMajor，arch22按3.3随GM→L1搬运组织NZ，LoadData再排列为A3所需的L0A/L0B分形布局；数学转置不产生整张量副本。h的完整128行NZ tile使用dstNzC0Stride=128，token tile使用64；不能把相同stride用于两者。完整维度均为16倍数，tile容量不变。尾块搬运与MMAD参数按1.4，只写回有效结果。

四owner不是“四head工作项再双缓冲一整份”：两组完整四head需要576 KiB，超过512 KiB。四owner各自按最后MTE1完成的credit循环复用，跨工作项写同一owner前等待credit。

原文档的两owner ping/pong仅用144 KiB，同样满足A3容量；它与当前源码四owner属于两种实现基线。本文迁移当前源码故保留四份，不能把旧文档的“并不驻留四份”当作当前代码事实。没有A3证据时不主动回退或扩展驻留份数。

### 2.3 L0A/L0B：各两槽，每槽32 KiB

| 公式 | 完整(M,N,K_reduce) | L0A               | 有效大小 | L0B              | 有效大小 |
| ---- | ------------------ | ----------------- | -------: | ---------------- | -------: |
| A    | (64,64,128)        | dO[64,128]        |   16 KiB | vNew^T[128,64]   |   16 KiB |
| Q    | (64,128,128)       | 复用A的dO[64,128] |   16 KiB | H_right[128,128] |   32 KiB |
| D    | (64,128,64)        | Aqk^T[64,64]      |    8 KiB | dO[64,128]       |   16 KiB |

| 资源 | 区间 / KiB | 对象  | 复用条件                               |
| ---- | ---------- | ----- | -------------------------------------- |
| L0A  | [0,32)     | slot0 | MMAD末次读取完成；A的dO必须等Q读取完成 |
| L0A  | [32,64)    | slot1 | 同上                                   |
| L0B  | [0,32)     | slot0 | 对应MMAD读取完成                       |
| L0B  | [32,64)    | slot1 | 同上                                   |

A/Q/D不分配专属物理区。每公式递增formulaGeneration；L0B与L0C按其奇偶切换。A的aSlot保存dO，Q的L0B使用qSlot，但L0A仍使用aSlot；D使用自己的dSlot。L0A所有权由事件决定，不能在A结束就因为公式编号变化而释放。物理tile使用Cube对应分形排列，表中shape为数学视图。

未分块合并A/Q的B tile为128×192×2=48 KiB，大于单槽32 KiB，不采用该变化。槽内部未使用尾部不借给并发数据。

### 2.4 L0C：FP32结果两槽，共64 KiB

```text
A = 64*64*4 = 16 KiB
Q = 64*128*4 = 32 KiB
D = 64*128*4 = 32 KiB
单槽 = max(A,Q,D) = 32 KiB，双槽64 KiB
```

| L0C区间 / KiB | 对象              |   大小 | 首次写 / 最后读    |
| ------------- | ----------------- | -----: | ------------------ |
| [0,32)        | FP32 result slot0 | 32 KiB | MMAD / FixPipe     |
| [32,64)       | FP32 result slot1 | 32 KiB | MMAD / FixPipe     |
| [64,128)      | 空闲，无数据      | 64 KiB | 额外分配须独立评审 |

A只使用当前槽前16 KiB；Q/D使用32 KiB。D在L0C仍为FP32，不能按GM BF16的16 KiB来分配。尾块按硬件tile粒度对齐，最大不超过完整块。

初始化FIX_M两个credit。每槽Wait FIX_M→MMAD→Set/Wait M_FIX→FixPipe到唯一GM目的地→Set FIX_M。FixPipe真实读完才释放；结尾drain两槽。三结果不同时驻留L0C。

### 2.5 UB：每AIV整head dAqk_raw/dAqk 双槽，共80 KiB

```text
dAqk_raw FP32[64,64] = 16 KiB/槽，双槽32 KiB
dAqk FP32[64,64] = 16 KiB/槽，双槽32 KiB
mask FP32[64,64] = 16 KiB，常驻一份
UB / AIV = 32+32+16 = 80 KiB
payload余量 = 192-80 = 112 KiB（未扣目标 SDK 可能要求的 API 临时区）
```

| 每AIV UB区间 / KiB | 对象                 | dtype/layout         |    大小 | 生命周期                          |
| ------------------ | -------------------- | -------------------- | ------: | --------------------------------- |
| [0,16)             | dAqk_raw ping，slot0      | FP32 RowMajor[64,64] |  16 KiB | MTE2写；Vector读完才复用          |
| [16,32)            | dAqk_raw pong，slot1      | 同上                 |  16 KiB | 同上                              |
| [32,48)            | dAqk ping，slot0   | FP32 RowMajor[64,64] |  16 KiB | Vector写；MTE3读完才复用          |
| [48,64)            | dAqk pong，slot1   | 同上                 |  16 KiB | 同上                              |
| [64,80)            | triScaleMask，整矩阵 | FP32 RowMajor[64,64] |  16 KiB | 启动期生成；block所有目标head复用 |
| [80,192)           | 空闲，无数据         | 未分配               | 112 KiB | 辅助缓冲需显式计数                |

上表的 `[80,192)` 只是本算子自有 payload 的规划余量。若目标 SDK 为 ND2NZ 隐式占用 8 KiB UB，且该区与本 AIV UB 共享预算，必须从这段余量中扣除；若该区属于 AIC 或独立硬件区，则在对应资源中单列。该临时区由 API 生命周期管理，不能被 dAqk_raw、mask 或其它辅助缓冲复用。

```text
dAqkRawOffset(slot) = slot*16384
dAqkOffset(slot) = 32768+slot*16384
maskOffset = 65536
```

两个AIV各有独立的一套80 KiB；它们分别处理自己的head，不能合并成共享UB。4-head工作项不会使每个AIV驻留四head×双槽；按照A5标杆，每个AIV只有两个循环slot，跨工作项按generation复用。NV=1时两AIV轮流处理不同工作项的head；没有固定将所有hv=0交给AIV0。

dAqk_raw与dAqk的UB槽物理分离：一个head的MTE3排空dAqk槽时，AIC可通过GM继续生产，AIV可在dAqk_raw槽的credit允许时搬入下一代。AIC不再向UB写数据，dAqk_raw本地V_MTE2保护的是AIV自身MTE2覆盖，GM_FREE保护的是AIC对GM覆盖。

Q/D按原文档直接写GM，因此去掉当前A5源码Q32 KiB和D16 KiB的UB；得到80 KiB而不是128 KiB。不存在BF16→FP32 Cast，也不另分配safe矩阵。尾块只使用前m行，输出每有效行保留64列，尾列精确零。

| 基线 / 方案 | UB / AIV | 192 KiB余量 | 说明                                        |
| ----------- | -------: | ----------: | ------------------------------------------- |
| 原A5文档    |   64 KiB |     128 KiB | BF16 dAqk_raw 双槽、FP32 dAqk 双槽、完整mask     |
| 当前A5源码  |  128 KiB |      64 KiB | FP32 dAqk_raw/dAqk/mask80 + Q32 + D16          |
| 本A3迁移    |   80 KiB |     112 KiB* | 保留整head FP32 dAqk_raw/dAqk 双槽，Q/D直接写GM |

`*` 仅表示本算子自有 payload 余量；若目标 SDK 需要 8 KiB ND2NZ 临时 UB，且与 AIV UB 共享预算，最终可分配余量为 104 KiB；若由 AIC 或独立硬件区承担，则按对应资源表扣除。

所以UB容量不是迁移的阻塞因素；迁移所需的新增空间主要是 GM dAqk_raw 中转，而不是另加一份UB中转矩阵。

### 2.6 GM user workspace：每AIC block四record，共64 KiB

```text
recordBytes = 64*64*4 = 16384 B
recordsPerBlock = 2 AIV * 2 slot = 4
blockScratchBytes = 65536 B
userWorkspaceBytes = Align512(P*65536)
```

GM索引：

```text
record = 2*aiv+slot
recordOffset = (logicalBlock*4+record)*16384
dAqk_raw[row,col] = userBase + recordOffset + (row*64+col)*4
```

| 每block GM相对区间 / KiB | 对象           |   大小 |
| ------------------------ | -------------- | -----: |
| [0,16)                   | AIV0 dAqk_raw slot0 | 16 KiB |
| [16,32)                  | AIV0 dAqk_raw slot1 | 16 KiB |
| [32,48)                  | AIV1 dAqk_raw slot0 | 16 KiB |
| [48,64)                  | AIV1 dAqk_raw slot1 | 16 KiB |

每record由AIC FixPipe产生，唯一目标AIV MTE2读取；另一路只同步确认，不读dAqk_raw。跨grid-stride工作项循环复用，不按所有chunk/head分配。P=24时为1.5 MiB，P=28时为1.75 MiB，仅为公式示例。

userBase采用框架规定的user-workspace获取方式，与system workspace分开；tiling返回框架system workspace加上述user scratch，不能把64 KiB/block当作ACLNN总workspace。公开GetWorkspaceSize还包含executor中元数据转换等所需空间，以实际返回值为准。Kernel不能继续忽略workspace参数。GM区不计入UB/L1/L0。record在真实MTE2消费结束后才能覆盖，不因READY已发出或copy已提交就释放。

有限64 KiB/block依赖同一MIX kernel的消费闭环。若改为两个独立launch，Stage0结束前Stage1尚未消费，必须为全部(chunk,hv)保留dAqk_raw，保守上界Align512(Nbase*HV*16 KiB)，不能套用此循环scratch大小。父backward为三个公开输出保留的存储另算，不重复计入dAqk_raw scratch。

### 2.7 双缓冲、跨核flag与本地事件

#### 独立的资源编号

| 资源            | 编号                      | 释放条件                        |
| --------------- | ------------------------- | ------------------------------- |
| L1              | 工作项内owner0..3         | 此head最后MTE1完成              |
| L0A             | aSlot/dSlot；Q复用aSlot   | 对应最后MMAD完成读取            |
| L0B/L0C         | formulaGeneration奇偶两槽 | 分别MMAD / FixPipe读取完成      |
| GM dAqk_raw          | aiv×slot，四record       | 目标AIV的MTE2完成且同步确认齐全 |
| 各AIV UB dAqk_raw    | 自己的slot0/1             | Vector最后读取完成              |
| 各AIV UB dAqk | 自己的slot0/1             | MTE3最后读取完成                |

GM/UB使用相同aiv×slot便于寻址，credit相互独立。generation跨work连续；不能用r奇偶重置这些生命周期。

#### A3跨核握手：两路确认，但只有一路处理数据

旧平台的mode0x4支持按AIV独立握手和AIV1的+16 stride；A3采用官方mode0x2的AIC↔两AIV协议，不复制0x4/+16。[CrossCoreSetFlag](https://www.hiascend.com/doc_center/source/en/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0273.html)

为保持整head分工，在A3使用广播READY和两路确认：两个AIV都按同一generation消费READY，只有被选中的AIV执行MTE2/Vector/MTE3；非目标AIV直接确认该record。AIC等两路确认后复用GM。因此“参与握手”不表示两个AIV共同计算同一head。

| flag role       | 数量 | 候选ID | 初始状态          | 生产/消费                                        |
| --------------- | ---: | ------ | ----------------- | ------------------------------------------------ |
| READY[record]   |    4 | 4..7   | 无token           | AIC在A FixPipe完成后发布；两个AIV都wait          |
| GM_FREE[record] |    4 | 0..3   | 两个AIV各发布一次 | 目标AIV在MTE2完成后发布，非目标确认；AIC聚合wait |

共8个逻辑ID，具体可用编号范围和保留编号按目标 A3 SDK 核对；本方案只使用 READY[4..7] 与 GM_FREE[0..3]，不假定其它编号可用。使用基础 Cube 路径，不与占用同 flag 的 Matmul 高阶 API 混用。具体等待 API 形式和 MIX block 归一化按目标 SDK 核对。READY 关联 PIPE_FIX 的完成；目标 GM_FREE 关联 PIPE_MTE2 的完成；非目标确认也使用明确排序的 pipe 同步，不通过 scalar 提前绕过 READY。

CrossCoreWaitFlag在A3会阻塞该核等待点之后的所有流水指令，不能把模板中的PIPE_FIX/PIPE_MTE2理解为A5式的单pipe阻塞。Set仍关联真实生产pipe；Wait显式填写目标SDK支持的参数，不依赖PIPE_S默认值。AIC只在相应record即将写出前等待FREE，AIV在其MTE2读取前等待READY；已发射工作可继续推进，但等待期间不能保证后续Q/D或下一head继续发射。双缓冲只提供有限解耦，实际重叠程度由profiling证明。[CrossCoreWaitFlag](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/sync_control/inter_core_sync/CrossCoreWaitFlag_ISASI.md)

这是硬件协议适配，计算映射保持整head分工。不能把旧平台的非目标`continue`放在READY wait之前，否则mode2缺少一个确认，会死锁；也不能让非目标AIV为了确认而再次读取或计算整份dAqk_raw。每record仅在前代FREE收齐后发布下一代READY，计数不无限堆积。

#### 本地HardEvent资源表

事件按方向独立管理；A3硬件ID范围0..7不等于可自行使用8个ID。静态规划仅使用0..5，6/7不手工占用；采用TPipe时通过AllocEventID/FetchEventID协调分配，并计入库内部占用，drain后及时ReleaseEventID。不未经核对粘贴A5固定编号。[SetFlag/WaitFlag](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/sync_control/intra_core_sync/SetFlag_WaitFlag_ISASI.md)

| 核    | 方向      | 数量 / 方向 | 初态 / 生命周期                                  |
| ----- | --------- | ----------: | ------------------------------------------------ |
| AIC   | MTE1_MTE2 |           4 | 四owner可写；D末次MTE1归还                       |
| AIC   | MTE2_MTE1 |           6 | 两head pair×A/Q/D三batch；pair leader按公式wait |
| AIC   | M_MTE1    |           4 | L0A/B各两槽可写；A共享L0A等Q完成后才归还         |
| AIC   | MTE1_M    |           2 | 无token；每公式操作数到达set/wait                |
| AIC   | FIX_M     |           2 | 两L0C槽可写；FixPipe完成归还                     |
| AIC   | M_FIX     |           2 | 无token；MMAD完成set，FixPipe前wait              |
| 每AIV | V_MTE2    |           2 | dAqk_raw 双槽可写；Vector读完归还                      |
| 每AIV | MTE2_V    |           2 | 无token；copy完成set，Vector前wait               |
| 每AIV | MTE3_V    |           2 | dAqk 双槽可写；MTE3写完归还                     |
| 每AIV | V_MTE3    |           2 | 无token；Vector完成set，MTE3前wait               |

算子每方向最多6个ID，MTE2_MTE1恰好占满静态0..5；该方向不得再叠加库内部事件而声称仍有两个可用ID。必须核算Resource/TPipe内部事件的方向、占用和生命周期并检查分配失败；资源封装若冲突，需复用可证明不重叠的生命周期或移除其额外同步占用。partial pair只发布真实存在的pair A/Q/D事件，无效pair不发也不等。

#### 统一同步伪代码

以下是同步语义，不是可粘贴编译的API代码。AIC和两个AIV使用同一有效work/head列表，并共同递增generation。

```text
Init AIC:
    初始化L1四owner、L0A/B双槽、L0C双槽credit

Init each AIV:
    初始化RAW_UB_FREE[0/1]和OUT_UB_FREE[0/1]
    在UB[64,80)生成完整mask，PIPE_V barrier
    mode2 Set GM_FREE[record=0..3]  // 两个AIV均发布初始确认

AIC for work:
    按源码预取相邻head pair的A/Q/D batch到L1四owner
    每pair等待实际owner可写并按batch发布ready
    for head in actual heads, generation++:
        aiv=generation&1; slot=(generation>>1)&1; record=2*aiv+slot
        按pair leader规则等待A batch，加载dO/vNew并计算A
        Wait GM_FREE[record] on PIPE_FIX
        FixPipe A FP32 L0C -> GM dAqk_raw[block,record]
        Set READY[record] on PIPE_FIX
        按Q batch ready加载h，复用A的L0A dO计算Q
        FixPipe Q FP32 -> dqRawOut，Q读完后归还共享L0A
        按D batch ready加载Aqk/dO，D末次MTE1归还此L1 owner
        计算D；FixPipe FP32->BF16 dvOut
        每公式独立遵守L0B/L0C credit

Each AIV for same work/head list, generation++:
    target=generation&1; slot=(generation>>1)&1; record=2*target+slot
    Wait READY[record] on PIPE_MTE2   // 两个AIV都消费；A3实际阻塞所有后续pipe
    if myAiv != target:
        Set GM_FREE[record] on PIPE_MTE2 // READY已消费后仅确认
        continue
    Wait RAW_UB_FREE[slot]            // 防止 MTE2 覆盖旧 dAqk_raw
    MTE2 GM有效m×m -> dAqkRawUB[slot]，固定64列stride，只补齐至32 B块
    Set MTE2_V[slot]
    Set GM_FREE[record] on PIPE_MTE2  // GM已读完，可交还AIC
    Wait MTE2_V[slot]
    Wait OUT_UB_FREE[slot]            // 防止Vector覆盖MTE3尚未搬完的dAqk
    Vector复制有效dAqk_raw -> dAqk，并清无效列/上三角
    PipeBarrier<PIPE_V>()            // dAqk写后读，不由MTE2_V代替
    FP32 Mul dAqk *= mask
    Set RAW_UB_FREE[slot] on PIPE_V
    Set/Wait V_MTE3[slot]
    MTE3 dAqk[:m,:64] -> dAqkOut整head有效行
    Set OUT_UB_FREE[slot] on PIPE_MTE3

Drain AIC:
    Wait四record最终GM_FREE；未使用record回收其初始确认
    drain全部L1/L0A/L0B/L0C credit，释放事件

Drain each AIV:
    Wait自己dAqk_raw/dAqk两个slot的最终credit，释放事件
```

READY广播和FREE两路确认是A3新增的协议工作；真正的Vector服务仍是AIV0处理generation0/2/...，AIV1处理1/3/...。不增加PIPE_ALL屏障，但FREE/READY跨核Wait本身在A3具有上述全流水阻塞语义。每AIV的dAqk_raw读取、dAqk写回各自闭环；Q/D没有独立的AIV缓冲credit，仍可能受到AIC等待A结果record复用的回压。

同一GM record每四个head generation回绕，同一AIV的dAqk_raw/dAqk slot每两次本AIV数据任务回绕。最后GM_FREE只说明GM消费结束，不说明最后MTE3结束；AIV 必须自行 drain dAqk。短head窗口、NV=1、尾chunk和连续work均遵守同一generation规则，不存在工作项级全局屏障。

### 2.8 容量复核与验证

| 资源                   | 计算                               | payload峰值 |         预算 | 最大连续空闲区     |
| ---------------------- | ---------------------------------- | ----------: | -----------: | ------------------ |
| L1                     | 4×72 KiB                          |     288 KiB |      512 KiB | [288,512)，224 KiB |
| L0A                    | 2×32 KiB                          |      64 KiB |       64 KiB | 无                 |
| L0B                    | 2×32 KiB                          |      64 KiB |       64 KiB | 无                 |
| L0C                    | 2×32 KiB                          |      64 KiB |      128 KiB | [64,128)，64 KiB   |
| UB / AIV 自有 payload  | 2×16 dAqk_raw + 2×16 dAqk + 16 mask |      80 KiB |      192 KiB | [80,192)，112 KiB（未扣 API 临时区） |
| ND2NZ API 临时 UB      | 目标 SDK 要求时单列预留             |       8 KiB | 按目标核确认 | 从对应片上资源余量扣除，不得复用 |
| GM scratch / AIC block | 2 AIV×2 slot×16 KiB              |      64 KiB | 设备全局内存 | 不计片上空间       |

首先验证A3/arch22编译、MIX类型和block索引；查询真实容量；确认A/Q FP32与D BF16的FixPipe GM写出、ND stride和尾块安全搬运；验证mode2两路确认及非目标AIV不搬运、不计算；确认事件含库内部占用未超限。

精度覆盖dense T=1/31/32/33/63/64/65/127/128、varlen多序列/空序列/尾块、HV=1/2/3/4/5/32/96、两种state_v_first（非对称h）、正/零/有限负scale以及至少100次重复launch。特别检查跨work generation不重置、GM record四代回绕、UB slot回绕、上三角/尾列精确零、真实token不越界和最终drain；资源测试还要确认 ND2NZ API 临时 UB 已按目标 SDK 计入并且不与本算子 payload 重叠。

Profiling验证 A dAqk_raw 为 L0C→GM→目标 AIV UB，Q/D 直接写 GM，每 AIV 完整 mask 只生成一次，head 按 generation 奇偶分流且只有一份 dAqk_raw GM 读取。观察 AIC Q/D 与 AIV Vector/MTE3 重叠、新增 32 KiB/head 逻辑 GM 流量和 FREE 回压；性能以 msprof 目标 kernel 的 Task Duration(us) 判断，不把未完成的实测作为 A3 已达标的证据。

## 第三部分：BSND/TND 输入支持：将布局整理融合进 GM→L1 搬运

本部分说明 BSND/TND token 输入的搬运实现。BSND/TND 通过输入 offset 和行 stride 直接搬运，不先生成完整的 BNSD/NTD 临时副本。Stage 公式、整 head AIV 分工、输出布局、state 布局以及片上空间继续采用前述设计。本部分描述实现方案，尚未修改代码或验证设备性能。

### 参考 forward prepare 的已核对实现

已读取用户指定的 `chunk_kda_fwd_prepare` 实现，核对公共地址函数、arch22/arch35 Vector 输入搬运、Host layout 解析和输出地址。它将 BSND/TND 处理融合进当前 head 的搬运，不先生成整张量 BNSD/NTD 副本；以下是应迁移的具体机制。

| 参考位置（相对 chunk_kda_fwd_prepare）                                                                               | 已有实现                                                     | KernelA 中的对应处理                                 |
| -------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ | ---------------------------------------------------- |
| [utils：QkInputOffset / ValueInputOffset](../../chunk_kda_fwd_prepare/op_kernel/chunk_kda_fwd_prepare_utils.h)        | inputSequenceMajor 时按`(b*T+token)*heads+head` 寻址       | TokenInputOffset 按 BSND/TND 公式寻址                    |
| [utils：QkHeadTensorOffset / HeadTensorOffset](../../chunk_kda_fwd_prepare/op_kernel/chunk_kda_fwd_prepare_utils.h)   | 输出始终按`(b*heads+head)*T+token` 寻址                    | TokenOutputOffset 独立于输入layout                   |
| [arch22 Vector：StageV0](../../chunk_kda_fwd_prepare/op_kernel/arch22/chunk_kda_fwd_prepare_vec.h)                    | q/k 的 DataCopyPad 按有效行数分块，srcStride 跳过其它head    | 参考按head抽行的原理；KernelA直接在Cube GM→L1中完成 |
| [arch22 Vector：StageV6](../../chunk_kda_fwd_prepare/op_kernel/arch22/chunk_kda_fwd_prepare_vec.h)                    | v按ValueInputOffset与vStride搬入，结果按HeadTensorOffset写回 | v_new/d_o输入跨步、Q/D输出固定head-major             |
| [arch35 Vector 输入搬运](../../chunk_kda_fwd_prepare/op_kernel/arch35/chunk_kda_fwd_prepare_vec.h)                    | 同样用DataCopyPad的源间隔参数读 BSND/TND                          | 不把A5 Vector搬运目的地或计算API直接套到A3 Cube      |
| [Host op_api：layout解析与DMA stride校验](../../chunk_kda_fwd_prepare/op_host/op_api/aclnn_chunk_kda_fwd_prepare.cpp) | 显式解析layout，校验跨head DMA间隔，不靠shape猜测            | 对新layout及GM→L1字段范围做对应校验                 |

参考算子的 q/k 搬运参数等价于：

```cpp
// 参考 forward prepare 的 GM→UB 路径；以下省略实际类成员名。
offset = ((uint64_t(b) * T + tokenStart) * heads + head) * D;
copy.blockCount = validRows;
copy.blockLen = D * sizeof(bfloat16_t);
copy.srcStride = (heads - 1) * D * sizeof(bfloat16_t);
copy.dstStride = 0;
DataCopyPad(ubTile, gmInput[offset], copy, pad);
```

`srcStride` 是前一块末尾到后一块开头的跳过字节数；所以实际源行起点间距为 `blockLen+srcStride=heads*D*2` 字节。KernelA 的 ND2NZ 路径用 `srcDValue=heads*D` 元素表达同一行起点间距，二者对应的是同一个 head 抽取动作，但参数单位和目的存储格式不同。

前向参考实现实际先用 Vector 处理 q/k/v/g/beta，所以跨步输入搬入 UB 合理；KernelA 的 aqk/v_new/d_o 首个消费者是 Cube，直接抽取到 L1 可以保留现有两Stage并避免新增UB暂存和一次UB→GM→L1中转。3.3 的 GM→L1 方案是针对KernelA消费者作的适配，不是宣称参考算子已用同一ND2NZ调用直接读取原始 BSND/TND 输入。

### 3.1 支持范围与接口表达

输入/输出 shape、layout 和 metadata 约束统一以[第零部分的 Torch 接口](#torch-接口)为准。四种 layout 均显式传入；BSND/TND 直接进入跨步搬运，BNSD/NTD 使用对应 head-major 输入。

Host 不能继续使用 SameShape(aqk,dAqkOut) 或 SameShape(vNew,dvOut) 表达完整契约；应先解析 layout 的语义 B/NV/T/D，再校验固定 head-major 输出 shape。三份 token 输入须为物理连续 tensor；非连续 view 返回参数错误。

### 3.2 分开计算输入与输出地址

令 b 为 batch，hv 为当前 head，t0 为当前 chunk 的全局 token 起点，D 为输入末维宽度。以下 offset 和 rowStride 均以元素为单位：

```text
BNSD 输入：
    inputOffset = ((b*NV + hv)*T + t0)*D
    inputRowStride = D

BSND 输入：
    inputOffset = ((b*T + t0)*NV + hv)*D
    inputRowStride = NV*D

NTD 输入：
    inputOffset = (hv*T + t0)*D
    inputRowStride = D

TND 输入：
    inputOffset = (t0*NV + hv)*D
    inputRowStride = NV*D

固定 dense BNSD 输出：
    outputOffset = ((b*NV + hv)*T + t0)*D_out
    outputRowStride = D_out

固定 packed NTD 输出：
    outputOffset = (hv*T + t0)*D_out
    outputRowStride = D_out
```

aqk 使用 D=64，v_new/d_o 使用 D=128；dAqk 的 D_out=64，dv/dq_raw 的 D_out=128。TND 的 t0 使用对应序列在 packed 总 token 中的全局起点，不能改成局部序列长度。

原 TokenOffset 同时服务输入和输出的用法必须拆成 TokenInputOffset、TokenInputRowStride、TokenOutputOffset。Q/D 的 FixPipe 目的地址和 AIV MTE3 的 dAqkOut 地址始终调用输出函数，不能误用 BSND/TND 输入地址。GM dAqk_raw workspace 继续使用 [block][aiv][slot][64][64] 的索引，与输入 layout 无关。

h 继续通过独立的 StateOffset 寻址：

```text
dense:  ((b*NT + chunkIdx)*NV + hv)*K*V
packed: (stateIdx*NV + hv)*K*V
```

layout 不改变 h 的这组公式。state_v_first 只决定 h 的 K/V 数学视图，不能把它当成 BNSD/BSND/NTD/TND 选择开关。

### 3.3 直接跨步 GM→L1，按 head 收集 token 行

例如 BSND 输入 `[B,T,HV,128]` 或 TND 输入 `[T,HV,128]`，固定一个 head 后，其相邻 token 行在 GM 中相隔 `HV*128` 个元素。搬运只取每个 token 的当前 head 行，不搬入其它 head 的数据：

```text
GM（BSND/TND）：
token0: head0 | head1 | head2 | ...
token1: head0 | head1 | head2 | ...
token2: head0 | head1 | head2 | ...

固定 head1 的 chunk：
  token0.head1 ─┐
  token1.head1 ─┼─ MTE2 按输入行stride读取 → L1 当前owner的head tile
  token2.head1 ─┘
```

与用户给出的 forward prepare 例子不同，KernelA 的这些输入由 Cube 消费，目标首先是 L1，不必先整理到 AIV UB。优先使用 A3 支持的 GM→L1 ND2NZ `DataCopy`，用 `srcDValue` 表达源行起点之间的元素步长，在同一次搬运中抽取 head 的数据并组织成 L1 NZ tile。该接口支持非连续源矩阵；源行 stride 与搬入的有效列数是两个独立参数。[GM→L1 ND2NZ DataCopy](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/cube_compute_ISASI/cube_compute_load/DataCopy_GMToL1_ND2NZ.md)

以固定64行容量的 L1 NZ tile 为例，参数语义如下；具体矩阵间 stride 字段名称按目标 CANN 头文件填写，不把各版本的 `srcNdMatrixStride/srcNdCubeStride` 名称混用：

| 参数                 | d_o / v_new | aqk   | 含义                                             |
| -------------------- | ----------- | ----- | ------------------------------------------------ |
| ndNum                | 1           | 1     | 首个 BSND/TND 实现逐 head 搬运                       |
| nValue               | m           | m     | 只读当前 chunk 的有效 token 行                   |
| dValue               | 128         | m     | aqk 只搬 GEMM 所需前 m 列，整块时为64            |
| srcDValue，BNSD/NTD  | 128         | 64    | 物理源行步长，即使 aqk 尾块 dValue=m，仍按64跨行 |
| srcDValue，BSND/TND      | HV*128      | HV*64 | 物理源行步长，按整个 token 的 head 数跨行        |
| dstNzC0Stride        | 64          | 64    | 为固定64行 L1 NZ tile 选择分形列组步长           |
| dstNzNStride         | 1           | 1     | NZ 中相邻源行的目标排列                          |
| 源/目标矩阵间 stride | 0           | 0     | ndNum=1，无第二个矩阵                            |

`srcDValue` 的单位是元素，不是字节，也不是行尾需要跳过的空隙。若用“行长+跳过量”说明同一次读取，完整行 D 的等价源跳过字节数是 `(HV-1)*D*2`；尾块 aqk 只取 m 列时是 `(HV*64-m)*2`。不能把这组跳过字节数直接写入 srcDValue。

ND2NZ 的 L1 目的地为分形布局，并非字节意义的连续 RowMajor `[m,D]`；其数学内容是同一个 head 的 `[m,D]` tile。前文 owner 子区的地址边界和大小继续有效，arch22 的 LoadData 必须匹配这里实际的 NZ stride。dValue 的末尾补齐不等于自动定义所有未读行；尾块硬件尺寸和有效结果范围按1.4，不能将未初始化的归约padding当作零使用。

v_new 和 aqk 先按它们在 GM 中的物理行收集；数学上的 v_new^T、Aqk^T 再由 L1→L0 的 LoadData/layout 路径表达。不能因为公式中有转置，就把源 rowStride 设成1，或按逻辑转置矩阵在 BSND/TND 中连续读取。

整块路径的 `srcDValue` 范围为 1..65535 元素，BSND/TND 的 d_o/v_new 整块直读因此在 HV<=511 时可用；HV=512 时 HV*128=65536 已超限，Aqk 则在 HV=1024 时超限。逐 head 令 ndNum=1 只能避免矩阵间 stride 溢出，不能修复同一 head 的源行 stride。首个实现按每份输入的实际 rowStride 选择搬运；超限输入仍直接 GM→L1，改为逐 token 行 ND2NZ：每次 `ndNum=nValue=1`，`srcDValue=0`（单行调用不编码跨行间距；若目标 SDK 禁止 0，再按其要求填 D），`dValue` 为 128 或 Aqk 的 m；第 i 行源地址为 `inputOffset+i*HV*D`，L1 目的地址为 `tileBase+i*16` 个 BF16 元素，`dstNzC0Stride` 仍为 64、`dstNzNStride=1`。单行调用的目的偏移与仓库已有 [`gm_to_l1<ND,NZ>` 回退实现](../../../../../../common/include/kernel/gm_to_l1_iterator.h)一致，形成与整块路径相同的 NZ tile；在全部实际 copy 完成后发布原 pair ready，不新增 key、UB 或 transpose workspace。字段限制及 padding 语义以目标 SDK 为准。[ND2NZ参数与元素单位](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_00127.html)

Host以64位整数并检查溢出地计算HV*D、任务数、tensor大小和offset；先在64位域求min(可用AIC数,Nwork)，再转换blockDim，不能先截断Nwork。转换SDK字段前逐项校验；不可表示且不满足上述单行路径的参数明确拒绝，不截断字段。逐行路径可能增加MTE2发射成本，按3.8覆盖边界并实测。

### 3.4 四 owner、pair ready 与 dO 复用

BSND/TND 相邻 head 的输入起点差为 D 个元素；BNSD/NTD 则是 T*D。原 A5 pair 搬运辅助函数不能只替换起始 offset，必须同时适配“同 head 相邻 token 的 stride”和“相邻 head 起点的 stride”。

首个实现优先逐 head 搬运，但沿用原有 pair 级 A/Q/D ready，不增加事件数量：

```text
for each actual head pair in the 4-head work item:
    Wait pair 中每个实际owner的 L1_OWNER_FREE
    for each actual head in pair:
        strided MTE2 d_o  -> owner.dO
        strided MTE2 v_new -> owner.vNew
    Set pair.A_READY on PIPE_MTE2

    for each actual head in pair:
        MTE2 h -> owner.h                   // StateOffset，state布局路径
    Set pair.Q_READY on PIPE_MTE2

    for each actual head in pair:
        strided MTE2 aqk -> owner.Aqk
    Set pair.D_READY on PIPE_MTE2
```

每个 ready 在该 pair 所有实际输入 copy 都完成的事件点发布，leader 按原公式等待，follower 由同一 MTE1 指令流顺序保护。尾窗口只有一个 head 时只搬它，不向不存在的 owner 写入或等待；generation 的奇偶分配仍跨 work 连续。

d_o 每个 head 仍从 GM→L1 搬一次，供 A/Q/D 共用；A/Q 对 L0A 中 d_o 的复用也保留。BSND/TND 只改变输入搬运参数，不缩短 owner、L0A 或 L0C 生命周期。逐 head 搬运可能增加 MTE2 指令数；以后只有在源/目标 stride 都合法、pair ready 语义相同且实测有效时，才把两个 head 合成一次多矩阵搬运。

### 3.5 输出固定 head-major，减少整张量 transpose

计算路径为：

```text
BSND/TND aqk/v_new/d_o
  → 当前head的跨步GM→L1搬运，随路组织NZ tile
  → 原A/Q/D Cube流水
  → Q/D FixPipe直接按 BNSD/NTD 输出地址写 GM
  → A dAqk_raw GM中转 → 原目标AIV mask/scale → 按 BNSD/NTD 输出地址写 GM
```

不分配完整 BNSD/NTD 输入副本，不调用 Python `.permute(...).contiguous()` 或 L2 Transpose 把整张量先转换一遍。按 BSND/TND 物理布局创建的输入可为 contiguous；不能对一个仅 permute 得到的非连续 view 假定它已经是连续 BSND/TND 存储。

假设基线把三份输入全部实体转成 BNSD/NTD，则它额外分配的 BF16 临时张量总大小为：

```text
inputElements = B*HV*T*(64+128+128)
transposeTemporaryBytes = inputElements*2 = B*HV*T*640 B
transposeGmTrafficBytes = 2*transposeTemporaryBytes  // 转换的读+写
```

本方案省掉上述临时输入副本和转换读写，以及相应的独立转换 launch；正常算子自身的输入读仍然存在。B=1、HV=96、T=8192 时，若基线确实转换全部三个输入，对应临时容量480 MiB、额外逻辑GM读写960 MiB；若基线只转换部分输入，只能按实际转换的那些输入计数。

跨步 DMA 可能增加指令开销、cache line 放大或降低带宽效率，因此不能直接把省掉的字节数换算为已获得的时延收益。正式比较使用同一数学输入的 BSND/TND 直读与“显式 transpose + head-major KernelA”两条完整链路：dense 基线为 BNSD，packed 基线为 NTD。基线时间取所有实际 transpose task 与 KernelA 的 msprof Task Duration(us) 之和，同时单独观察 KernelA 的跨步 MTE2 成本。

### 3.6 内存与双缓冲峰值

容量沿用[2.8 的资源汇总](#28-容量复核与验证)。跨步输入直接进入原四 owner 的对应子区，不经过新增 UB 整理区；A/Q 输入复用、输出 tile 和 FixPipe 目的地相同，dAqk_raw 中转 record 不依赖输入 layout。新增全量 transpose workspace 为 0 B；ND2NZ API 若需要临时 UB，则按 2.1/2.5/2.8 单列扣除，不把它误记为 transpose workspace。

仅新增 layout 元数据和输入 stride/offset 计算；generation、dAqk_raw/dAqk slot和跨核READY/FREE数量不因 BSND/TND 增加。

### 3.7 Host、模板化分发和改动位置

Host 按第零部分先检查 metadata 成对性并解析 layout，再校验 dense/packed 模式和 rank。之后按具体 layout 解析 token shape：BNSD/NTD 为 head-major，BSND/TND 交换 token 与 head 维。三份输入的dtype、连续性、语义维度和可表示stride均校验；输出按固定head-major shape校验。dense h 的NT仍等于ceil(T/64)，varlen metadata仍以总T校验。未知layout或显式layout与metadata不匹配均返回参数错误，不能按shape自动猜测布局。

本次 A3 模板参数及四个合法组合统一以[第零部分的 Device 入口与属性](#device-入口与属性)为准。Host 完成契约校验后，只为通过校验的 layout/IS_VARIABLE 组合生成模板 key；A5 后续复用同一组四个 key。stateVFirst 仅写入对应平台的 tiling data，由 AIC 在主循环前选择 Cube 模板。AIV 只需要按 dense/packed 选择固定 dAqk 输出地址，以及与 layout 无关的 dAqk_raw 索引，不必重复按输入 layout 分支，也不处理 stateVFirst。

| 改动位置                                    | 必须实现的内容                                               |
| ------------------------------------------- | ------------------------------------------------------------ |
| Python / torch / OpDef / L0 / aclnn 入口 | layout 为必选参数，显式传递四种布局 |
| InferShape / Host校验                       | 语义B/HV/T解析与固定输出shape；取消直接输入输出SameShape假设 |
| Host tiling / tiling模板                    | 四个合法layout key、stride范围校验；stateVFirst仅写入tiling data |
| arch22 Cube输入搬运                         | 拆分输入与输出offset，跨步GM→L1与正确ND2NZ/LoadData布局  |
| arch22 AIC计算分发                          | 主循环前按stateVFirst选择Cube<true>/<false>，同layout共用key |
| Cube Q/D输出、AIV dAqk输出                  | 固定TokenOutputOffset，避免被输入layout污染                  |
| state / metadata寻址                        | 独立于layout，继续使用chunk-major state与canonical varlen    |
| 精度与性能用例                              | 等值 BNSD/BSND/NTD/TND 对照、Host负例、尾块与搬运量验证                |

### 3.8 验证与验收条件

使用同一数学数据在 CPU 上分别构造连续 BNSD、BSND、NTD 和 TND 输入，比较四条路径的三个固定 head-major 输出，并独立对照 CPU golden。测试数据使用可追踪的 batch/token/head/feature 编码和非对称矩阵，避免 shape 相同掩盖错误索引。

接口用例显式覆盖 BNSD、BSND、NTD、TND；负例覆盖 layout 为空或未知、只提供一项 metadata、BNSD/BSND 带 metadata、NTD/TND 无 metadata，以及 layout 与 rank 不匹配；模板组合检查确认未声明 layout 与 IS_VARIABLE 不匹配的分支。

分发检查确认 A3 仅生成 4 个 layout key；固定其他参数时，同一 layout 的 stateVFirst=false/true 必须生成相同 key，并分别进入正确的 Cube 模板实例。两种 h 布局使用非对称矩阵验证，确认 AIC 仅在主循环前分发一次，AIV 不处理 stateVFirst。

覆盖 B=1/2，NV=1/2/3/4/5/32/96，T=1/31/32/33/63/64/65/127/128；额外覆盖T=NV，以及BSND/TND的NV=511/512和1023/1024，分别验证两类输入的整块/逐token行搬运边界，使用m>1检验跨行，并覆盖跨chunk和跨batch地址。m=1需确认实际 MMAD 封装使用 `M_hw=Align16(m)`、未因有效行数重新进入 M=1 GEMV，且只写一行；m=31/33/63需检查非对齐读取及补零。NTD/TND覆盖多序列、空序列和多个尾chunk；state_v_first两种布局、正/零/有限负scale以及重复launch按前述要求回归。

检查 aqk 的 chunk 内列不是全局 token 列，尾块只读有效 m 行和前 m 列，dAqk 上三角与尾列精确为零；v_new/d_o 保持 128 列；Q/D 和 dAqk 输出分别保持 BNSD/NTD；h 索引不受输入 layout 影响。Host 负例覆盖未知 layout、三输入语义维度不一致、非连续输入、错误输出 shape、错误 metadata 和不可表示的 DMA stride。

Profiling 或 trace 中应无额外整张量 Transpose/Contiguous kernel，无新的全量 BNSD/NTD 输入 workspace；只有当前 head 的 GM→L1 跨步搬运，d_o 仍每 head 一次 GM 读取，片上缓冲峰值符合 3.6。BSND/TND 端到端收益和跨步 DMA 效率以实测记录，本部分不声明已完成实现或性能验收。

## 第四部分：编码文件树与 A5 后续适配

### 4.1 统一接口与平台范围

本次实现直接把原接口改为带 `layout` 的单一接口，A3 支持四种布局；不保留旧签名或第二套接口。后续 A5 直接实现同一接口的四种布局。

| 能力 | A3 本次实现 | A5 后续适配 |
| --- | --- | --- |
| BNSD / BSND | dense，显式 layout | 使用同一接口和输出契约 |
| NTD / TND | packed，显式 layout 和 metadata | 使用同一接口和输出契约 |
| state_v_first | 支持，写入 tiling data，不增加 key | 同一规则 |
| 输出 | dense BNSD、packed NTD head-major | 同一规则 |
| tiling key | 4 个合法 layout/IS_VARIABLE 组合 | 复用 4 个组合，平台资源另行实现 |

layout 与 metadata 的组合必须在 Host 阶段校验：BNSD/BSND 只接受两个 metadata 均为空，NTD/TND 只接受两个 metadata 均有效；空、未知或不匹配均报参数错误。不能通过 shape 猜测，也不能自动插入 transpose。

### 4.2 属性、Host 与 Device

OpDef 属性顺序为 `scale=0`、`chunk_size=1`、`state_v_first=2`、`layout=3`。`layout` 是必选 string attr，公开值只能是 BNSD、BSND、NTD、TND；不使用空字符串哨兵。Python、L2、L0、InferShape、tiling 共用同一布局解析和 shape 契约。

Host 负责字符串解析、metadata 成对性和 canonical 内容校验、语义维度解析、输出 shape 校验、DMA 字段范围校验以及平台能力校验。Device 只接收解析后的 layout 枚举、dense/packed 标志、任务参数和地址参数；不从 GM 读取 layout，也不重复解析字符串。

A3 使用独立 tiling 结构和 arch22 实现；后续 A5 可使用独立 arch35 tiling 结构，但公共接口、布局枚举、offset/stride 工具、输出地址规则和 CPU 数学标杆保持一致。A5 适配不得改变公开参数顺序。

### 4.3 共享逻辑与平台实现隔离

| 文件层 | 责任 | A5 后续复用 |
| --- | --- | --- |
| `chunk_kda_bwd_prepare_layout.h` | 四种布局枚举及 dense/packed 关系 | 直接复用 |
| `chunk_kda_bwd_prepare_utils.h` | TokenInputOffset、TokenInputRowStride、TokenOutputOffset、StateOffset | 直接复用 |
| `chunk_kda_bwd_prepare_validation.h` | shape、dtype、metadata、输出和布局契约 | 直接复用 |
| `chunk_kda_bwd_prepare_support.h` | SoC、DMA stride、workspace 和能力检查 | 增加 arch35 能力 |
| arch22 / arch35 | 各平台搬运、Cube、Vector、同步和资源生命周期 | 各自实现，不交叉包含平台专属 API |

公共工具不依赖 CATLASS_ARCH、TPipe、跨核 flag 或平台地址；字符串和 acl/GE descriptor 只在 Host 处理。A3、A5 的 workspace 大小和片上资源可以不同，但都必须由各自 tiling 返回真实值。

### 4.4 本次文件树

以下路径相对仓库根目录。`[A]` 新增，`[M]` 修改，`[K]` 保留源码但需按统一新接口重新接入。

```text
flash-linear-attention-npu/                                  [仓库根目录]
├── fla/ops/ascendc/kda/                                     [Ascend C 算子源码目录]
    ├── chunk_kda_bwd_prepare/                                [本算子目录]
    │   ├── CMakeLists.txt                                    [M] 注册源码、编译目标和平台文件
    │   ├── README.md                                         [A] 说明接口、四布局和平台支持矩阵
    │   ├── docs/                                             [接口、设计和验收文档目录]
    │   │   ├── api.md                                        [A] 记录统一公开接口和参数契约
    │   │   ├── design_a3.md                                  [M] 记录 A3 迁移、资源和实现方案
    │   │   └── validation.md                                 [A] 记录布局、边界和精度验收项
    │   ├── op_host/                                          [Host 侧算子注册、校验和 tiling]
    │   │   ├── CMakeLists.txt                                [M] 编译 Host 注册、接口和 tiling 源码
    │   │   ├── chunk_kda_bwd_prepare_def.cpp                  [M] 注册输入输出、属性和 layout
    │   │   ├── chunk_kda_bwd_prepare_infershape.cpp           [A] 推导固定 head-major 输出 shape
    │   │   ├── chunk_kda_bwd_prepare_validation.h             [A] 统一校验 dtype、rank、shape 和 metadata
    │   │   ├── chunk_kda_bwd_prepare_support.h                [A] 校验 SoC、DMA stride、workspace 能力
    │   │   ├── op_api/                                        [L0 和 aclnn 接口实现目录]
    │   │   │   ├── chunk_kda_bwd_prepare.h                    [M] 声明唯一带 layout 的 L0 接口
    │   │   │   ├── chunk_kda_bwd_prepare.cpp                  [M] 实现 L0 launcher 和 executor 调用
    │   │   │   ├── chunk_kda_bwd_prepare_api_utils.h           [A] 共用参数校验和 executor 构造工具
    │   │   │   ├── aclnn_chunk_kda_bwd_prepare.h              [M] 声明唯一 aclnn workspace/run 接口
    │   │   │   └── aclnn_chunk_kda_bwd_prepare.cpp            [M] 实现 aclnn 参数转换和算子调用
    │   │   └── op_tiling/                                     [Host tiling 和模板 key 目录]
    │   │       ├── chunk_kda_bwd_prepare_tiling.h             [M] 定义平台无关 tiling 数据结构
    │   │       ├── chunk_kda_bwd_prepare_tiling.cpp           [M] 解析属性、计算 workspace 并生成 key
    │   │       └── chunk_kda_bwd_prepare_tiling_processor.h   [M] 组织 4 个模板组合和平台 tiling
    │   └── op_kernel/                                         [Device kernel 和平台实现目录]
    │       ├── chunk_kda_bwd_prepare.cpp                     [M] 定义入口并分发 4 个模板 key
    │       ├── chunk_kda_bwd_prepare_struct.h                [A] 定义统一及平台无关的 tiling 字段
    │       ├── chunk_kda_bwd_prepare_layout.h                [A] 定义四种 layout 枚举及 dense/packed 关系
    │       ├── chunk_kda_bwd_prepare_utils.h                 [A] 提供输入、输出和 state 的 offset/stride 计算
    │       ├── chunk_kda_bwd_prepare_tiling_key.h             [A] 声明 4 个合法模板 key 组合
    │       ├── arch22/                                       [A3 arch22 的搬运、Cube、Vector 实现]
    │       │   ├── chunk_kda_bwd_prepare_common.h             [A] 管理 A3 共享 buffer、地址和同步协议
    │       │   ├── chunk_kda_bwd_prepare_cube.h               [A] 实现 A3 A→Q→D 的 Cube 计算和写回
    │       │   └── chunk_kda_bwd_prepare_vector.h              [A] 实现 A3 mask、scale、dAqk 的 Vector 阶段
    │       └── arch35/                                      [A5 平台实现目录，本次仅保留]
    │           ├── chunk_kda_bwd_prepare_common.h              [K] 保留 A5 公共资源和同步实现，后续接入统一接口
    │           ├── chunk_kda_bwd_prepare_cube.h                [K] 保留 A5 Cube 实现，后续增加四布局搬运
    │           └── chunk_kda_bwd_prepare_vector.h               [K] 保留 A5 Vector 实现，后续接入统一输出规则
└── tests/atk/chunk_kda_bwd_prepare/                         [本算子的 ATK 用例目录]
    ├── README.md                                           [A] 说明用例生成、执行和结果判定
    ├── chunk_kda_bwd_prepare.yaml                           [A] 配置输入输出、属性和精度标准
    ├── gen_chunk_kda_bwd_prepare.py                         [A] 生成四布局及负例 JSON 用例
    ├── executor_chunk_kda_bwd_prepare.py                    [A] 构造 Python 调用并执行算子
    └── reference/chunk_kda_bwd_prepare.py                   [A] 提供 CPU 数学标杆和输出布局
```

`arch35/` 三个文件不属于本次 A3 实现修改范围；`[K]` 表示保留现有文件，后续 A5 支持 layout 时再在其上改造。

### 4.5 后续 A5 适配边界

后续 A5 只需扩展平台实现，不再改公开接口：

1. arch35 Cube 按共用输入 offset/stride 读取 BNSD、BSND、NTD、TND，并保留固定 head-major 输出地址。
2. arch35 tiling 使用同一 layout 枚举、四个模板组合和 `GET_TPL_TILING_KEY(...)`；stateVFirst 仍写入 tiling data，不增加 key。
3. A5 按自身 L0C→UB、workspace 和 mode 0x4 资源约束实现搬运与同步；不得把 A3 的 GM handoff 或片上容量直接套用到 A5。
4. A5 完成四布局的精度、边界和性能验证后，再更新平台能力表和 README。

统一 L0 声明被覆盖后，当前组合 backward 的调用点也必须纳入后续 A5 适配：`fla/ops/ascendc/kda/chunk_kda_bwd/op_host/op_api/aclnn_chunk_kda_bwd_v2.cpp` 当前调用 `l0op::ChunkKdaBwdPrepare` 时没有 layout 参数，后续应根据组合算子的实际输入 layout 显式传入 BNSD、BSND、NTD 或 TND，并同步更新其头文件包含关系、校验逻辑和测试。A3 本次不保留旧签名或兼容包装；在该调用点完成改造前，不能把旧组合入口与新 L0 声明同时编译通过作为验收条件。

### 4.6 实施与验收

实施顺序为：统一接口和属性 → 公共布局/metadata 校验 → A3 tiling 与 arch22 kernel → Python/ATK → A3 四布局精度和性能 → 后续 A5 arch35 适配。

| 验收项 | 必须证明的行为 |
| --- | --- |
| 统一接口 | 只有一组 aclnn 符号和一份 L0 签名，layout 为必选参数 |
| A3 布局 | BNSD、BSND、NTD、TND 均可执行，输出固定为 BNSD/NTD head-major |
| key | 只有 4 个 layout/IS_VARIABLE 模板组合，stateVFirst 不增加 key |
| 数学与边界 | 两种 state 布局、尾块、空序列、metadata canonical 规则和非对齐搬运正确 |
| A3 资源 | GM scratch、L1/L0/UB、ND2NZ 临时 UB、事件和 workspace 均按目标 SDK 规则实现并验证 |
| 后续 A5 | 使用同一接口和布局契约适配 arch35 |

本文件冻结接口和实现方案；实际编译、设备验证及性能结果以 ATK README 的验收记录为准。

## 第五部分：搬运与同步优化实施补充

在不改变接口、四 head 工作项、数学公式及资源容量的前提下，进一步采用 arch35 的
批量预载、批量 L1→L0 搬运和流式 cache 策略。平台差异继续按本设计处理：A 的结果
只从 L0C Fixpipe 到 GM，再由目标 AIV 搬入 UB；Q/D 各一次 Fixpipe 写公开输出。

1. ND2NZ 的源行距及相邻 head 间距均能用 16 位字段表达时，ndNum=2 将两个 head
   搬到独立 L1 owner；矩阵间距超限时逐 head，行距超限时逐 token 行。三组 pair READY
   均在两份对应输入完成后发布，不增加事件或破坏 owner 的消费后释放协议。
2. L0A 按 M 分形行批量搬 K 条带；L0B 按 N 分形列搬 K 条带并通过 dstGap 交错写入。
   转置视图在物理 row pitch 与 N 分形范围一致时，完整 NZ tile 可一次连续搬到 ZN。
   各分形地址及转置与原逐分形路径相同，非对齐 K/N 和尾块沿用原数学尺寸。
3. 完整 chunk 的三角处理改为 packed bit mask Select，再批量 Muls 缩放；启动时通过
   S_V 配对事件保证 mask 的 Scalar 写入先于 Vector 读取。Select 保证被屏蔽的 NaN/Inf
   不泄漏，尾块仍只读取已写入的有效下三角。UB 预留容量仍为 80 KiB。
4. 输入和公开输出沿用 arch35 的 CACHE_MODE_DISABLE，输入复用由 L1 承担；GM 中转
   record 保留默认 cache。FREE/READY 的双 AIV 广播确认、raw/output UB 双缓冲和
   L1/L0 的最后消费者事件均保留，不将 A5 的 mode 0x4/+16 或直接 UB 投递移植到 A3。

本次最终精度回归范围由用户指定为原始 ATK 用例 0..19，输入值域、种子及原生混合容差
不变。历史全量、确定性及内存检测结果只对应初版实现；优化版本结果另见验收记录。
