# Recurrent Gated Delta Rule A2/A3 设计

方案设计规则版本：`V2`

接口和支持范围以 [api.md](api.md) 为准。本设计采用已确定的单 AIV、整 head、固定 UB 方案。递推在 token/window 间有依赖，不能拆分后独立计算。

## 1. 目标

融合 FP32 Q/K norm、Q scale、beta sigmoid、raw gate、State decay、prediction、delta、外积更新、attention 输出和 checkpoint 写回。完整调用链为 Python 校验 → Stable ABI → V2 ACLNN → L0 → host tiling → AIV kernel。历史 ACLNN ABI 固定走 V1，V2 ACLNN 显式设置内部属性 `use_v2`；gk 是否为空不参与版本选择，V2 拒绝非空 gk。

历史 ACLNN 签名和 V1 计算实现保持原样。共享 kernel 新增 `A_log`、`dt_bias` 输入后，历史 L0 的推导和启动列表为这两个可选位置传空 Tensor，保证输出、workspace 和 tiling 参数的位置正确；该适配不向旧接口增加参数，也不启用 V2 属性。

本方案不使用 Cube、L1、L0 或 GM 中间矩阵。每个逻辑任务独占一个请求的一个 value head，完整 State 在 UB 驻留，输入以 8 token 窗口流入，序列长度不受窗口大小限制。

性能比较固定使用同一批性能用例和同一 profiler 配置，以 `op_summary` 的目标 kernel `Task Duration(us)` 为准。基线在安装新算子前采集；新实现安装并验证后用相同脚本采集。原始结果和逐 case 对比归档至测试交付件。Python 同步耗时只用于定位执行链路。

## 2. Stage 0 完整详设

### 2.1 符号与任务映射

令 `B` 为请求数，`HV` 为 value head 数，`HK` 为 key head 数，`G=HV/HK`，`N=B*HV`，`C` 为平台可用 AIV 数。任务 `task=b*HV+hv`，对应 `hk=hv/G`。

Host 计算 `M=ceil(N/C)`、`blockDim=ceil(N/M)`。Kernel 以商和余数将连续 task 区间均衡分给各核：每核 `N/blockDim` 个，前 `N%blockDim` 个核各增加一个。任务数、shape、stride、checkpoint 表宽度、地址乘加均为 64 位。例如 `B=3,HV=5,C=32` 时 `N=15,M=1,blockDim=15`，每核一个 head；`N=65,C=32` 时 `M=3,blockDim=22`，前 21 核各 3 个任务，最后一核 2 个任务。

State 在 L0 同时作为输入和原地输出。CANN 的 `INFER_SHAPE` 会将输出描述符规范化为逻辑 shape、连续 stride 和相应 storage 大小；这不能替代调用方已有的带 padding/offset 存储。L0 在推导前按值保存 State 的 storage shape、view shape、view strides 和 offset，成功后依次恢复 view shape、strides、offset 和 storage shape。`SetViewShape` 会重建 stride，因此必须先恢复 shape 再恢复 stride；`SetStorageShape` 同时恢复实际 storage 字节数。该处理不复制 State，也不改变 kernel 的寻址和计算公式。

每个任务先计算请求起点，再检查有效长度。空请求直接跳过，既不读取初始 State/accepted，也不产生 ready/free。非空任务在本核按 token 顺序处理，State bank 的 src/dst 奇偶跨窗口连续。

Tiling 保留历史 32 位字段前缀供 legacy kernel 使用，新增 `totalTokens/keyHeads/valueHeads/stateSlots/batchSize/taskCount/checkpointWidth` 和 Q/K/V/State 的 64 位 stride。通用 kernel 仅使用新增 64 位维度寻址。

### 2.2 TilingKey 与平台边界

`D_T_STATE`、`D_T_BETA`、`D_T_G` 各自选择 BF16/FP32，使用 `ASCENDC_TPL_ARGS_DECL`、`ASCENDC_TPL_SEL` 声明全部 8 种组合，host 使用 `GET_TPL_TILING_KEY` 生成 key。g 缺省按 FP32 选择，不发出 g 读取。

A2/A3 的历史 ACLNN 入口使用历史 kernel 和历史 UB 计算，g/gk 维持旧接口语义；V2 入口显式进入新实现。A5 保留历史实现及其 shape/dtype/stride/属性限制。Host 明确拦截历史实现不具备的新能力，避免由注册新增 dtype 导致静默进入不匹配的 kernel。

### 2.3 固定 UB 地址与容量

只申请一个 `TBuf<VECCALC>`，大小 `162240 B`。平台 UB 容量至少为 `162240+2048 B`；2048 B guard 不申请、不写入。各区域起点按 32 B 对齐。

实现中以 `kHeadDim`、`kWindowTokens`、`kBlockBytes` 和元素类型的 `sizeof` 推导区域大小，后一区域起点由前一区域起点加容量得到；访问点直接使用具名偏移常量，不添加短小访问封装。常量化不改变下表的实际布局或事件协议。

| 字节半开区间 | 对象 | 生命周期与复用条件 |
| --- | --- | --- |
| `[0,65536)` | FP32 State bank 0 | 初始状态及逐 token src/dst；覆盖前回收本 bank 的 MTE3 |
| `[65536,131072)` | FP32 State bank 1 | 与 bank 0 交替保存工作状态 |
| `[32768,65536)`、`[98304,131072)` | 对应 State bank 的 BF16 snapshot 别名 | 仅旧源 State 最后消费后覆盖，不能与该 bank 的完整 FP32 状态同时有效 |
| `[131072,137280)` | raw bank 0 | Q/K/V 各 2048 B，beta/g 各 32 B |
| `[137280,143488)` | raw bank 1 | 与 bank 0 交替，支持下一窗口 Q 预取 |
| `[143488,147584)` | FP32 Q 窗口 | 整窗口 Cast，逐 token norm/scale 后消费 |
| `[147584,151680)` | FP32 K 窗口 | 整窗口 Cast，prediction/外积更新消费 |
| `[151680,155776)` | FP32 V 窗口 | 整窗口 Cast，delta 消费 |
| `[155776,156288)` | prediction/delta | 每 token 完成后复用 |
| `[156288,156800)` | FP32 out | attention 归约后 Cast，下一 token 复用 |
| `[156800,157312)` | 归约临时区 | norm 归约和 FP32 分量拆分交替掩码复用 |
| `[157312,161408)` | delta broadcast | 每个 delta 占一个 32 B block，外积更新消费 |
| `[161408,161440)` | FP32 beta 窗口 | 保存当前窗口至多 8 个已转换 beta |
| `[161440,161472)` | FP32 g 窗口 | 保存当前窗口至多 8 个已转换 g，g 缺省不读取 |
| `[161472,161600)` | 3 个 32 B 预处理临时块和 1 个预留块 | Exp、Ln、norm 等按最后消费顺序复用 |
| `[161600,161856)`、`[161856,162112)` | 两个 BF16 out bank | V 写、MTE3 读，按 bank 回收 |
| `[162112,162144)` | head 参数预留区 | 当前实现的 head 参数可保留在 Scalar 局部变量 |
| `[162144,162176)` | INT32 write-slot 窗口 | 当前窗口最后 token 写回发出后复用 |
| `[162176,162208)` | metadata 临时区 | 保留，不覆盖活跃 write slots |
| `[162208,162240)` | 控制量预留区 | 控制量优先使用 Scalar 局部变量 |

两块 State 共 131072 B，两个 raw bank 共 12416 B，三个 FP32 输入窗口共 12288 B，其余临时区共 6464 B，总量 162240 B。两次点积的完整乘积矩阵临时复用已释放的旧 State bank；不另行申请 UB 或 GM 外积矩阵。

### 2.4 输入、预处理与数值门禁

本轮批处理实现保持固定 UB 容量：两条 dtype 路径均先整块计算一次衰减 State。旧 src 最后一次读取完成、对应 MTE3 回收后作为完整乘积 scratch；两次矩阵向量运算各以 repeat=128 批量发出两个半行 Mul、一次 Add fold 和一次 WholeReduceSum，按 Mul→fold→reduce→复用的真实依赖同步。BF16 与 FP32 State 的外积更新统一使用两次半行 MulAddDst，每次 repeat=128。BF16 snapshot 仍只在旧 src 最后消费后覆盖其后半区；未引入额外 GM 中间量。

每个窗口有效长度为 `n=min(8,L-position)`。Q/K/V 分别用一次 `DataCopyPad` 描述 n 行，每行 256 B，GM 行间空隙为 `(tokenStride-128)*2`，head 偏移使用各自真实 stride。三个输入各有 ready，Q ready 后先 Cast Q，同时搬 K；K ready 后 Cast K，同时搬 V；V ready 后 Cast V。

beta/g 只收集当前 head 的 n 个值。Scalar 用 GM GetValue 读取有效的 2/4 B，用 SetValue 写入各自 32 B typed raw UB 槽；短窗尾部由 Scalar 补零，不访问额外 GM。CopyWindowMeta 完成后发布 SetFlag<HardEvent::S_V>(scalarToVector_)，PrepareWindow 在系数转换前执行匹配的 WaitFlag<HardEvent::S_V>(scalarToVector_)。随后 BF16 Cast 或 FP32 的 32 B UB 内 DataCopy 生成 FP32 窗口，并用 V barrier 保护后续系数计算。每窗系数计算末尾的 V→S 事件同时保证 raw 系数最后一次 Vector 读取完成后，后续窗口才由 Scalar 复用对应槽位。

Q/K norm 使用 FP32 平方和及 `rsqrt(sum+1e-6)`，Q scale 只执行一次。sigmoid 在整个窗口上计算 `exp(min(beta,0))/(1+exp(-abs(beta)))`，两个指数参数均不为正，不依赖短窗口的压缩 mask 选择；allow-neg 只在 sigmoid 开启时生效。softplus 使用 `max(z,0)+log1p(exp(-abs(z)))`。普通路径令 `x=exp(-abs(z))`、`u=round(1+x)`，向量计算 `Ln(u)+(x-(u-1))/u`，补回 `1+x` 舍入丢失的小尾部。`0≤x≤1`，残差校正的高阶误差低于 FP32 精度，不需要逐 token Scalar 分支或短窗口 mask。FP32 本身的溢出、下溢和非有限行为与标杆比较。

硬件 Rsqrt 作为初值，随后对 `x=sum+1e-6` 执行三轮 FP32 Newton 修正 `r=r*(1.5-0.5*x*r*r)`。原平方和、倒数根和修正项都保存在现有向量临时块中，每轮真实读写依赖均有 V barrier。最后用 `Brcb` 将每个 token 的倒数根广播到 128 个特征，一次性缩放整个窗口；Q/K 均使用修正后的 FP32 系数。归一化路径不再逐 token 读取 Scalar 系数，也不为每个 token 发出 V→S 同步。

beta sigmoid、softplus 和普通 gate 路径均按窗口批量执行 VEC 运算。beta/g 恢复 Scalar 收集和每窗一次 S→V 交接；checkpoint slot 仍仅由 Scalar 读写，不交给 Vector。超越函数的计算顺序由必要的 V pipe 依赖屏障保护，输入值域和 ATK 精度标准保持不变。

beta sigmoid、raw gate 以及 norm 的 Rsqrt/Newton 修正分别使用独立的外部 Mask 区域：入口调用 `SetMaskNorm()`、`SetVectorMask<float, MaskMode::NORMAL>(8)`，区域内统一使用高维切分 API 的 `<float, false>` 重载，mask 参数为 `MASK_PLACEHOLDER`、repeatTime 为 1。单目及标量运算使用 `{1,1,8,8}`，双目运算使用 `{1,1,1,8,8,8}`；Duplicate 使用 block stride=1、repeat stride=8。每段结束执行 `ResetMask()`；norm 的三轮 Newton 循环复用同一次 Mask 设置。beta 和 gate 的 Cast/DataCopy 放在各自外部 Mask 区域之前，Brcb 放在 norm 区域恢复默认 Mask 之后，避免其他 API 改写共享 Mask。有效元素仍为补齐后的 8 个 FP32 值，原算术顺序及 V pipe 依赖屏障保持一致。

编译使用 `--cce-auto-sync=off`，选项在算子通用 `add_ops_compile_options` 中设置，覆盖 `BUILD_OPS_RTY_KERNEL=OFF` 的二进制包构建。softplus 的取负→Exp、Duplicate→Exp、Mul→Muls，以及 norm 的行内 fold→归约、拷贝→Rsqrt、Newton 的 Muls→Adds 均显式表达 V pipe 依赖。norm 的 fold 和归约分别按有效行数批量发出，通过一次 V barrier 连接，保持每行加法及归约顺序。系数计算完成后每窗使用一次 V→S 事件，保证 Token 的 beta/alpha `GetValue` 读取已完成的 Vector 结果；该事件放在 Q/K norm 前，不为每 token 增加等待。

BF16 和 FP32 State 统一消费 `PrepareWindow` 已完成的 FP32 raw gate 向量结果。每窗通过向量 Exp、Ln、Div 和逐元素运算计算 `alpha=exp(-exp(A_log)*softplus(g+bias))`；现有 softplus 向量尾部修正保持原样。Token 只读取当前 token 的 alpha，不保存 raw gate 副本，不按 State dtype 或输入值域切换到标量重算。

已删除 `recurrent_gated_delta_rule_df.h`、全部引用及其 double-float 函数调用；不把该文件的函数迁移到其他位置。对应 raw gate 临时拷贝、一次 FP32 State 窗口内 V barrier、逐 token GetValue、范围比较、分支、标量多项式与幂缩放循环一并去除。当前 UB 恢复为 162240 B，第四临时块保留地址预留，State 写回生命周期不变。

本算子的 `-ffp-contract=off -fno-fast-math` 编译选项保持原样，以保留两版既定 FP32 运算顺序；本轮不改变 V1 编译选项、接口或计算源码。精度使用原始 ATK 双标杆标准，不修改输入范围或阈值。性能仍固定前10条 BF16 State 用例，因此该矩阵只用于回归，不据此宣称 FP32 raw gate 的性能收益；V1 对比沿用删除前同轮采集的相同输入基线，当前 V2 重新采集。

### 2.5 完整 State 递推与写回

初始 FP32 State 通过 128 行 DMA 直接载入 bank 0。初始 BF16 State 搬入 `kSnapshotOffset` 对应的 `[32768,65536)`，MTE2→V 后 Cast 到 bank 0 的完整 FP32 区域。

每个 head 的 128 行乘积一次写入已回收的旧 State bank；通过 repeat=128 批量折叠每行前后 64 项，再用 WholeReduceSum 批量归约。完整乘积区在归约完成后复用。delta 通过 `Brcb(...,16,{1,8})` 扩展成 `[128,8]`。BF16与FP32 State均用MulAddDst覆盖K的两个64元素半段。

FP32 State 路径先在 dst 生成完整 `alpha*src`，用 dst 计算 prediction、delta 和更新，再计算 attention。MTE3 直接读取 dst 写 checkpoint。下一 token 的 Muls 可与 MTE3 同时读取 src；Muls 后、DotRows 把 src 改作乘积 scratch 前，等待对应 MTE3 完成。

BF16 和 FP32 State 均直接以 `MulAddDst` 更新外积：`target += key * delta`。K=128分成两个64元素半行，每次repeat=128批量覆盖全部value行，key的repeat stride为0，delta使用Brcb广播。外积路径不使用TwoProd/TwoSum、乘积拆分、误差恢复、有限值mask或额外tile scratch；完成两次互不重叠的半行更新后，通过一次V barrier供attention点积读取。

BF16 在 Muls 前等待 dst 中上一 token 的 BF16 snapshot 搬出；FP32 在 Muls 后等待 src 中上一 token 的 FP32 checkpoint 搬出。两种路径的等待位置通过 State dtype 的 `if constexpr` 固定，首 token 没有待回收 State 事件。src 回收后作为整 head 点积 scratch。prediction 和外积共享同一份衰减 State，保持衰减在归约前的 FP32 运算顺序。BF16 在 attention 点积最后消费后将新 dst Cast 到旧 src 后半的 snapshot，再写 GM。下一 token 继续从完整 FP32 dst 递推；BF16 snapshot 不反馈到递推链。

State 初始读和每次 checkpoint 写均使用一次 128 行 DMA。GM slot/head 基址为 `slot*stateStride[0]+head*stateStride[1]`，行间空隙为 `(stateStride[2]-128)*sizeof(State)`。out 每 token 写 256 B，使用两个 256 B bank 避免覆盖在途 MTE3 读取。

同一请求的 token 顺序不变，MTE3 按发出次序写 checkpoint，重复槽最终保存最后 token 状态。不同请求的写冲突由调用前校验排除。

### 2.6 同步资源与 task 边界

| 资源 | 事件及数量 | 发布、等待与回收 |
| --- | --- | --- |
| 两个 raw bank 的 Q/K/V | MTE2→V，各 3 个，共 6 个 | 每次 DMA 后 ready；对应 Cast 前等待，禁止把三路 ready 合并 |
| 整块 raw bank | V→MTE2，共 2 个 | Q/K/V Cast 及 beta/g 转换全部完成后发布；同 bank 再预取 Q 前等待 |
| raw beta/g bank | S→V，共 1 个 | Scalar 收集及补零完成后发布，Vector 系数转换前等待；后续 Scalar 复用由每窗 V→S 完成保证 |
| FP32 beta/g 窗口 | V→S，共 1 个 | 每窗预处理完成后发布并等待，保护窗口内所有 Scalar GetValue；位于 Q/K norm 前 |
| State bank | V→MTE3、MTE3→V，各 2 个 | checkpoint 生产后发布，MTE3 发出后登记 free；该 bank 重新写入前等待 |
| BF16 out bank | V→MTE3、MTE3→V，各 2 个 | Cast 后发布，写回后登记 free；该 bank 重新 Cast 前等待 |
| task 边界 | V→MTE2、MTE3→MTE2，各 1 个 | 新 State DMA 前保证旧 UB 读及 GM 写完成 |

事件由 TPipe 分配，并在处理完全部 task 后按相同 HardEvent 类型释放。raw/state/out 不再维护 pending 数组，也不预置空闲事件。首窗与后续窗口分别实例化 `RunWindow<true/false>`：首窗预取未使用的 bank 1，不等待 rawFree；后续预取必然复用前一轮已发布 rawFree 的槽，固定等待后再搬入。

首窗的第 0、1 个 token 分别实例化 `Token<true,false>`、`Token<false,false>`，后续实例化 `Token<false,true>`。模板参数分别表示首 token 和 out 槽是否复用；State 的回收位置再由 dtype 的 `if constexpr` 选择。长度为 1 时不执行第二个 token；后续窗口存在时首窗必然已有 8 个 token，因此稳态 out 两槽均已发布过 free。Q/K/V 输入编号同时作为 CopyInput 模板参数，搬运来源通过 `if constexpr` 选择。融合开关、序列长度和尾窗判断来自运行时 tiling 或 metadata，保留普通 if。

对非空 head，令 token 数为 L、窗口数为 W=ceil(L/8)。State 每 token 发布一次 free，后继 token 消费 L-1 次，末尾按 dtype 等待最后一次。out 稳态消费 max(L-2,0) 次，末尾消费 min(L,2) 个最后使用的槽。raw 预取复用消费 max(W-2,0) 次，末尾消费 min(W,2) 个最后处理的窗口槽。末尾槽由最后 token/window 的奇偶位置确定，不等待未使用的槽。每个槽下一次 Set 前均已消费上一 Set；所有 head 结束时事件排空。空请求和空任务跳过 RunHead，不产生通知或等待。

V pipe 内的同址读写使用 `PipeBarrier<PIPE_V>()`，不能用它代替跨 pipe 事件。State 的两次 bank 轮转、raw 的窗口轮转和 out 的 token 轮转各自闭环。

### 2.7 GM 搬运与 workspace 汇总

每个非空请求、value head 初始读取一份 State；每 token 读取本 head 的 Q/K/V、beta 和可选 g，写一份完整 checkpoint 和一行 out。共享同一 key head 的多个 value head 按各自任务读取 Q/K，不进行跨核共享。A_log/dt_bias 在 raw gate 时按 head 读取一次，metadata 用于起点和槽位寻址。

通用 kernel 不使用用户 GM workspace。ACLNN 路径仍预留框架系统 workspace，不能将其计为 GM 中间矩阵。性能分析分别记录任务时长和各 pipe 指标，pipe 时间有重叠，不能相加得到任务时长。

### 2.8 V2 各函数职责与调用流程

| 函数 | 职责 |
| --- | --- |
| `RGDRV2` 构造函数 | 保存本次 tiling 参数，供寻址、任务分配和功能分支使用。 |
| `Init` | 绑定各 GM Tensor，申请整块 UB，并按 raw、State、out 槽位分配硬事件。 |
| `Process` | 将 `B×HV` 个请求/head 任务分给 AIV，计算请求起点，跳过空请求，调用 `RunHead`，最后释放事件。 |
| `RunHead` | 根据 accepted 和索引表找到初始 State；预取首个 Q 窗口，分别调用首窗及稳态窗口，结束时按最后槽位回收在途搬运。 |
| `RunWindow` | 按首窗模板决定 raw 复用等待和首两个 token 的调用；预取下一窗口后执行剩余稳态 token。 |
| `CopyInput` | 以模板参数选择 Q/K/V 输入，按输入 stride 将当前 head 的一窗数据搬到对应 raw UB 区，发布对应 MTE2→V ready 事件。 |
| `CopyWindowMeta` | Scalar 收集本窗 beta、可选 g 和 checkpoint slot，补零系数尾部后发布 S→V ready。 |
| `PrepareWindow` | 组织 Q/K/V 搬运与 FP32 转换；整窗计算 beta sigmoid、gate、Q/K norm 和 Q scale，标记 raw bank 可回收。 |
| `NormalizeWindow` | 批量平方、逐行归约，整窗求倒数根并做 Newton 修正，再广播行系数缩放 Q 或 K。 |
| `LoadState` | 等待上个任务的 UB/GM 使用结束，按 State stride 读入初始矩阵，并将 BF16 输入转为 FP32 工作状态。 |
| `Token` | 读取本 token 预处理结果，完成 decay、prediction、delta、外积更新、attention、out 转换及 checkpoint 写回；接续完整 FP32 State。 |
| `DotRows` | 计算矩阵各行与向量的点积，用于 `State×K` 和 `State×Q`；整 head 多行 repeat 计算，共享回收的旧 State bank 作乘积 scratch。 |
| `EmitState` | 按 slot/head/行 stride 将完整 checkpoint 写回 GM，记录对应 State bank 的 MTE3 完成事件。 |
| `ReleaseEvents` | 按分配时相同的 HardEvent 类型释放全部事件资源。 |

主调用链为 `Init → Process → RunHead → RunWindow`。每个 head 先 `LoadState → CopyInput<kQueryInput>`，每个窗口执行 `PrepareWindow`，随后依次调用 `Token`；`PrepareWindow` 内调用 `CopyWindowMeta`、`CopyInput<kKeyInput/kValueInput>` 和可选的 `NormalizeWindow`。`Token` 调用两次 `DotRows`，外积统一使用 `MulAddDst`，再由 `EmitState` 写回。所有任务结束后由 `Process` 调用 `ReleaseEvents`。窗口内预处理批量进行，State 递推仍按 token 先后顺序执行。

### 2.9 Host 结构化报错

ACLNN、inferShape 和 tiling 的拒绝分支使用 `op_common/log/log.h` 的类型化宏，向 ErrorManager 上报具体参数、实际值、期望条件和 `EZxxxx` 编号。上层传播失败时保留原消息，不再用通用日志覆盖具体原因。executor、内部 view/输出分配和启动器失败上报内部错误。

workspaceSize/executor 空指针检查位于 DFX phase-1 之前，保留框架原有返回状态并补全具体参数名。V2 的完整 descriptor、属性、storage 和别名检查也在 DFX 缓存查询之前执行，避免相同 shape 命中缓存时绕过写别名检查。V1 的公开签名、计算流程和计算源码保持原样，仅整改报错分支。直接接口脚本同时检查返回状态、`aclGetRecentErrMsg()`、executor/workspace 和 State 未变，错误编号及复现入口见接口说明和 ATK README。
