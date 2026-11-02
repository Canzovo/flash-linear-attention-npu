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

实现中以 `kHeadDim`、`kWindowTokens`、`kBlockBytes` 和元素类型的 `sizeof` 推导区域大小，后一区域起点由前一区域起点加容量得到；访问点直接使用具名偏移常量，不添加短小访问封装。补偿外积 scratch 单独以元素为单位推导，并通过静态断言检查其位于回收的 State bank 内。常量化不改变下表的实际布局或事件协议。

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
| `[156800,157312)` | 归约/掩码临时区 | norm 归约和 FP32 分量拆分交替掩码复用 |
| `[157312,161408)` | delta broadcast | 每个 delta 占一个 32 B block，外积更新消费 |
| `[161408,161440)` | FP32 beta 窗口 | 保存当前窗口至多 8 个已转换 beta |
| `[161440,161472)` | FP32 g 窗口 | 保存当前窗口至多 8 个已转换 g，g 缺省不读取 |
| `[161472,161600)` | 4 个 32 B scalar 临时块 | Exp、Ln、norm 等按最后消费顺序复用 |
| `[161600,161856)`、`[161856,162112)` | 两个 BF16 out bank | V 写、MTE3 读，按 bank 回收 |
| `[162112,162144)` | head 参数预留区 | 当前实现的 head 参数可保留在 Scalar 局部变量 |
| `[162144,162176)` | INT32 write-slot 窗口 | 当前窗口最后 token 写回发出后复用 |
| `[162176,162208)` | metadata 临时区 | 保留，不覆盖活跃 write slots |
| `[162208,162240)` | 控制量预留区 | 控制量优先使用 Scalar 局部变量 |

两块 State 共 131072 B，两个 raw bank 共 12416 B，三个 FP32 输入窗口共 12288 B，其余临时区共 6464 B，总量 162240 B。两次点积的完整乘积矩阵临时复用已释放的旧 State bank；不另行申请 UB 或 GM 外积矩阵。

### 2.4 输入、预处理与数值门禁

本轮批处理优化保持固定 UB 容量与公开语义：两条 dtype 路径均先整块计算一次衰减 State。旧 src 最后一次读取完成、对应 MTE3 回收后作为完整乘积 scratch；两次矩阵向量运算各以 repeat=128 批量发出两个半行 Mul、一次 Add fold 和一次 WholeReduceSum，按 Mul→fold→reduce→复用的真实依赖同步。FP32 补偿保留相同舍入次序，通过原地保存加法误差将七份 tile 临时量降为四份，tile 行数由 State bank 容量减去固定分量/广播区后推导并向下对齐到 8 行，得到 24 行，尾部 8 行使用实际 repeat。BF16 snapshot 仍只在旧 src 最后消费后覆盖其后半区；未引入额外 GM 中间量。

每个窗口有效长度为 `n=min(8,L-position)`。Q/K/V 分别用一次 `DataCopyPad` 描述 n 行，每行 256 B，GM 行间空隙为 `(tokenStride-128)*2`，head 偏移使用各自真实 stride。三个输入各有 ready，Q ready 后先 Cast Q，同时搬 K；K ready 后 Cast K，同时搬 V；V ready 后 Cast V。

beta/g 只收集当前 head 的 n 个值。GM 标量读取只访问有效的 2/4 B，写入对应 typed raw UB 槽，通过一次 S→V 事件后整窗口转换为 FP32；FP32 输入用 32 B UB 内 copy，不调用 Float→Float Cast。尾窗口不访问额外的 GM 值。窗口转换完成后 raw beta/g bank 才允许被下一窗口的 MTE2 复用，不覆盖仍被 token 使用的 FP32 窗口。

Q/K norm 使用 FP32 平方和及 `rsqrt(sum+1e-6)`，Q scale 只执行一次。sigmoid 在整个窗口上计算 `exp(min(beta,0))/(1+exp(-abs(beta)))`，两个指数参数均不为正，不依赖短窗口的压缩 mask 选择；allow-neg 只在 sigmoid 开启时生效。softplus 使用 `max(z,0)+log1p(exp(-abs(z)))`。普通路径令 `x=exp(-abs(z))`、`u=round(1+x)`，向量计算 `Ln(u)+(x-(u-1))/u`，补回 `1+x` 舍入丢失的小尾部。`0≤x≤1`，残差校正的高阶误差低于 FP32 精度，不需要逐 token Scalar 分支或短窗口 mask。FP32 本身的溢出、下溢和非有限行为与标杆比较。

硬件 Rsqrt 作为初值，随后对 `x=sum+1e-6` 执行三轮 FP32 Newton 修正 `r=r*(1.5-0.5*x*r*r)`。原平方和、倒数根和修正项都保存在现有向量临时块中，每轮真实读写依赖均有 V barrier。最后用 `Brcb` 将每个 token 的倒数根广播到 128 个特征，一次性缩放整个窗口；Q/K 均使用修正后的 FP32 系数。归一化路径不再逐 token 读取 Scalar 系数，也不为每个 token 发出 V→S 同步。

beta sigmoid、softplus 和普通 gate 路径均按窗口批量执行 VEC 运算。beta/g 的 GM 非连续 head 收集是唯一保留的 Scalar 元数据读取，收集结束后通过一次 S→V 事件交给窗口向量预处理。超越函数的计算顺序由必要的 V pipe 依赖屏障保护，输入值域和 ATK 精度标准保持不变。

FP32 State 的 raw gate 在有限正常域使用 Scalar double-float 预处理。每个值由两个 FP32 分量组成；TwoSum、按位拆分的 TwoProd、修正除法保存局部舍入残差，不使用 FP64 或硬件 FMA。Exp 先按双分量 ln(2) 做范围规约，再用十二阶多项式和二的整数幂缩放；Log1p 使用 `y=x/(2+x)` 的 atanh 奇次级数至 25 次。计算只占 Scalar 局部变量，不增加固定 UB。

预处理仍保留 CPU FP32 的逐步舍入顺序：`z=g+bias`、`tail=exp(-abs(z))`、`l=log1p(tail)`、`softplus=max(z,0)+l`、`magnitude=exp(A_log)`、`logGate=-magnitude*softplus`、`alpha=exp(logGate)`。每个超越函数在 double-float 内完成后舍入回 FP32，再进入下一步；不能将整条链替换为 FP64 最后一次舍入。仅当 `A_log∈[-80,80]` 且 gate、bias 均在 `[-1e20,1e20]` 时进入此路径；比较也排除 NaN/Inf。超出此域保留普通路径；极小 alpha 在安全下溢判断后直接返回零。beta sigmoid 和已处理 log gate 保留原指数路径。

该算子的编译选项限定为 `-ffp-contract=off -fno-fast-math`，使 TwoSum/TwoProd 所依赖的分步 FP32 舍入保持成立。禁止编译器把表达式收缩、重新结合或按实数代数消去残差计算；这些选项不扩展到其它算子。它们同样作用于本算子的历史 kernel，因此最终包还需重跑旧 C ABI 执行兼容性检查。数值改进的效果由原始 ATK 标准及失败用例复检评价，不改变输入范围或阈值。

### 2.5 完整 State 递推与写回

初始 FP32 State 通过 128 行 DMA 直接载入 bank 0。初始 BF16 State 搬入 `kSnapshotOffset` 对应的 `[32768,65536)`，MTE2→V 后 Cast 到 bank 0 的完整 FP32 区域。

每个 head 的 128 行乘积一次写入已回收的旧 State bank；通过 repeat=128 批量折叠每行前后 64 项，再用 WholeReduceSum 批量归约。完整乘积区在归约完成后复用。delta 通过 `Brcb(...,16,{1,8})` 扩展成 `[128,8]`。BF16 State 路径用 MulAddDst 覆盖 K 的两个 64 元素半段；FP32 State 路径使用下述补偿乘加。

FP32 State 路径先在 dst 生成完整 `alpha*src`，用 dst 计算 prediction、delta 和更新，再计算 attention。MTE3 直接读取 dst 写 checkpoint。下一 token 只读此状态并写另一 bank；该 bank 再成为目的区前等待对应 MTE3 完成。

A2/A3 的 vmla/vmadd 不能据 API 名称推断为 IEEE 单次舍入乘加。FP32 外积更新采用 TwoProd/TwoSum 补偿：按位清除乘数低 12 位尾数形成 high/low 两部分，计算乘积残差及加法残差，再将总残差加回。旧 src 的 V 消费结束且对应 MTE3 回收后，将其前 59776 B 用作分量、广播及 24 行 tile 临时区；四份 tile 临时量为 product、productError、sum、recovered。原地保存 c-v 后，旧 accumulator 不再存活，product/recovered 可继续复用。没有新增 UB 申请或 GM 中间结果。归约临时区暂作 16 位交替掩码。非有限 baseline 保留原加法结果，避免补偿残差对 Inf 引入额外 NaN。tile 内真实读写依赖均使用 V barrier。

BF16 与 FP32 State 路径均先回收 dst，整块执行一次 `alpha*src` 得到衰减 State；V 读取结束并回收 src 的 MTE3 后，src 作为整 head 点积 scratch。prediction 和外积共享同一份衰减 State，保持衰减在归约前的 FP32 运算顺序。BF16 在 attention 点积最后消费后将新 dst Cast 到旧 src 后半的 snapshot，再写 GM。下一 token 继续从完整 FP32 dst 递推；BF16 snapshot 不反馈到递推链。

State 初始读和每次 checkpoint 写均使用一次 128 行 DMA。GM slot/head 基址为 `slot*stateStride[0]+head*stateStride[1]`，行间空隙为 `(stateStride[2]-128)*sizeof(State)`。out 每 token 写 256 B，使用两个 256 B bank 避免覆盖在途 MTE3 读取。

同一请求的 token 顺序不变，MTE3 按发出次序写 checkpoint，重复槽最终保存最后 token 状态。不同请求的写冲突由调用前校验排除。

### 2.6 同步资源与 task 边界

| 资源 | 事件及数量 | 发布、等待与回收 |
| --- | --- | --- |
| 两个 raw bank 的 Q/K/V | MTE2→V，各 3 个，共 6 个 | 每次 DMA 后 ready；对应 Cast 前等待，禁止把三路 ready 合并 |
| raw QKV bank | V→MTE2，共 2 个 | 三次 Cast 完成后发布；同 bank 再预取 Q 前等待 |
| raw beta/g bank | S→V，共 1 个 | Scalar 收集后发布；窗口转换前等待，转换完成后 raw bank 才能回收 |
| State bank | V→MTE3、MTE3→V，各 2 个 | checkpoint 生产后发布，MTE3 发出后登记 free；该 bank 重新写入前等待 |
| BF16 out bank | V→MTE3、MTE3→V，各 2 个 | Cast 后发布，写回后登记 free；该 bank 重新 Cast 前等待 |
| Scalar/Vector 元数据 | S→V，共 1 个 | 仅在非连续 beta/g/slot 收集后发布，窗口向量预处理前等待 |
| task 边界 | V→MTE2、MTE3→MTE2，各 1 个 | 新 State DMA 前保证旧 UB 读及 GM 写完成 |

事件由 TPipe 分配，并在处理完全部 task 后按相同 HardEvent 类型释放。每次 SetFlag 都有匹配 WaitFlag，pending 标志只登记已发布的 free。任务结束回收所有在途 raw/scalar/state/out 槽；没有任务或空请求不会制造未发布事件的等待。

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
| `RunHead` | 根据 accepted 和索引表找到初始 State；预取首个 Q 窗口，循环准备窗口、预取下一窗口和递推 token，结束时回收在途搬运。 |
| `CopyInput` | 按输入 stride 将当前 head 的一窗 Q、K 或 V 搬到对应 raw UB 区，发布对应 MTE2→V ready 事件。 |
| `CopyWindowMeta` | 收集本窗有效 token 的 beta、可选 g 和 checkpoint slot，补齐 UB 尾部，发布 S→V 事件。 |
| `PrepareWindow` | 组织 Q/K/V 搬运与 FP32 转换；整窗计算 beta sigmoid、gate、Q/K norm 和 Q scale，标记 raw bank 可回收。 |
| `NormalizeWindow` | 批量平方、逐行归约，整窗求倒数根并做 Newton 修正，再广播行系数缩放 Q 或 K。 |
| `LoadState` | 等待上个任务的 UB/GM 使用结束，按 State stride 读入初始矩阵，并将 BF16 输入转为 FP32 工作状态。 |
| `Token` | 读取本 token 预处理结果，完成 decay、prediction、delta、外积更新、attention、out 转换及 checkpoint 写回；接续完整 FP32 State。 |
| `DotRows` | 计算矩阵各行与向量的点积，用于 `State×K` 和 `State×Q`；整 head 多行 repeat 计算，共享回收的旧 State bank 作乘积 scratch。 |
| `CompensatedOuter` | FP32 State 专用的分块外积补偿乘加，复用已消费的旧 State bank 保存乘积及加法残差。 |
| `EmitState` | 按 slot/head/行 stride 将完整 checkpoint 写回 GM，记录对应 State bank 的 MTE3 完成事件。 |
| `ReleaseEvents` | 按分配时相同的 HardEvent 类型释放全部事件资源。 |

主调用链为 `Init → Process → RunHead`。每个 head 先 `LoadState → CopyInput(Q)`，每个窗口执行 `PrepareWindow`，随后依次调用 `Token`；`PrepareWindow` 内调用 `CopyWindowMeta`、`CopyInput(K/V)` 和可选的 `NormalizeWindow`。`Token` 调用两次 `DotRows`，FP32 路径另外调用 `CompensatedOuter`，再由 `EmitState` 写回。所有任务结束后由 `Process` 调用 `ReleaseEvents`。窗口内预处理批量进行，State 递推仍按 token 先后顺序执行。
