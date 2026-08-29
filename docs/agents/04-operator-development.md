# 算子开发

本阶段依据已确认接口、CPU 标杆和评审通过的设计实现 Ascend C 算子。开发阶段负责把设计落成代码，不重新定义接口语义、能力边界或测试标准。

## 阶段输入

- [`01-interface-confirmation.md`](01-interface-confirmation.md) 输出的接口契约。
- [`02-reference-generation.md`](02-reference-generation.md) 输出的 CPU 标杆和对齐 case。
- [`03-solution-design.md`](03-solution-design.md) 输出的、分别评审通过的 Stage 划分和具体详设。
- 当前算子的相邻实现、生成输入、构建入口和代码规范。
- 特性修改任务还需要已确认的差异清单、兼容策略和既有场景回归范围。
- 优化任务还需要经评审的瓶颈结论、性能基线和优化前后比较口径。

任一输入缺失或互相冲突时，返回对应阶段修正；Stage 划分或具体详设任一步未评审通过时不得开始编码，不在代码中增加隐藏限制或临时语义。

对于新算子，本阶段是首次允许查看仓内其他算子的代码、README 和设计文档。只参考目录组织、注册、构建、API 封装、平台适配和编码方式；Stage、公式依赖、数据落点、资源规划、同步协议和支持范围仍以已评审设计为准。参考代码暴露出设计缺口时，返回 `03` 更新并重新评审，不得让现有代码写法反向决定方案。

## 实现顺序

建议按依赖关系实施：

1. 更新算子定义、InferShape、参数校验和错误语义。
2. 实现 host tiling、tiling data、任务描述、workspace 计算和 TilingKey 选择。
3. 实现 L2/L0、stage 调度、kernel 模板、搬运与同步。
4. 同步 op_api/aclnn、schema、ctypes、Python wrapper 和公开导出。
5. 接入 CPU 标杆和最小精度 case，确认当前实现可以进入完整测试。

修改生成文件前先查明 YAML、生成器或模板来源；应修改生成输入时，不要只改生成结果。op_host、kernel、op_api、Python 和测试层中的参数名称、顺序、类型及默认值必须与接口契约一致。

## 特性修改实施

- 只实现 `01` 和 `03` 已确认的特性差异，保持未受影响的接口、语义、支持场景和 CPU 标杆结果不变。
- 新增或修改参数、校验、tiling、kernel、op_api、schema、wrapper 和测试时，按设计同步所有受影响层级。
- 修复已支持场景时，以现有接口契约和 CPU 标杆为准，不通过修改预期结果适配错误实现。
- 实施中发现需要扩大差异范围时停止编码：接口或支持范围变化返回 `01`，CPU 标杆变化返回 `02`；Stage 类型、依赖、数据落点、驻留份数或容量规划变化返回 `03` 第一步，其他数据流、地址、同步或资源细节变化返回 `03` 第二步。

## 优化任务实施

- 只修改评审方案覆盖的内部实现，例如 tiling、任务划分、stage、workspace、同步、搬运、模板和流水。
- 保持公开接口和 ABI、数学语义、输入输出与属性、支持范围、CPU 标杆和精度标准不变。
- 每轮尽量只引入可独立验证的优化变量，保留可复现的优化前基线，避免多个策略同时变化后无法归因。
- 实现中发现瓶颈判断、Stage 划分、资源预算或同步设计不成立时，先返回 `03-solution-design.md` 对应步骤更新证据和方案，再继续编码。
- 不得用 shape 特例、隐藏限制、低性能 fallback、放宽阈值或修改 CPU 标杆制造目标场景收益。
- 如需改变公开参数、返回值、默认值、dtype/layout 语义、数学语义或支持范围，立即停止优化任务，并转入接口变更或新功能流程。

## 复用与平台差异

从相邻 Ascend C 算子复用：

- host tiling、tiling data 和有限模板选择模式。
- AIC/AIV 分工、Catlass GEMM、blocked solve 和状态传播结构。
- workspace 生命周期、cross-core flag、pipe event 和写回协议。
- fixed/varlen、head ratio、tail chunk 和多 SoC 的工程处理方式。

不复制具体算子的 tile、窗口、slot 数、同步计数或未经推导的资源常量。同一 L0 定义和调用路径服务所有支持 SoC；平台差异放入 tiling、workspace、kernel 模板或架构 trait。

## host tiling 与模板实现

- host 侧完整校验 shape、dtype、layout、属性、`cu_seqlens`、`chunk_indices` 和尾块约束。
- host 侧生成 kernel 所需的任务描述、offset、有效长度、workspace 和模板字段，避免 kernel 热路径重复推导。
- kernel 入口只选择设计中枚举的有限模板实例，模板内部使用 `if constexpr` 裁剪路径。
- TilingKey 与选择条件必须一一对应；不支持组合在 host 侧明确拦截。
- 不把 SoC、模板、workspace 或可推导信息新增为公开接口或 L0 参数。

## kernel、搬运和同步实现

- 矩阵主路径使用适合的 cube/Catlass 实现，不以 scalar/vector 兜底替代目标路径。
- 热路径优先连续整行或整 tile 搬运，避免内层循环中的小搬运和 `GetValue/SetValue` 标量访问。
- tail 和 partial chunk 优先使用 padding、中性值和有效区 mask 保持批量路径。
- double buffer、MTE、VEC/CUBE、MTE3 和 cross-core 事件必须按设计成对闭合。
- workspace slot 覆盖前确认消费者已释放；空任务和 varlen 无效区仍要遵守 ready/free 计数协议。
- `PipeBarrier<PIPE_V>()` 只解决对应 pipe 内依赖，不能代替跨 pipe 事件。

实现过程中若发现原设计无法满足容量、同步或性能约束，先更新设计和评审结论，再改变 Stage、slot 或协议。涉及 Stage 类型、依赖、数据落点、驻留份数或 L1/UB 容量时返回 Stage 划分；只涉及已确认 Stage 内的地址、调度、同步或 API 细节时返回具体详设，不能直接在代码中改变。

## 开发期精度定位

开发期使用 CPU 标杆进行小 shape 和逐阶段对比：

- 固定行/chunk/head 的块状误差、维度错位或整片符号异常：检查任务映射、layout、offset、mask、搬运和写回。
- 同一输入重复结果不一致：检查 workspace 重叠、cross-core flag、pipe event 和 UB 提前复用。
- NaN/Inf：定位第一处非有限中间值，检查累加、归一化、状态传播和未初始化数据。
- 随机分散误差：再评估累加 dtype、cast 时机、算法迭代和混合容差。
- padding/无效区误差：回到接口契约确认公开语义，并让实现和测试后处理保持一致。

不得通过修改 CPU 标杆、放宽阈值、删除 case 或缩小已确认支持范围来适配当前实现。

## 开发期性能定位

性能分析必须基于 profiling，并与设计目标对照：

- Scalar bound：检查热路径标量访问和逐元素循环。
- MTE bound：检查搬运是否过碎、重复、非连续或粒度过小。
- VEC bound：检查是否承担应由 cube 完成的矩阵工作，或 repeat 粒度过小。
- CUBE bound：检查 tile、数据复用、矩阵形状和有效计算占比。
- AIC/AIV wait：检查 producer-consumer 队列、flag、MTE3 写回、double buffer 和流水距离。

性能目标 shape 达标后仍要保留设计中的功能范围和泛化矩阵，不能用单个锚点代替完整验收。

## 阶段输出

进入完整测试前确认：

- 实现未偏离已确认接口、CPU 标杆和设计。
- 实现中的每个 Stage 类型、公式、依赖、数据落点、L1/UB/GM 规划和 head 处理与已评审的 Stage 划分及具体详设一致。
- op_host、InferShape、tiling、kernel、op_api、schema、Python 导出同步。
- 所有设计中的模板、TilingKey、workspace 和同步路径均有对应代码。
- 最小合法 case 和关键 stage 已能与 CPU 标杆对比。
- 没有把构建产物、调试输出、profiling 数据、缓存或临时脚本加入提交。
- 实现中发现的新限制、风险或设计变化已回写上游阶段文档。
- 特性修改的每项代码变化都能对应已确认差异，未受影响场景保留兼容路径。
- 优化任务的实现改动能够逐项对应瓶颈证据和已评审方案。

完成开发期自检后，按 [`05-operator-testing.md`](05-operator-testing.md) 执行正式验证。
