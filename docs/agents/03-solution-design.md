# 方案设计

本阶段把已确认接口和 CPU 标杆转换为可评审的 NPU 算子方案。设计必须说明数据如何流动、资源如何规划、各阶段如何同步，以及如何证明精度、泛化和性能。

## 阶段输入

- 已确认的接口契约、支持矩阵和性能目标。
- 已与用户标杆对齐的 CPU 标杆和基础 case。
- 当前算子或相邻 Ascend C 算子的实现、README 和设计文档。
- 目标 SoC、CANN 版本及仓库现有工程约束。
- 优化任务还需要当前精度基线、profiling 数据、目标 shape 和性能目标。

复用相邻算子的计算结构和工程模式，不直接复制其 tile、窗口、slot、同步计数或资源数字。

## 设计文档要求

新增算子、新接口、融合、较大重构或性能策略调整必须先完成设计文档，至少包含：

1. 目标、非目标、接口摘要和支持矩阵。
2. 用户标杆、CPU 标杆和关键数学定义。
3. 修改前后的 L2/L0 调用图与数据依赖图。
4. stage 划分、AIC/AIV 分工和生产者/消费者关系。
5. L1、UB、workspace、slot、flag 和生命周期规划。
6. host tiling、tiling data、TilingKey、模板和多 SoC 差异。
7. 精度、边界、确定性、内存、性能和回归验证计划。
8. 风险、兼容性、旧路径收敛和回退方式。

设计中的资源数字必须针对当前算子、目标 shape 和 SoC 重新推导，并写出推导依据。

## 优化任务的现状与瓶颈分析

优化任务不从预设技巧开始。每次都要重新读取当前工作树中的设计文档、实现、CPU 标杆和测试，先确认文档与代码一致，再判断现有方案是否还有优化空间。

1. 对照实际代码还原当前 L2/L0、stage、workspace、同步、tiling、模板和任务划分；文档过期时先更新现状，不在错误基线上设计。
2. 固定目标 SoC、shape、构建产物、测试输入和 profiling 方法，记录当前精度与性能基线。
3. 用 profiling 区分 Scalar、MTE、VEC、CUBE、AIC/AIV wait、资源占用或无效计算等瓶颈，并定位到具体 stage 和代码路径。
4. 对每个候选方案说明改动点、瓶颈依据、预期收益、适用范围、资源代价、精度/同步风险和回退方式。
5. 比较候选方案后只实施有证据支撑的方案；无法证明瓶颈或收益时，应报告当前结论，不为“必须优化”而修改代码。

优化方案默认冻结公开接口和 ABI、数学语义、输入输出与属性、支持范围、CPU 标杆和精度标准。若候选方案需要改变这些内容，停止优化评审，将任务重新归类为接口变更或新功能。

## 数据依赖和分层

先把计算分成三类：

- 无跨 chunk/task 依赖的大并行计算：优先按 batch、head、chunk、tile 切分，矩阵主路径使用 AIC cube/Catlass。
- 有串行依赖的状态传播：单独设计阶段、调度和 workspace，明确 carry 的生产与消费顺序。
- layout、cast、copy、mask 和边界适配：放在合适的 AIV、L0 或 L2 层，避免污染矩阵热路径。

PyTorch 层串联多个算子只能证明功能可组合，不能替代 Ascend C L2/L0 对 workspace、layout、dtype、同步和性能边界的控制。

## 多阶段协同和 workspace

多阶段算子必须为每条数据边明确生产者、消费者、owner 和生命周期：

- AIC 负责矩阵主路径和可复用矩阵结果。
- AIV 负责 gate、scale、mask、cast、padding 和逐元素修饰。
- workspace 是 producer-consumer 协议的一部分，不只是临时地址。

每个 workspace slot 都要说明由谁写、由谁读、何时可复用。ready/free flag 必须成对设计；空任务、tail chunk 和 varlen 无效区也要保持计数协议。生产者覆盖 slot 前必须确认消费者已释放。

## 维度、任务和 tiling

- 显式推导 `H_out`/`H_do` 与 `H_qk` 的 head ratio，说明 Q/K head、输出 head 和 workspace slot 的映射。
- 结合 `K`、`V`、`chunkSize` 推导模板、tile、UB/L1 预算和 workspace。
- 分别设计 fixed length 与 varlen 的 loop index、batch、token 起点和有效长度映射。
- host tiling 校验 `cu_seqlens`、`chunk_indices`、shape、属性和尾块约束；kernel 只消费已计算的任务描述。
- 编译期模板承载 dtype、固定维度和改变核心路径的选项；运行时 tiling 承载规模、offset、layout、workspace 和任务划分。
- 不为每个属性组合滥用 TilingKey，不把平台或模板选择泄漏为新的公开/L0 参数。

建议将 fixed/varlen offset 逻辑封装为统一 strategy，并在设计中枚举全部可达 TilingKey 及其选择条件。

## 搬运和同步方案

- 热路径尽量整行或整 tile 连续搬运，避免内层循环的小搬运和逐元素标量访问。
- double buffer 必须让 MTE 与 VEC/CUBE 实际重叠，并说明各 buffer 的 owner 切换时机。
- UB slot 更换 owner 前闭合跨 pipe 生命周期；`PipeBarrier<PIPE_V>()` 不能替代 MTE/V、MTE/CUBE 或 MTE3 事件。
- 尾块优先使用 padding 和中性值继续走批量 cube/vector 路径，不用 scalar/vector 替代矩阵主路径。
- 明确 cross-core flag、pipe event、MTE3 写回、workspace 复用和异常/空任务路径的同步关系。

## 精度和性能计划

精度计划应说明如何用 CPU 标杆逐阶段对比，而不只检查最终输出：

- 为关键中间量、chunk、head、状态和最终输出定义可观察点。
- 区分结构性错误、数值误差、padding/无效区和非确定性问题的定位路径。
- 明确累加 dtype、workspace dtype、算法迭代、混合容差和阈值依据。

性能计划应先判断预期 bound，再定义目标：

- Scalar、MTE、VEC、CUBE 和 AIC/AIV wait 分别使用什么 profiling 证据。
- 目标 shape、基线、目标值、泛化矩阵和模板优势域。
- 性能优化失败时如何保持既有功能范围和 correctness 路径。

## 阶段输出

设计评审前确认：

- 接口、CPU 标杆、调用图和数据依赖一致。
- 每个 stage、workspace、slot、flag 和同步事件都有唯一职责。
- 所有支持维度、边界、SoC 和可达 TilingKey 有实现及测试计划。
- 公共组件或 ABI 改动已展开所有受影响算子和回归范围。
- 风险、兼容策略、旧路径删除条件和回退方案明确。
- 优化任务已记录当前基线、瓶颈证据、优化前后比较口径和预期收益。

设计评审通过后，进入 [`04-operator-development.md`](04-operator-development.md)。未经确认的设计不得通过实现细节固化。
