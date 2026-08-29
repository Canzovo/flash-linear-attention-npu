# 算子开发方法

本文说明在本仓开发 Ascend C 算子时如何定义边界、完成设计、组织实现和定位问题。仓库级强制红线以根目录 `AGENTS.md` 为准；具体算子的参数、资源数字、同步协议和性能结论必须记录在该算子的 README 或设计文档中。

## 推荐开发顺序

```text
需求与数学语义
  -> 公开接口和能力边界
  -> 可复用实现与相邻算子
  -> 数据依赖和 L2/L0 调用图
  -> stage、tiling、workspace 和同步设计
  -> op_host、kernel、op_api 与 Python 适配
  -> 小 shape 精度
  -> 目标 shape、泛化和边界精度
  -> 确定性、内存和性能验证
  -> 文档、测试和交付检查
```

不要从“先写一个 kernel 试试”开始。实现前先确定算什么、支持什么、信息从哪里获得，以及各阶段如何生产和消费数据。

## 三类信息源

### 数学语义

从论文、参考 Python/Triton 或三方实现确认：

- 公式、计算顺序和返回值语义。
- 输入输出 shape、dtype、layout 和状态参数。
- 初始状态、最终状态、padding、无效区和异常行为。
- 哪些误差来自算法近似，哪些属于实现错误。

### NPU 实现

从本仓相邻 Ascend C 算子确认：

- host tiling、tiling data 和模板选择方式。
- AIC/AIV 分工、Catlass GEMM、blocked solve 和状态传播。
- workspace 生命周期、cross-core flag、pipe event 和写回协议。
- fixed/varlen、head ratio、tail chunk 和多 SoC 的处理方式。

复用的是计算结构和工程模式，不是具体算子的 tile、窗口、slot 数或同步计数。

### 工程契约

从算子定义、aclnn 头文件、Python wrapper、README、测试和仓库规则确认：

- required/optional、默认值、dtype、format、shape 和错误语义。
- op_host、InferShape、tiling、kernel、op_api、schema 和 Python 导出的对应关系。
- 支持范围、性能目标、测试矩阵和 PR 合入要求。

修改生成文件前先查明 YAML、生成器或模板来源；应修改生成输入时，不要只改生成结果。

## 先定能力边界

新增或修改算子前，明确记录：

- 本次支持的 layout、dtype、shape、SoC、chunk、head 关系和状态参数。
- 预留但暂不支持的参数，以及对应的 host 拦截和反向用例。
- 公开输出、中间量和 workspace 的边界。
- padding、无效 token、partial chunk 和脏区是否有公开语义。
- 功能支持范围、性能目标和模板优势域。
- correctness fallback 与性能路径的适用条件。

不能验证的能力不得宣称已支持。已有公开接口、ABI 和 L0 约束按根目录 `AGENTS.md` 执行。

## 设计文档

新增算子、融合、较大重构或性能策略调整，应先完成当前算子的设计文档。至少包含：

1. 目标、非目标和支持矩阵。
2. 数学定义、输入输出和信息来源。
3. 修改前后的 L2/L0 调用图与数据依赖图。
4. stage 划分、AIC/AIV 分工和生产者/消费者关系。
5. L1、UB、workspace、slot、flag 和生命周期规划。
6. host tiling、tiling data、TilingKey、模板和多 SoC 差异。
7. 精度、边界、确定性、内存和性能验证方案。
8. 风险、兼容性、旧路径收敛和回退方式。

设计中的资源数字必须针对当前算子、shape 和目标 SoC 重新推导，不能从其他算子直接复制。

## 数据依赖和分层

先把计算分成三类：

- 无跨 chunk/task 依赖的大并行计算：优先按 batch、head、chunk、tile 切分，矩阵主路径使用 AIC cube/Catlass。
- 有串行依赖的状态传播：单独设计阶段、调度和 workspace，明确 carry 的生产与消费顺序。
- layout、cast、copy、mask 和边界适配：放在合适的 AIV、L0 或 L2 层，避免污染矩阵热路径。

PyTorch 层串联多个算子只能证明功能可组合，不能替代 Ascend C L2/L0 对 workspace、layout、dtype、同步和性能边界的控制。

## 多阶段协同和 workspace

多阶段算子必须明确每条数据边的生产者、消费者、owner 和生命周期：

- AIC 适合矩阵主路径和可复用矩阵结果。
- AIV 适合 gate、scale、mask、cast、padding 和逐元素修饰。
- workspace 是 producer-consumer 协议的一部分，不只是临时地址。

每个 workspace slot 都要说明由谁写、由谁读、何时可复用。ready/free flag 必须成对设计；空任务、tail chunk 和 varlen 无效区也要保持计数协议。生产者覆盖 slot 前必须确认消费者已释放。

## 维度与策略

线性注意力算子的维度通常相互耦合：

- `H_out` 或 `H_do` 可能是 `H_qk` 的整数倍，必须显式推导 head ratio，并正确映射 Q/K head 和 workspace slot。
- `K`、`V`、`chunkSize` 会共同决定模板、tile、UB/L1 预算和 workspace。
- fixed length 与 varlen 的 loop index 到 batch、token 起点和有效长度的映射不同。

建议把 fixed/varlen offset 逻辑封装为统一 strategy。host tiling 负责校验 `cu_seqlens`、`chunk_indices`、shape 和尾块约束，kernel 热路径只消费已计算的任务描述。

## 搬运、同步和生命周期

- 热路径尽量整行或整 tile 连续搬运，避免内层循环中的小搬运和逐元素标量访问。
- double buffer 必须能够让 MTE 与 VEC/CUBE 实际重叠。
- UB slot 更换 owner 前闭合跨 pipe 生命周期；`PipeBarrier<PIPE_V>()` 不能替代 MTE/V、MTE/CUBE 或 MTE3 事件。
- 同一输入多次运行结果不一致时，优先检查 cross-core flag、pipe event、workspace 重叠和 UB 提前复用。
- 矩阵主路径不要用 scalar/vector 兜底。尾块优先通过 padding 和中性值继续使用批量 cube/vector 路径。

## 编译期模板与运行时 tiling

编译期模板适合 dtype、固定维度和会改变核心计算路径的选项；运行时 tiling 适合规模、offset、layout、workspace 和任务划分。

host tiling 写入必要字段，kernel 入口选择有限的模板实例，模板内部使用 `if constexpr` 裁剪路径。不要为每个属性组合滥用 TilingKey，也不要把平台和模板选择泄漏成新的公开或 L0 参数。

## 精度定位

复杂算子尽量支持逐阶段对比，不要只检查最终输出。

- 结构性错误：固定行/chunk/head 的块状误差、维度错位、NaN/Inf 或固定输入结果不一致。优先检查索引、layout、offset、mask、同步和 workspace。
- 数值误差：误差随机分散且没有固定结构。再评估累加精度、迭代次数、workspace dtype、混合容差语义和性能取舍。
- padding 或无效区误差：先确认该区域是否有公开语义，再决定 kernel 或测试后处理方式。

出现 NaN/Inf 时追踪第一处非有限值或极大值；第一现场通常比最终输出更接近根因。

## 性能定位

性能结论以 profiling 为准。先判断主要 bound，再改代码：

- Scalar：检查热路径中的 `GetValue/SetValue` 和逐元素循环。
- MTE：检查搬运是否过碎、重复、非连续或粒度过小。
- VEC：检查是否承担了应由 cube 完成的矩阵工作，或 repeat 粒度过小。
- CUBE：检查 tile、数据复用、矩阵形状和有效计算占比。
- AIC/AIV wait：检查 producer-consumer 队列、flag、MTE3 写回、double buffer 和流水距离。

性能目标 shape 通过后仍要验证模板优势域和原功能范围，不能只用单个锚点证明算子可交付。

## 交付闭环

交付前确认：

- 接口契约、代码拦截、报错文本、返回码和文档约束一致。
- op_host、tiling、kernel、op_api、schema、Python 导出和测试资产同步。
- bugfix 有稳定触发问题的回归用例。
- 公共组件改动的影响面和回归范围已按根目录 `AGENTS.md` 闭环。
- 性能优化同时提供性能锚点、泛化矩阵和实际 TilingKey 命中证据。
- 构建、测试、profiling、缓存和临时产物没有进入提交。
- 对外只记录公开测试项和结果，不暴露内部环境信息。

具体测试范围和自测试流程见 [`validation.md`](validation.md)。
