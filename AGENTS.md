# AGENTS.md

本文件是 `flash-linear-attention-npu` 的仓库级 Agent 规则。根文件只保留强制约束、任务路由、阶段门禁和交付要求；具体方法按任务读取 `docs/agents/` 中对应阶段文件。若子目录存在更近的 `AGENTS.md`，以更近文件为准。

## 开始工作前

- 先执行 `git status --short`，保护用户已有改动，不回滚、不覆盖无关内容。
- 使用 `rg` / `rg --files` 查找接口、实现、测试和文档，不凭记忆猜测路径。
- 以目标分支与当前 `HEAD` 的 diff 确定改动范围和受影响算子。
- 构建和运行验证默认面向 Linux + CANN + NPU；缺少环境时，只报告已执行项、未执行项和原因，不得伪造通过结论。
- 公开 PR、Issue、评论和测试说明不得包含内网地址、机器名、用户名、绝对路径、临时目录、日志路径、token 或其他内部环境信息。

## 任务路由

| 任务类型 | 必读内容 |
| --- | --- |
| 新增算子或新公开接口 | 按顺序读取 `docs/agents/interface-confirmation.md`、`reference-generation.md`、`solution-design.md`、`operator-development.md`、`operator-testing.md` |
| 修改既有接口或能力边界 | `docs/agents/interface-confirmation.md`；语义变化时继续读取 `reference-generation.md` 和后续阶段 |
| 修改数据流、stage、workspace、同步、tiling 或性能策略 | `docs/agents/solution-design.md`、`operator-development.md`、`operator-testing.md` |
| 修改 op_host、InferShape、kernel、op_api 或 Python 适配 | `docs/agents/operator-development.md`、`operator-testing.md` 和相邻实现 |
| 修改或新增 ATK 用例 | `docs/agents/operator-testing.md`、`tests/atk/README.md` 和当前算子的 ATK README |
| 修改公共组件、ABI、代码生成模板或 Python runtime | `docs/architecture/torch-npu-decoupled-architecture.md`，并识别全部受影响算子 |
| 修改 wheel、OPP、构建或安装流程 | `docs/开发者指南.md` 和相关构建脚本 |
| 修改 PR、分支、CODEOWNERS 或 CI 规则 | `docs/repository-rules.md`、`.github/pull_request_template.md` 和现有 workflow |
| 修改 Triton 算子 | 当前 Triton 实现、导出入口、对应测试和 README；不要套用 Ascend C 专属实现约束 |

目录索引、阶段输入输出和按任务阅读顺序见 `docs/agents/README.md`。

## 新接口开发流程

新接口开发必须依次完成以下五个阶段，不得从 kernel 实现开始倒推接口或标杆。

### 前置门禁：用户提供标杆

- 用户必须先提供可定位版本的标杆，包括代码、链接、接口文档、测试或可运行示例中的至少一种；同时说明版本、来源和预期支持范围。
- Agent 先分析用户标杆，再生成独立的 CPU 标杆。不得把相邻算子、论文描述或 Agent 自行猜测的语义当作用户已确认标杆。
- 标杆缺失、版本不明或关键语义冲突时，停止新接口实现，列出缺失信息并请求用户补充。

### 五阶段门禁

1. **接口确认**：形成接口契约、能力边界和待确认问题；取得用户确认后进入下一阶段。
2. **标杆生成**：分析用户标杆并生成 CPU 标杆，完成代表性 case 对齐；语义差异未解决时不得进入设计。
3. **方案设计**：形成数据依赖、L2/L0、stage、workspace、同步、tiling、精度和性能方案；设计评审通过后再开发。
4. **算子开发**：按已确认接口、CPU 标杆和设计实现各层代码，不在实现阶段静默改变语义或支持范围。
5. **算子测试**：以 CPU 标杆为精度依据，完成受影响算子的精度、边界、确定性、内存、性能及必要回归验证。

每个阶段的输出是下一阶段的输入。上游结论变化时，必须回到对应阶段更新文档、CPU 标杆、设计、实现和测试，不能只修改末端代码。

## 强制红线

### 公开接口与 ABI

- 已发布的算子原型、aclnn 接口和 `fla_npu.ops.ascendc.<op_name>` 接口必须保持参数名称、数量、类型、顺序、默认值和既有语义兼容。
- 不得把可由 tensor descriptor、已有属性、InferShape、host tiling、tiling data 或 workspace 推导的信息新增为公开接口或 L0 参数。
- 如确需修改公开接口、ABI、L0 原型、拆分或融合 L0、增加 V2 L0，实施前必须按仓库规则提交完整设计并取得 `@weinachuan` 明确确认，禁止先改后问。
- 修改 aclnn 原型时，必须同步 ctypes 类型表、wrapper 实参、schema、公开文档和 ABI 契约测试。

### L0、泛化与平台差异

- 同一算子在所有支持的 SoC 上复用同一 L0 定义、原型和 L2 调用路径；平台差异放在 host tiling、tiling data、workspace、kernel 模板或架构 trait 内部。
- 分别声明功能支持范围、性能目标和模板优势域。不得通过 shape 特例、隐藏限制或低性能 fallback 缩小既有功能范围。
- 融合实现覆盖原功能、通过精度并达到性能目标后，删除被替代的未融合 L0 路径、注册和构建入口。
- 具体算子的资源数字、窗口、slot、同步协议和授权只记录在该算子的 README 或设计文档中，不写成全仓通用规则。

### 精度与证据

- CPU 标杆必须独立于待测 NPU 实现，语义来自用户提供并确认的标杆，不得调用目标算子形成自验证。
- 精度问题未定位和正面修复前，不得放宽阈值、缩小输入范围、删除或跳过失败 case、降低覆盖强度，或用 scalar/vector 兜底替代目标 cube 路径。
- 性能结论必须来自适当的 profiling 或 CI 数据，不使用 Python wall time 直接下结论。
- 输入 shape 只能证明预期 TilingKey；实际命中必须由 host tiling UT 或运行时证据确认。
- GitHub NPU CI 与仓内 ATK 自测试是独立门禁，不能互相替代。

## 测试与交付

- 算子私有改动映射到对应单算子测试和 `tests/atk/<op>`。
- 如果修改了公共组件、ABI、代码生成模板或 runtime，必须识别所有受影响的算子，并逐一完成对应测试。
- 无法可靠确定影响范围时必须显式报告并补充分析，不能把“没有匹配目录”当作无需测试。
- 文档-only 改动至少执行格式、链接和内容一致性检查；测试-only 改动还要验证被修改测试可以正常执行。
- 未执行的验证必须说明原因、影响和剩余风险。
- 改动保持聚焦，不提交构建目录、wheel、run 包、测试输出、profiling 输出、缓存或临时脚本。
- PR 描述使用仓库模板，关联公开 Issue，并清楚说明范围、兼容性、已执行测试、未执行项和回退方案。
- push 新 commit 后，旧 commit 的 CI 结果不再代表当前 head；按 `docs/repository-rules.md` 重新满足合入门禁。

结束任务前确认：五阶段产物一致；接口、CPU 标杆、设计、实现、测试和文档同步；相关 SoC、layout、dtype、dense/varlen 和边界场景没有遗漏；没有生成物或敏感信息混入；所有测试结论都能对应到实际执行记录。
