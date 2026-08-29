# AGENTS.md

本文件是 `flash-linear-attention-npu` 的仓库级 Agent 规则。根文件只保留强制约束、任务路由和交付门禁；具体方法、命令和工具参数按任务读取对应文档。若子目录存在更近的 `AGENTS.md`，以更近文件为准。

## 开始工作前

- 先执行 `git status --short`，保护用户已有改动，不回滚、不覆盖无关内容。
- 使用 `rg` / `rg --files` 查找实现、接口、测试和文档，不凭记忆猜测路径。
- 以目标分支与当前 `HEAD` 的 diff 确定改动范围和受影响算子。
- 构建和运行验证默认面向 Linux + CANN + NPU；缺少环境时，只报告已执行项、未执行项和原因，不得伪造通过结论。
- 公开 PR、Issue、评论和测试说明不得包含内网地址、机器名、用户名、绝对路径、临时目录、日志路径、token 或其他内部环境信息。

## 任务路由

| 任务类型 | 必读内容 |
| --- | --- |
| 新增、修改或优化 Ascend C 算子 | `docs/agents/operator-development.md`、`docs/agents/validation.md`、当前算子的 README/设计文档和 `tests/atk/<op>/README.md` |
| 修改 InferShape、op_host、tiling、kernel、workspace 或同步协议 | `docs/agents/operator-development.md`、`docs/agents/validation.md` 和相邻实现 |
| 修改或新增 ATK 用例 | `docs/agents/validation.md`、`tests/atk/README.md` 和当前算子的 ATK README |
| 修改公共组件、ABI、代码生成模板或 Python runtime | `docs/architecture/torch-npu-decoupled-architecture.md`，并识别全部受影响算子 |
| 修改 wheel、OPP、构建或安装流程 | `docs/开发者指南.md` 和相关构建脚本 |
| 修改 PR、分支、CODEOWNERS 或 CI 规则 | `docs/repository-rules.md`、`.github/pull_request_template.md` 和现有 workflow |
| 修改 Triton 算子 | 当前 Triton 实现、导出入口、对应测试和 README；不要套用 Ascend C 专属实现约束 |

目录索引和按任务阅读顺序见 `docs/agents/README.md`。

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

- 精度问题未定位和正面修复前，不得放宽阈值、缩小输入范围、删除或跳过失败 case、降低覆盖强度，或用 scalar/vector 兜底替代目标 cube 路径。
- 性能结论必须来自适当的 profiling 或 CI 数据，不使用 Python wall time 直接下结论。
- 输入 shape 只能证明预期 TilingKey；实际命中必须由 host tiling UT 或运行时证据确认。
- GitHub NPU CI 与仓内 ATK 自测试是独立门禁，不能互相替代。

## 算子开发流程

1. 明确数学语义、公开接口、支持范围、性能目标和不包含范围。
2. 阅读当前算子及相邻实现，画出数据依赖、L2/L0 调用、stage、workspace 和同步关系。
3. 先更新或补齐算子设计文档，再修改 op_host、tiling、kernel、op_api、Python 导出和测试资产。
4. 按改动风险执行自测试闭环；精度通过后再形成性能结论。
5. 同步 README、aclnn 文档、示例和 CI case，检查聚焦 diff 与未跟踪文件。

## 自测试闭环

### 识别影响范围

- 算子私有改动映射到对应单算子测试和 `tests/atk/<op>`。
- 如果修改了公共组件、ABI、代码生成模板或 runtime，必须识别所有受影响的算子，并逐一完成对应测试。
- 无法可靠确定影响范围时必须显式报告并补充分析，不能把“没有匹配目录”当作无需测试。

### 快速开发迭代

1. 更新当前算子的测试契约，包括 README、YAML、生成器、已评审 JSON、executor 和覆盖矩阵。
2. 按目标 SoC 构建并安装当前代码，确认测试使用的是本轮产物。
3. 执行受影响算子的 ATK 精度测试；当前代码、当前用例和当前构建产物全部通过，才算一次有效迭代。

### 轮次验收

- 固定选中的精度通过版本，运行 ATK `all`，覆盖精度、确定性和 mssanitizer；性能相关改动再单独运行 `performance`。
- 按风险补充 ABI 契约、公开入口、Example/ST、多 SoC 和 NPU CI 验证。
- 文档-only 改动至少执行格式、链接和内容一致性检查；测试-only 改动还要验证被修改测试可以正常执行。
- 未执行的验证必须说明原因、影响和剩余风险。

具体命令、参数、环境准备和结果判定以 `tests/atk/README.md`、当前算子的 ATK README 和 `docs/开发者指南.md` 为准。

## 提交与交付

- 改动保持聚焦，不提交构建目录、wheel、run 包、测试输出、profiling 输出、缓存或临时脚本。
- PR 描述使用仓库模板，关联公开 Issue，并清楚说明范围、兼容性、已执行测试、未执行项和回退方案。
- push 新 commit 后，旧 commit 的 CI 结果不再代表当前 head；按 `docs/repository-rules.md` 重新满足合入门禁。

结束任务前确认：改动范围正确；接口、实现、测试和文档一致；相关 SoC、layout、dtype、dense/varlen 和边界场景没有遗漏；没有生成物或敏感信息混入；所有测试结论都能对应到实际执行记录。
