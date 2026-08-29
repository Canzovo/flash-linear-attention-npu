# 算子开发 Agent 指南

本目录只保存算子开发 Agent 必须按需加载的方法和验证规则。仓库级强制约束与任务路由以根目录 `AGENTS.md` 为准；构建、安装、runtime 架构、PR 和 CI 细节保存在各自负责的文档中。

## 阅读顺序

| 任务 | 必读文档 | 继续阅读 |
| --- | --- | --- |
| 新增或修改 Ascend C 算子 | [`operator-development.md`](operator-development.md) | 当前算子的 README、设计文档和相邻实现 |
| 精度、性能、回归或交付验证 | [`validation.md`](validation.md) | [`../../tests/atk/README.md`](../../tests/atk/README.md) 和当前算子的 ATK README |
| 修改公共组件、ABI、代码生成模板或 runtime | [`../architecture/torch-npu-decoupled-architecture.md`](../architecture/torch-npu-decoupled-architecture.md) | 所有受影响算子的开发与验证文档 |
| 修改构建、wheel 或 OPP 安装 | [`../开发者指南.md`](../开发者指南.md) | 构建脚本和安装检查脚本 |
| 修改分支、PR、CODEOWNERS 或 CI | [`../repository-rules.md`](../repository-rules.md) | PR 模板和现有 workflow |

## 内容边界

- `operator-development.md` 说明如何定义能力边界、设计依赖与流水、实现和定位问题。
- `validation.md` 说明改动应覆盖哪些测试，以及 Agent 自测试如何形成闭环。
- ATK 命令、参数、环境变量和目录资产只写在 `tests/atk/README.md`。
- Python runtime、wheel、动态库、device、stream、autograd 和图编译架构只写在 `docs/architecture/`。
- 具体算子的 shape、资源数字、TilingKey、slot、同步协议和性能结果只写在该算子的 README、设计文档或 ATK README。

新增内容时先判断归属，不创建无边界的“经验汇总”文件，也不在多个文档重复维护同一强制规则。
