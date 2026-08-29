# 算子开发 Agent 指南

本目录按新接口开发的五个阶段组织，每个阶段只维护该阶段需要的输入、方法、输出和退出门禁。仓库级强制约束与任务路由以根目录 `AGENTS.md` 为准；构建、安装、runtime 架构、PR 和 CI 细节保存在各自负责的文档中。

## 新接口开发入口

用户必须先提供可定位版本的标杆。Agent 不直接开始写算子，而是按以下顺序分析标杆、生成 CPU 标杆，再完成设计、开发和测试：

```text
用户提供标杆
  -> 接口确认
  -> 标杆生成
  -> 方案设计
  -> 算子开发
  -> 算子测试
```

## 五阶段文件

| 阶段 | 文件 | 主要输入 | 阶段输出 | 退出门禁 |
| --- | --- | --- | --- | --- |
| 1. 接口确认 | [`01-interface-confirmation.md`](01-interface-confirmation.md) | 用户标杆、需求和支持范围 | 接口契约、能力边界、待确认问题 | 用户确认接口和语义 |
| 2. 标杆生成 | [`02-reference-generation.md`](02-reference-generation.md) | 已确认接口、用户标杆 | 标杆分析、独立 CPU 标杆、基础对齐 case | CPU 标杆与用户标杆对齐，差异已闭环 |
| 3. 方案设计 | [`03-solution-design.md`](03-solution-design.md) | 接口契约、CPU 标杆、目标 SoC/性能要求 | 算子设计文档和验证计划 | 设计评审通过 |
| 4. 算子开发 | [`04-operator-development.md`](04-operator-development.md) | 已确认设计、CPU 标杆、相邻实现 | op_host、tiling、kernel、op_api 和适配代码 | 实现与前三阶段产物一致，基础精度可运行 |
| 5. 算子测试 | [`05-operator-testing.md`](05-operator-testing.md) | 当前实现、CPU 标杆、接口与设计 | 精度、边界、确定性、内存、性能和回归结论 | 所有必测项闭环，未执行项已说明 |

上游产物发生变化时，从对应阶段重新向后检查，不能只修改某个下游文件。

## 既有任务按需读取

| 任务 | 必读文件 |
| --- | --- |
| 修改既有接口、属性或支持范围 | `01-interface-confirmation.md`；语义变化时继续读取后四个阶段 |
| 修改数据依赖、stage、workspace、同步、tiling 或性能策略 | `03-solution-design.md`、`04-operator-development.md`、`05-operator-testing.md` |
| 修改 op_host、kernel、op_api 或 Python 适配 | `04-operator-development.md`、`05-operator-testing.md` |
| 修改 ATK、精度、性能或回归用例 | `05-operator-testing.md`、[`../../tests/atk/README.md`](../../tests/atk/README.md) 和当前算子的 ATK README |
| 修改公共组件、ABI、代码生成模板或 runtime | [`../architecture/torch-npu-decoupled-architecture.md`](../architecture/torch-npu-decoupled-architecture.md) 和所有受影响算子的阶段文档 |

## 内容边界

- 五个阶段文件不重复维护同一规则；跨阶段强制规则只写在根 `AGENTS.md`。
- ATK 命令、参数、环境变量和目录资产只写在 `tests/atk/README.md`。
- Python runtime、wheel、动态库、device、stream、autograd 和图编译架构只写在 `docs/architecture/`。
- 具体算子的 shape、资源数字、TilingKey、slot、同步协议和性能结果只写在该算子的 README、设计文档或 ATK README。
- 不创建无边界的“经验汇总”文件；新增知识应归入它实际约束的阶段。
