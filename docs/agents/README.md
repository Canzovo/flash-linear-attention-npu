# 算子开发 Agent 指南

本目录按新接口开发的五个阶段组织，每个阶段只维护该阶段需要的输入、方法、输出和退出门禁。仓库级强制约束与任务路由以根目录 `AGENTS.md` 为准；构建、安装、runtime 架构、PR 和 CI 细节保存在各自负责的文档中。

## 工作流入口

### 新接口开发

用户必须先提供可定位版本的参考资料或参考实现。Agent 不直接开始写算子，而是先分析这些材料、生成或验收 CPU 标杆，再完成设计、开发和测试：

```text
用户提供参考资料或参考实现
  -> 接口确认
  -> 确定 CPU 标杆来源与验收路径
       -> 用户提供 GPU 服务器：运行原始 GPU 参考实现
       -> 只有 NPU 服务器：独立运行 Triton Ascend 参考实现
       -> 用户直接提供 CPU 标杆：验收并补全
  -> 完成 CPU 标杆对齐或验收
  -> 方案设计
  -> 算子开发
  -> 算子测试
```

GPU 和 Triton Ascend 路径都使用相同输入与 CPU 标杆比较。用户直接提供 CPU 标杆时不重复生成另一份，但必须确认它与接口一致、可以运行且不依赖待开发 NPU 算子；详细规则见 [`02-reference-generation.md`](02-reference-generation.md)。

### 算子特性修改

特性修改先对比当前接口、CPU 标杆、设计、实现和测试，再根据影响分流：

```text
当前算子 + 本次特性要求
  -> 接口、语义、支持范围或 CPU 标杆需要变化
       -> 01 差异确认 -> 02 标杆更新或验收 -> 03 -> 04 -> 05
  -> 接口和 CPU 标杆不变，只修复已支持场景
       -> 03 修复方案 -> 04 实施 -> 05 新旧场景回归
  -> 仅内部性能变化
       -> 转入算子优化流程
```

特性修改只允许改变已确认的差异，未受影响的行为必须保持兼容。

### 算子优化

“优化、提速、降低时延、提升吞吐、减少 workspace”从当前实现进入 `03`，不默认重走 `01`/`02`：

```text
当前工作树的设计、实现、CPU 标杆、测试和 profiling
  -> 03 现状与瓶颈分析，更新方案设计
  -> 04 实施内部优化
  -> 05 精度、泛化和性能回归
```

优化任务必须重新读取当前工作树中的阶段文件和算子文档，公开接口、数学语义、支持范围与 CPU 标杆保持不变。需要改变其中任一项时，停止优化并转入新接口或接口变更流程。

## 五阶段文件

| 阶段 | 文件 | 主要输入 | 阶段输出 | 退出门禁 |
| --- | --- | --- | --- | --- |
| 1. 接口确认 | [`01-interface-confirmation.md`](01-interface-confirmation.md) | 用户提供的参考资料、需求和支持范围 | 接口契约、能力边界、待确认问题 | 用户确认接口和语义 |
| 2. 标杆生成 | [`02-reference-generation.md`](02-reference-generation.md) | 已确认接口、用户提供的参考实现或 CPU 标杆 | 参考实现分析、独立 CPU 标杆、基础对齐 case | CPU 标杆完成对齐或验收，差异已闭环 |
| 3. 方案设计 | [`03-solution-design.md`](03-solution-design.md) | 接口契约、CPU 标杆、目标 SoC/性能要求 | 算子设计文档和验证计划 | 设计评审通过 |
| 4. 算子开发 | [`04-operator-development.md`](04-operator-development.md) | 已确认设计、CPU 标杆、相邻实现 | op_host、tiling、kernel、op_api 和适配代码 | 实现与前三阶段产物一致，基础精度可运行 |
| 5. 算子测试 | [`05-operator-testing.md`](05-operator-testing.md) | 当前实现、CPU 标杆、接口与设计 | 精度、边界、确定性、内存、性能和回归结论 | 所有必测项闭环，未执行项已说明 |

上游产物发生变化时，从对应阶段重新向后检查，不能只修改某个下游文件。

## 既有任务按需读取

| 任务 | 必读文件 |
| --- | --- |
| 优化、提速、降低时延、提升吞吐或减少 workspace | 当前算子的 README/设计、实现和测试，以及 `03-solution-design.md`、`04-operator-development.md`、`05-operator-testing.md` |
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
