# RecurrentKda 非连续 State 支持设计

## 1. 背景与目标

RecurrentKda 的 state 常从多层 state pool 中按 layer 切片获得。此类 tensor 的逻辑 shape 正确，
但 slot/head 外层 stride 可能大于连续布局。原实现通过 `Contiguous + ViewCopy` 中转，会增加一次
完整 state 搬运，并掩盖调用侧真实 storage、stride 和 offset。

本设计参考 [flash-linear-attention-npu PR #184](https://github.com/flashserve/flash-linear-attention-npu/pull/184)，
让 kernel 按真实 stride 直接访问 state，同时支持：

- V-first：`[state_capacity,H_v,V,K]`；
- K-first：`[state_capacity,H_v,K,V]`；
- FP32/BF16 state；
- 原位与非原位 final state；
- legacy 与 arch35 kernel。

公开 aclnn/Python 接口和算子 schema 保持不变。当前仓库没有 RecurrentKda fast-kernel-launch wrapper，
本次不新增该入口。

## 2. 支持边界

本次支持外层非连续、内部二维 state 矩阵稠密。设四维 stride 为
`[s0,s1,s2,s3]`，单位为元素：

- 所有 stride 必须为正数；
- `s3 = 1`；
- `s2 = shape[3]`；
- `s1 >= shape[2] * s2`；
- `s0 >= (shape[1] - 1) * s1 + shape[2] * s2`。

后两条保证同一 slot 内的 head 平面、相邻 slot 的地址区间不重叠。允许 `s1`、`s0` 大于连续值，
因此可以覆盖从带额外 layer/padding 维的 pool 中切片得到的 view。

不支持内部矩阵转置、最后一维步长采样或负 stride；这些场景需要 gather/scatter 路径，会改变现有
计算流水和性能特征。

## 3. 地址公式

### 3.1 V-first

逻辑 shape 为 `[state_capacity,H_v,V,K]`。元素 `(slot,head,v,k)` 的 GM 元素偏移为：

```text
offset = s0 * slot + s1 * head + s2 * v + s3 * k
```

内部 `[V,K]` 平面稠密，现有按连续 K 行的批量搬运保留。每次搬运只把 slot/head/v 的基地址
替换为 stride 公式。

### 3.2 K-first

逻辑 shape 为 `[state_capacity,H_v,K,V]`。元素 `(slot,head,k,v)` 的 GM 元素偏移为：

```text
offset = s0 * slot + s1 * head + s2 * k + s3 * v
```

K-first 继续使用现有 UB 转置式逐元素搬运，以保持内部计算仍使用 V-first 形式的局部矩阵。
仅 GM 地址改为真实 stride，不改变 recurrent 计算和同步流水。

## 4. Tiling 数据与 Host 校验

共享 `RecurrentKdaTilingData` 增加输入、输出各四个 `uint64_t` stride：

```text
stateInStride0 ... stateInStride3
stateOutStride0 ... stateOutStride3
```

Host tiling 从 `GetInputStride(5)` 读取 initial state stride。输出 stride 按模式选择：

- 原位：读取 state-ref 输出 stride；若运行时未提供，则复用输入 stride；
- 非原位：读取 final-state 输出 stride。

运行时缺失 stride 元数据时，根据 state 的逻辑 shape 生成行主序连续 stride。共享 tiling processor
统一执行正 stride、内部稠密和外层无重叠校验。这样 legacy 与 arch35 使用同一套规则，未来增加
新的调用入口时也无需复制校验逻辑。

stride 在 host 校验后转换为 `uint64_t` 写入 tiling data，kernel 不再自行推导外层连续跨度。

## 5. aclnn 数据流

非 state tensor 继续执行既有的连续化和必要 cast。state 数据流改为：

1. 对非连续 `initial_state` 调用 `CreateView`，保留 view shape、storage shape、stride 和 offset；
2. 原位模式把 kernel state 输出直接指向输入 state view；
3. 非原位模式把 kernel state 输出直接指向 final state view；
4. 原位且调用者额外传入独立 final-state 输出时，保留一次必要的 `ViewCopy`；
5. 删除 state 的预先 `Contiguous` 以及执行后的通用 copy-back。

Python wrapper 不对 NPU state 调用会改变 stride 的转换。原位模式返回值保持输入 storage 与 stride；
非原位模式保持输入内容不变，final state 使用输出 tensor 自身的 stride。

## 6. Kernel 改动

legacy 与 arch35 kernel 都在构造阶段读取八个 stride 字段。

`PrefetchState`：

- V-first 按输入 stride 计算每个 slot/head/v 行的 GM 基址，继续批量搬运 K；
- K-first 按输入 stride 计算每个 `k,v` 元素地址，继续执行现有 UB 转置。

`CopyOutState`：

- V-first 按输出 stride 计算 GM 基址；
- K-first 按输出 stride 计算逐元素 GM 地址。

q/k/v、gate、beta、metadata、UB 切分、事件同步和 recurrent 数学流水均不改变。

## 7. 兼容性

- 公开 aclnn 函数签名、Python 参数和算子 schema 不变；
- 连续 state 的 stride 与旧地址公式等价；
- V-first 与 K-first 共用同一 tiling 校验，分别使用布局相关地址公式；
- FP32/BF16 继续复用已有模板实例；
- 原位 state 的 alias 语义不变；
- legacy 与 arch35 均消费同一份共享 tiling data；
- 不新增 fast-kernel-launch wrapper。

tiling data 布局发生变化，因此 run 包与 Python wheel 必须配套安装，不能混用旧 host 与新 kernel 产物。

## 8. 测试矩阵

`tests/pta/test_accuracy.py` 增加从五维 pool 的 layer 维切片构造的 NPU 非连续 view，
确保非连续性不会被 `.to(device)` 意外消除。

| 布局 | State 布局 | State dtype | V | 写回 | 索引 |
| --- | --- | --- | --- | --- | --- |
| BSND | V-first | FP32 | 128 | 原位 | packed `ssm_state_indices` |
| TND | K-first | BF16 | 256 | 非原位 | 顺序 slot |

每个正向用例比较 `out` 和 `final_state` 与 CPU golden，并检查：

- 构造出的 state 确实非连续；
- 原位模式 storage 和 stride 不变；
- 非原位模式输入 state 不变；
- 未命中的 state slot 不变；
- 相邻 layer/padding guard 未被写坏。

负向用例构造：

- 最后一维步长为 2 的内部非稠密矩阵；
- slot stride 为 1 的外层重叠 view。

两类输入都必须在 host tiling 阶段失败，不能进入 kernel。

## 9. 构建与验收

按仓库 README 的方式 B 分别构建：

1. 仅 RecurrentKda 的自定义算子 run 包；
2. `torch_custom/fla_npu` Python wheel。

验收项：

- run 包和 wheel 构建、安装成功；
- `python fla/ops/ascendc/kda/recurrent_kda/tests/pta/test_accuracy.py` 全部通过；
- `git diff --check` 通过；
- 新增非连续正向用例通过 MindStudio Sanitizer `memcheck`；
- sanitizer 前通过目标 `.o` 符号和运行日志确认实际命中 sanitizer kernel。

## 10. 风险与控制

- stride 取错输入/输出索引会导致 GM 越界或写错 pool，因此原位与非原位分别取输出 stride，并以
  guard layer 和 memcheck 验证；
- K-first 的逻辑维顺序与内部 V-first 计算顺序不同，必须只替换 GM 地址，避免改变 UB 转置流水；
- 允许重叠外层 stride 会让多个 head/slot 并发写同一地址，host 必须拒绝；
- 任意内部 stride 会引入低效 gather/scatter 且扩大越界面，本次明确拒绝；
- 新旧 tiling data 二进制不兼容，构建和安装时必须使用同一次源码产物。
