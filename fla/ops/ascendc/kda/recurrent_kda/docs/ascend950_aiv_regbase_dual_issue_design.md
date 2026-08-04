# RecurrentKda Ascend950 纯 AIV RegBase 双发射设计

## 1. 背景与结论

`recurrent_kda` 的递推核心由逐元素乘加、向量归约、标量门控和 GM/UB 搬运组成，本质是纯 Vector 算子。此前将矩阵向量计算等价拆成 Cube + Vector 路径，会增加 BF16 中间表示、AIC/AIV 协同、搬运和同步开销，且没有 L1 驻留收益，性能由约 20 us 劣化到约 36 us。

本轮只保留两类优化：

1. Ascend950 arch35 使用 MicroAPI RegBase 实现热点 Vector 计算；
2. 保持 `KERNEL_TYPE_AIV_ONLY`，由 tiling 直接启动全部 AIV，让两组 Vector 单元并行处理不同任务；禁止为“1AIC:2AIV”误切到 MIX kernel。

明确不引入 Cube 计算、CATLASS GEMM、L0A/L0B/L0C、L1 驻留、AIC/AIV 中间数据交换或 L0C 到 UB 路径。

## 2. 目标与非目标

### 2.1 目标

- Ascend950 与 Ascend910B/Ascend910_93 代码隔离；
- 保持原纯 AIV 数学公式、接口、精度和状态更新语义；
- arch35 热点计算使用 RegBase；
- 全部 AIV 独立分摊 `(batch, value_head)`，避免空 AIC 调度；
- `test_accuracy.py` 全部通过；
- 相同 case 使用 msprof 测量设备侧耗时并与约 20 us 基线比较。

### 2.2 非目标

- 不做 CV 数学分解；
- 不使用 Cube、L1/L0 或新增 workspace；
- 不修改公共 API、dtype、layout、tiling key 或输出语义；
- 不修改 Ascend910B/Ascend910_93 kernel；
- 不缩减测试或放宽精度阈值。

## 3. 当前基线

- kernel 入口为 `KERNEL_TYPE_AIV_ONLY`；
- host 按 `(batch, value_head)` 任务数选择 block dim；
- arch35 已将 `MatVecMul`、K/Q 融合乘加和 ReduceSum 热点改为 `AscendC::MicroAPI`；
- 原任务循环会让每个 AIV 扫描全部 `B * NV` 组合，再通过取模判断归属；
- 相同 case 已测约 20 us，最终对比以本轮同设备 msprof 为准。

## 4. 实现方案

### 4.1 平台隔离

`op_kernel/recurrent_kda.cpp` 保持编译期头文件分支：Ascend950 包含 `arch35/recurrent_kda.h`，其他平台继续包含通用头文件；kernel 类型统一保持 `KERNEL_TYPE_AIV_ONLY`。

### 4.2 纯 AIV 双发射

Ascend950 使用 `KERNEL_TYPE_AIV_ONLY`，Block Num 由 host 的 AIV 数与任务数共同确定。该模式直接调度 Vector core，实测 `Mix Block Num=0`、`aicore_time=0`；不产生空 AIC 分支。

arch35 将“全量扫描后取模”改为每核直接步进任务：

```text
vectorCoreNum = GetBlockNum()
taskNum       = B * NV
for taskId = blockIdx; taskId < taskNum; taskId += vectorCoreNum:
    batch = taskId / NV
    head  = taskId % NV
```

每个 `taskId` 只由一个 AIV 处理，不同任务访问不同 head/state 切片，无需核间 flag 或全局同步。该方式保留全部 AIV 并行度，同时消除重复扫描、取模和无关 state metadata 校验。

### 4.3 RegBase 与同步

保持 arch35 已有 MicroAPI 热点实现，包括 state/gate、state/key、delta、`delta * key + state`、query 和 ReduceSum。RegTensor 只在 `__VEC_SCOPE__` 内使用，尾部由 `UpdateMask` 屏蔽。

继续使用 TPipe/TQue 管理 GM/UB 生命周期，保留现有 MTE2→V、V→MTE2、V→S EventID 和必要的 V pipe barrier。不增加跨核共享 buffer、workspace、L1 或 L0 占用。

### 4.4 Double buffer 结论

当前 Q/K/V/gate/beta/state 输入队列深度为 1；attn/state 输出队列由 tiling 在 UB 容量允许时选择 1 或 2。本性能 case 的每条逻辑序列只有 1 token，`V=128` 且一次 `vStep` 即可完成，输入侧不存在第二个可重叠迭代；强行把输入队列改为 2 会增加 UB 占用，可能降低 `vStep`，不纳入最终实现。输出 double buffer 机制继续保留，但单 token case 会立即 flush，收益有限。

## 5. 风险与防护

1. MIX kernel 会启动空 AIC 并增加调度开销，因此入口必须保持 `KERNEL_TYPE_AIV_ONLY`；
2. 直接任务步进必须以 `GetBlockNum()` 为步长，保证每个 `taskId` 恰好执行一次；
3. arch35 任务循环优化不得影响 Ascend910B/Ascend910_93 通用头文件；
4. double buffer 只有在存在可重叠迭代且 UB 不压缩 `vStep` 时才启用；
5. 单次 smoke 若超时，停止 profiler 并确认进程和设备状态。

## 6. 修改范围

- `op_kernel/arch35/recurrent_kda.h`：每核直接步进 `B * NV` 任务；
- `tests/pta/test_accuracy.py`：在首次 `set_device` 前加载 custom tiling；
- 新增固定 case 的 msprof 脚本；
- 本设计文档、测试报告和问题记录。

不修改 tiling：host 已按 AIV 数与任务数设置 Block Num，纯 AIV 路径可直接使用。

## 7. 验证计划

1. `git diff --check`；
2. README 方式 B 构建并安装 Ascend950 run 包和 Python wheel；
3. 新 Python 进程验证接口和 Ascend950 binary；
4. 运行 `fla/ops/ascendc/kda/recurrent_kda/tests/pta/test_accuracy.py`；
5. 相同性能 case 普通 smoke；
6. msprof 5 次预热、10 次正式采样，仅统计后 10 次 `RecurrentKda` kernel；
7. 报告 mean、median、min、max、CV 和 Block Num。

## 8. 验收与回退条件

精度必须全部通过，不允许 timeout、重复写或非法访问。最终采用“纯 AIV + arch35 RegBase + 直接任务步进”，不使用 MIX AIC、CV 分解或输入 double buffer。最终同 case 复测均值 18.381 us，相对 20.189 us 基线提升约 9.0%。
