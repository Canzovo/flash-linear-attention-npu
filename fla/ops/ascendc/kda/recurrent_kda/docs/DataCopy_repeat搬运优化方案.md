# RecurrentKDA DataCopy repeat 搬运优化方案

## 1. 目标与结论

目标是审计 `op_kernel/arch35/recurrent_kda.h` 中全部 `DataCopy`，减少逐行地址计算和重复寄存器搬运指令，并以 ATK profiler 时间验证收益。

关键结论：

1. arch35 RegTensor `DataCopy` 没有“输入 repeatTimes，一条指令装入任意多行”的公开接口；单个 RegTensor 只能承载一个向量宽度。
2. repeat 优化应分为三类：
   - LocalTensor 向量 API 的 `repeatTime + RepeatParams`，一次处理多行；
   - RegBase `PostLiteral::POST_MODE_UPDATE`，在 VF 循环中自动更新 UB 指针；
   - arch35 dual `vlds/vsts`，一次装入或存出两个寄存器。
3. 当前公开接口只支持 `K=128`，性能用例固定命中 K=128 专用函数，不会进入 generic `MatVecMul`、`ProcessKQ` 和 `ReduceSumDispatch`。
4. 首要性能候选应是 K=128 热路径的 dual load/store；`MatVecMul` repeat 化作为独立 generic 候选验证，不与热路径改动混合。

## 2. API 证据

当前 CANN arch35 对应 `__NPU_ARCH__ == 5102`。Reg Compute DataCopy 公共接口提供：

- 单寄存器 load/store；
- `PostLiteral::POST_MODE_UPDATE` 指针更新；
- `AddrReg` 地址模式；
- dual load：`DataCopy<T, LoadDist>(reg0, reg1, src)`；
- dual store：`DataCopy<T, StoreDist>(dst, reg0, reg1, mask)`。

dual FP32 合法布局只有：

- load：`LoadDist::DIST_DINTLV_B32`；
- store：`StoreDist::DIST_INTLV_B32`。

其语义不是“前 64 / 后 64”，而是把连续 128 个 FP32 按偶数列和奇数列分发到两个寄存器，写回时再交织恢复原布局。

LocalTensor 向量 API 的 `Mul`、`MulAddDst`、`Select` 等支持 `repeatTime` 和 `BinaryRepeatParams`，这才是无需逐行 RegTensor DataCopy 的批量 repeat 计算方式。

## 3. DataCopy 分类清单

### 3.1 已是块搬运，不修改

以下调用已经使用 MTE block/repeat 参数，不存在逐元素 DataCopy 循环：

- GM→UB 的 gate、q、k、v：`DataCopyPad + DataCopyExtParams`；
- state GM→UB、UB→GM：`DataCopyPad + DataCopyParams`；
- FP32 state queue→UB：LocalTensor 批量 `DataCopy(count)`；
- attention UB→GM：`DataCopyPad`。

这些调用的 blockCount、blockLen、srcStride/dstStride 已表达二维搬运，改成 RegBase repeat 不会减少指令，反而会破坏 MTE 流水。

### 3.2 一次性 RegBase load，不需要 repeat

- gate/k/q/vec 的函数级只读加载；
- beta 标量广播加载。

它们位于循环外，只执行一次。可在 K=128 场景把成对的 64-lane load 合并成 dual deinterleave load，但不能使用多行 repeat。

### 3.3 可使用 LocalTensor repeat 的 generic 函数

#### MatVecMul

当前为“列块循环 × 行循环”，每行执行 load→mul→store。可迁移 legacy kernel 已验证的 repeat 写法：

```cpp
uint8_t rowStride = alignK_ / FP32_NUM_PER_BLOCK;
for (uint32_t col = 0; col < alignK_; col += REPEAT_LENTH) {
    uint64_t mask = min(REPEAT_LENTH, alignK_ - col);
    for (uint32_t row = 0; row < rows; row += MAX_REPEAT_TIME) {
        uint8_t repeat = min(MAX_REPEAT_TIME, rows - row);
        Mul(dst[row * alignK_ + col],
            cube[row * alignK_ + col],
            vec[col],
            mask,
            repeat,
            {1, 1, 1, rowStride, rowStride, 0});
    }
}
```

这样去掉逐行 RegTensor DataCopy 和逐行计算循环，由一条 repeat Mul 处理多行。当前 K=128 公开路径不会调用它，因此不预期改善固定 ATK 性能。

#### ReduceSumDispatch

可使用高阶 `ReduceSum<float, Pattern::Reduce::AR, true>` 取代三个手工 RegBase 归约函数。该改动同样只影响 generic 路径，且需要单独评估编译体积与泛化精度，不与首轮热路径候选合并。

### 3.4 不能仅靠 DataCopy repeat 去循环

- `ProcessKQ`：每行需要从紧凑标量数组广播不同 cube 值，标量行距为 4 字节，不能用 block-stride repeat 直接表达；
- `ProcessDeltaKQReduce128`：每行存在 delta→state update→dot→reduce 的真实数据依赖；
- `DecayMatVecReduce128`：每行需要更新 state 并立即做点积归约；
- `ReduceSum64/128/VF`：每行 ReduceSum 产生独立标量，DataCopy 只是依赖链的一部分；
- `ReduceSumVF` 的列循环：同一个累加寄存器跨列块存在 RAW 依赖。

这些循环不能简单删除；可优化的是成对 load/store 或指针 post-update。

## 4. 候选 A：K=128 dual load/store

### 4.1 DecayMatVecReduce128

将：

```cpp
DataCopy(state0, addr);
DataCopy(state1, addr + 64);
// ...
DataCopy(addr, state0, mask);
DataCopy(addr + 64, state1, mask);
```

改为：

```cpp
DataCopy<float, LoadDist::DIST_DINTLV_B32>(state0, state1, addr);
// state0/state1 分别为偶/奇列
DataCopy<float, StoreDist::DIST_INTLV_B32>(addr, state0, state1, mask);
```

gate 和 vec 同样使用 dual load。

### 4.2 ProcessDeltaKQReduce128

k、q、state 使用 dual load，state 使用 dual store。偶/奇列上的 update 和 dot 完全独立，两个乘积寄存器相加后 ReduceSum 仍等于原 128 列总和。

### 4.3 ReduceSum128

dual load 后两个寄存器分别含偶/奇列，按 lane 相加再 ReduceSum，与“前 64 + 后 64”具有相同数学总和。

### 4.4 等价性证明

设原向量为 `x[0..127]`。dual load 得到：

- `x0[lane] = x[2 * lane]`；
- `x1[lane] = x[2 * lane + 1]`。

逐元素运算对每一列独立，因此布局置换不改变结果。点积归约：

```text
sum_lanes(x0 * y0 + x1 * y1)
= sum_even(x * y) + sum_odd(x * y)
= sum_0^127(x * y)
```

dual store 使用匹配的 `DIST_INTLV_B32`，恢复原 state 内存布局。归约加法配对顺序会由“前后半区配对”变为“相邻偶奇列配对”，浮点舍入可能有微小差异，因此必须通过四项精度门禁验证，不能只做位级假设。

## 5. 候选 B：PostLiteral 指针更新

若候选 A 有收益，再将每行 source/destination 指针提到循环外，使用
`PostLiteral::POST_MODE_UPDATE` 自动增加 `alignK_` 或 1，减少循环内乘法和地址生成。

该模式仍保留 VF 循环；其作用是减少标量地址计算并帮助编译器生成更紧凑的 repeat 循环，不宣称“一条指令处理全部行”。

候选 B 必须在候选 A 之后独立评估，避免无法区分 dual DataCopy 与地址更新的收益。

## 6. 候选 C：MatVecMul repeat Mul

独立把 generic `MatVecMul` 改为 LocalTensor repeat Mul。验收重点是：

- 方式 A 编译；
- generic kernel 定向单元测试或源码契约测试；
- 代码生成中不再出现逐行 RegTensor load/store；
- 固定 K=128 ATK 性能预计不变，不以其无变化否定 generic 优化本身。

若仓库公开接口继续限制 K=128，候选 C 不与性能交付版本绑定，除非有可运行的 kernel 级 generic benchmark 证明收益。

## 7. 实施顺序

1. 复用当前两轮 ATK profiler 基线；
2. 只实现候选 A 的 `DecayMatVecReduce128` dual load/store；
3. 方式 A 构建、PTA 精度、两轮 ATK performance；
4. 有稳定收益才叠加 `ProcessDeltaKQReduce128` dual load/store；
5. 再评估 PostLiteral 指针更新；
6. generic `MatVecMul` repeat 化单独实现和验证；
7. 只保留两轮总耗时均改善且关键长 case 不稳定退化的候选；
8. 最优版本执行 ATK accuracy、PTA、torch_custom、严格 pytest 和至少两轮 performance。

## 8. 停止条件

任一候选满足以下条件即回退：

- CCE 不支持所选 dual dist 或出现寄存器 spill；
- dual deinterleave 改变结果超过现有精度阈值；
- 两轮性能没有稳定提升；
- 长 case 稳定退化；
- 改动只影响当前不可达的 generic 路径且没有可运行性能证据。

## 9. 修改范围

首轮只修改：

- `fla/ops/ascendc/kda/recurrent_kda/op_kernel/arch35/recurrent_kda.h`。

同时新增：

- 本设计文档；
- DataCopy repeat 开发问题记录；
- 最终测试报告。

不修改 host tiling、UB 容量、同步、算子接口、测试数据或精度容差。
