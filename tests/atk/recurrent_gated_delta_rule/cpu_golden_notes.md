# RecurrentGDN CPU 标杆审查与修订

本文保留 CPU 标杆准备阶段的审查记录；文末“尚未接入”等表述仅描述该阶段。本轮已完成 ATK executor 接入，设备验收结论见本目录 README 和配套测试交付报告。

对照文档：配套交付件根目录的 `recurrent_gdn_a2a3_design_20261027.md`，2026-10-30 更新的设计。

原文件的主递推公式正确，并已支持长序列、scalar gate、三个融合开关、GQA、两种 checkpoint 表和 BF16/FP32 State。主要缺项是产品约束与异常拦截；本目录现在同时提供 FP64 golden 和 FP32 bench，分别用于高精度语义校验与贴近算子计算精度的 CPU 回归。

## 审查与修改

| 设计语义 | 原文件情况 | 修订后的处理 |
| --- | --- | --- |
| QKV 仅 BF16；State/Beta/g 支持 BF16/FP32；head 参数 FP32 | 大部分浮点 dtype 未检查；query 为 FP64 时隐式切换全部计算精度 | 完整检查产品 dtype；bench 固定 FP32，golden 固定 FP64；两者都不根据输入 dtype 隐式改变计算精度 |
| K=V=128，HV/HK 为 1～16，不限制 head 总数为 256 | 未检查 K/V 和 ratio 上界 | 拒绝非法维度和 ratio，保留超过 256 head 的合法情况 |
| 四个开关必须 bool；scale 必须有限实数 | 开关可被字符串/整数的真假值误触发，scale 可接受 bool/字符串 | 在计算前检查并拒绝非法属性 |
| QKV 外层允许非连续，末轴 stride=1；DMA gap 有上限 | PyTorch 索引能读取合法非连续输入，但未拦截不支持的布局 | 保留直接视图读取，检查正 stride、末轴、token stride、DMA gap 和 storage bounds |
| State 支持行/head/slot padding、置换与 offset；禁止可写重叠和输入别名冲突 | 未检查 State 自重叠与写读别名 | 检查布局和真实占用字节区间；同 storage 中互不相交的区域仍允许 |
| 不同请求写集合不相交，初始读槽不得被其他请求写入 | 未拦截，串行 CPU 可能把本应拒绝的竞争输入算成依赖处理顺序的结果 | 完整收集写槽与初始读槽，在开始计算前检查竞争；同请求重复槽保持合法 |
| `[prefix,L0,...]`，不是 cumulative offsets；二维 accepted 可大于当前长度 | 原有实现正确 | 保留，并覆盖 prefix、空请求、二维 accepted 边界和全部索引值域 |
| 原地更新 State，只返回 out | 原函数明确为非破坏性 `(out,final_state)` 测试入口 | 保留历史入口，新增与设计同签名的原地 `recurrent_gated_delta_rule` CPU 入口 |
| 稳定 softplus 数学公式 | `F.softplus` 默认在输入大于 20 时使用线性分支 | 改为 `max(z,0)+log1p(exp(-abs(z)))`；不照搬 kernel 的小尾部多项式近似 |
| BF16 checkpoint 舍入不影响下一 token 的递推状态 | 原有 recurrent_state 与 final_state 分离，在合法输入上正确 | 保持正确行为；显式逐 token cast checkpoint，下一 token 接续独立的 FP32 或 FP64 递推状态 |
| 未写 State 槽保持不变 | 原文件先把整个 pool 转为计算 dtype，最后再转回；特殊 NaN 位模式可能被转换 | 按原 dtype 克隆 pool，仅转换真正写入的 checkpoint；原地入口仅提交写槽 |
| 8-token UB 窗口、整窗口 Cast、双 State、DMA/Event | 不模拟硬件结构 | CPU golden 直接表达数学递推，保持与 kernel 独立；硬件流水由 kernel 测试验证 |

`F.softplus` 的线性分支见 [PyTorch 官方文档](https://docs.pytorch.org/docs/2.9/generated/torch.nn.functional.softplus.html)。该处为严格公式对齐；对正常 FP32 输入通常不会造成可见最终输出差异，不作为原文件主公式错误的证据。

## 使用

依赖为 Python 3.10+ 和 PyTorch CPU。文件无需导入 NPU、Triton、ATK 或仓库包。

```python
from reference import recurrent_gated_delta_rule

# 参数均为 CPU Tensor，dtype/shape 与设计一致。
out = recurrent_gated_delta_rule(
    query, key, value, state,
    beta=beta,
    actual_seq_lengths=actual_seq_lengths,
    ssm_state_indices=ssm_state_indices,
)
# state 已原地更新，scale 缺省为 1.0。
```

需要非破坏性的两个输出时，可显式选择两条 CPU 标杆：

```python
from reference import (
    recurrent_gated_delta_rule_golden,
    recurrent_gated_delta_rule_bench,
)

golden_out, golden_state = recurrent_gated_delta_rule_golden(
    query, key, value, state, beta, 1.0,
    actual_seq_lengths, ssm_state_indices,
)
bench_out, bench_state = recurrent_gated_delta_rule_bench(
    query, key, value, state, beta, 1.0,
    actual_seq_lengths, ssm_state_indices,
)
# 两个入口都不修改输入 state；历史 reference/cpu_golden 别名指向 FP64 golden。
```

prefix 的输出在 CPU 中填零以便比较，产品对 prefix 没有数值保证；仅比较有效 token，State 比较完整 pool。没有无效区的统一数值要求，不能据此要求 NPU 清零 prefix。

CPU 双标杆均使用 PyTorch matvec/reduction，不模拟 Ascend 指令的归约树、FMA 或近似函数，因而不承诺与 NPU 逐位相等。`recurrent_gated_delta_rule_golden` 和历史 `recurrent_gated_delta_rule_reference` 在 FP64 中计算，`recurrent_gated_delta_rule_bench` 在 FP32 中计算；两者在写出 `out` 和 State 快照时都转换回公开 dtype。产品形态的 `recurrent_gated_delta_rule` 使用 bench 路径并原地更新 State。FP64 只表示 CPU 对照精度，不表示产品接受 FP64 QKV。

## CPU 检查

在本目录运行：

```bash
python test_reference.py
```

设置可选环境变量 `RGDN_ORIGINAL_REFERENCE` 为原始上传文件时，测试额外执行修改前后合法输入对比。该变量只用于测试定位，不影响标杆公式。结果见 [`validation.md`](validation.md)。

当前交付只涉及上传标杆的修订。仓库旧 ATK executor 仍有旧接口约束，尚未接入这里的新契约；本目录检查不代表现有 ATK/NPU 全量验收通过。
