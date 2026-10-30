# CPU 标杆检查记录

本文为 CPU 标杆准备阶段的历史记录，不描述本轮设备验收的当前状态；当前 ATK 接入及验收结论见本目录 README 和配套测试交付报告。

检查日期：2026-10-30。

本记录仅为上传 CPU 标杆的语义审查和 CPU 回归，不代表 NPU 精度、性能、内存检查或现有 ATK 工程验收完成。未修改算子 kernel、ATK 输入 range 或原生精度标准。

## 检查结果

Python 3.12.3、PyTorch 2.15.0+cpu，纯 CPU 运行。使用独立测试环境，不要求 CUDA/NPU/ATK/Triton。

`test_reference.py` 的 **16 类检查全部通过**。测试同时覆盖 FP64 golden 与 FP32 bench，包括：

- 192 组 State/Beta/g dtype、L2Norm、Beta sigmoid、raw/log/无 gate、allow_neg_eigval 组合。
- 4 组完整特征轴的密集随机回归，固定种子 20261030，覆盖两种 State dtype、norm 开关、GQA、prefix 与多请求。
- 上述 196 组合法输入均执行双标杆对照；在公开输出 dtype 上，FP64 golden 与 FP32 bench 满足测试设定的混合容差。FP32 bench 与原上传文件在这些合法输入上的 `out` 与完整 State 结果逐位一致；此为确定输入上的 CPU 回归事实，不推出所有输入/NPU 均逐位一致。
- 独立单变量手算递推，L=1/7/8/9/16/17/32/33/65，验证长序列、同请求重复槽与最后 token 覆盖。
- BF16 checkpoint 与递推状态隔离；专门构造逐 token 舍入会改变结果的用例，分别验证 FP64 golden 和 FP32 bench 的下一 token 使用未舍入状态。
- 二维 checkpoint 表，accepted 大于有效序列长度，空请求忽略无效 accepted，混合 prefix/长度。
- 卷积式混合 QKV 视图、head-first 转置、GQA 特征映射、ratio=16 和合法只读 QKV 别名。
- State 行/head/slot padding、轴置换、storage offset、同 backing storage 的不相交视图，以及合法的交错布局。
- State 原地写回、未写槽及 padding 不变；特殊 NaN 位模式在未写槽保持原样。
- 全空请求且有 prefix，超过 256 head 的合法输入。
- 跨请求写/写、初始读/写竞争，非法 dtype/shape/flag/scale/metadata/layout、State 自重叠及 State/输入别名的拦截；异常时 State 不变。
- raw gate 极值、关闭开关时不使用 A_log/dt_bias 的数值，以及 Beta/g/metadata 可连续化的展开视图。

运行输出结尾：

```text
Ran 16 tests ...

OK
```

## 复现

```bash
python test_reference.py
```

需要比较原文件时，设置 `RGDN_ORIGINAL_REFERENCE` 指向原始上传文本或其原样 `.py` 副本后运行同一命令。未设置时仍执行当前标杆的公式、布局、边界和异常检查，仅跳过与原文件的比较。

测试中的逐位比较用于 FP32 bench 与历史文件的回归及可手算用例；golden 与 bench 的比较使用输出 dtype 对应的混合容差，没有替代或放宽 ATK 原生混合容差。原文件尚未同步接入新接口的仓库 ATK executor；正式接入后仍需按当前设计验证 NPU 结果与全部 State pool。
