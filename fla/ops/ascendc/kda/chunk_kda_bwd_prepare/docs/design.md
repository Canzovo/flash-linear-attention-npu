# 当前实现设计

当前规则版本：`V2`。本轮按已冻结的 `design_a3.md` 保持四 head 工作项及整 head 的 AIV 映射。

本轮以 [design_a3.md](design_a3.md) 为完整设计，采用 arch22 的 A2/A3 GM 中转路径。
接口由 [api.md](api.md) 固定；验收用例及结果记录在
[ATK README](../../../../../../tests/atk/chunk_kda_bwd_prepare_a3/README.md)。

Stage 0 顺序计算 A、Q、D；A 通过每个 AIC 的四条 FP32 workspace record 交给两个 AIV，
Q/D 直接写公开输出。Stage 1 负责三角掩码、scale 和尾列清零。每份 L0C 只有一次 Fixpipe。

资源为 L1 288 KiB、L0A/L0B/L0C 各 64 KiB、每个 AIV UB 80 KiB、每个实际 AIC
user workspace 64 KiB。系统 workspace 单独由 SDK 查询并计入总量。

目标 SDK 查询得到 A2/A3 的 L1 可用值为 524032 字节、L0A/L0B=64 KiB、L0C=128 KiB、
UB 可用值为 196352 字节，实现使用量均在容量内。系统 workspace 为 16 MiB，
实际 AIC 数由平台查询并与任务数取最小值。ND2NZ 使用 arch22 硬件搬运路径，
SDK 实现直接调用 GM→L1 指令，不使用软件临时 UB。

CANN 9.1.0 的 TPipe 在 AIC 内部占用三个 M_MTE1 事件；算子通过 AllocEventID
另申请四个，该方向总计七个。MTE2_MTE1 申请六个，其余方向分别申请二或四个。
全部事件按同一方向成对使用并在退出前 drain/release；不手工占用内部事件编号。

CHUNK_INDEPENDENT V1 开发参考提供独立任务跨步调度和同步闭环的工程组织。
本实现的四 head 窗口、四 record、双 slot、分形布局和同步计数均取自本算子的设计。

## 批量搬运优化

以 arch35 的 pair 预载、四份 L1 owner、A/Q 共用 dO 和消费者完成后释放资源为参考。
arch22 的 ND2NZ 源行距和矩阵间距均为 16 位：两者不超过 65535 元素时，
相邻两个 head 通过 ndNum=2 搬入相隔 36864 个 BF16 元素的独立 owner；
矩阵间距超限时分别搬 head，行距超限时沿用逐 token 行搬运。
每对 head 的 A/Q/D READY 只在对应两份输入的最后一次搬运之后设置。

L1→L0 使用 LoadData 的 repeatTimes/srcStride/dstGap 批量搬运 K 条带。
L0A 每个 M 分形行发一条指令；L0B 每个 N 分形列发一条指令，dstGap=N 分形数-1。
每个 repeat 的源、目的地址和分形内转置与原单分形路径一致，M/K/N 及尾块语义不变。
当转置视图的 N 分形数等于 L1 的物理 row pitch / 16 时，NZ 到 ZN 的分形顺序相同，
整块 L0B 进一步用一次连续 LoadData 完成；其它情况沿用 K 条带搬运。
A 的 dO 继续驻留 L0A，Q 仅搬 H 到另一 L0B；D 完成 L1→L0 后释放当前 L1 owner。
两个 L0C slot 仍分别由 M_FIX/FIX_M 保护，每份结果只 Fixpipe 一次。

A 的结果保持 FP32 L0C→GM→UB，mode 2 广播由两个 AIV 同时确认 READY/FREE；
非目标 AIV 仅确认，目标 AIV 在 MTE2 完成 GM 读取后释放 workspace record。
完整 64 行 chunk 使用一次 Select 去除上三角，再一次 Muls 缩放；Select 防止无效区
NaN/Inf 通过零乘法泄漏。三角 bit mask 初始化经 S_V 配对事件后供 Vector 读取。
尾块仍只读取已初始化的有效三角区，严格清零上三角和尾列。raw UB 在 V 消费完成后
释放，独立 output UB 在 MTE3 写回完成后释放，workspace、UB、L1 和 L0 容量保持不变。

沿用 arch35 的流式 cache 策略：四份输入只从 GM 读取一次并在 L1 复用，三份公开输出
按流式写回，均设置 CACHE_MODE_DISABLE；小容量 GM 中转 record 保留默认 cache 策略。

本轮最终精度回归按用户指定以原始用例 id 0..19 为范围；其余历史验收结果不代表优化版本
已重新完成全量、确定性或内存验收。模型性能继续使用 case 198/199，原输入和种子不变，
每条预热 5 次后采集一次正式调用，读取 msprof 的 Task Duration(us)。
