# 当前实现设计

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
