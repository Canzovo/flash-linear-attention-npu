# ChunkKdaBwdPrepare ATK

ATK 测试脚本、配置和生成的 JSON 仅保留在开发工作目录，不随本次源码提交。
下述命令用于该工作目录中保留的测试文件。

调用 `fla_npu.ops.ascendc.chunk_kda_bwd_prepare`，固定 BF16、C=64、K=V=128。
输入为 BNSD/BSND 密集布局或 NTD/TND packed 布局，输出统一为 BNSD/NTD。
状态按 chunk 排列，两种 `state_v_first` 均覆盖。packed 元数据允许空序列，chunk 对必须规范排序。
完整接口见算子 `docs/api.md`，计算及同步方案见 `docs/design_a3.md`。

精度采用 ATK 26.8.8 原生 `mixed_tolerance_bm`，CPU 标杆逐 chunk 使用 FP64 计算，
按公开输出 dtype 舍入。保留原始正态分布（std=0.05）、原始 200 个用例结构及首轮种子；
增加两个固定种子，不修改输入值域和原有覆盖。

## 生成

```bash
cd tests/atk/chunk_kda_bwd_prepare_a3
atk case -f chunk_kda_bwd_prepare_a3.yaml -p gen_chunk_kda_bwd_prepare_a3.py -dt 200 -en 0 -s 20260831
python gen_chunk_kda_bwd_prepare_a3.py --kind full --atk-template-json result/chunk_kda_bwd_prepare_a3/json/all_chunk_kda_bwd_prepare_a3.json
python gen_chunk_kda_bwd_prepare_a3.py --kind mss
python gen_chunk_kda_bwd_prepare_a3.py --kind perf
```

精度 JSON 共 600 条，记录 id = 原始 case_id + 200 × seed_index。
executor 使用保留的 `case_id` 选择结构，并使用 JSON 中的 `seed` 生成实际数据。
内存与确定性 JSON 含 50 条，性能 JSON 含原始模型用例 198、199。

## 执行

使用当前构建和安装的 wheel，加载 CANN 环境并选择空闲卡。
只暴露一张物理卡时，ATK 的设备号为 0。

本轮使用 ATK 原生 `-sp`，避免多进程文件消息传输的发布竞态。各项执行完整对应 JSON，
不设置 case 范围。正式包执行前三项；安装 sanitizer 包后执行内存检测，结束后使用正式包。

```bash
bash tests/atk/chunk_kda_bwd_prepare_a3/scripts/run_native.sh accuracy
bash tests/atk/chunk_kda_bwd_prepare_a3/scripts/run_native.sh performance
bash tests/atk/chunk_kda_bwd_prepare_a3/scripts/run_native.sh determinism
# 编译并安装 --sanitizer 包后执行：
bash tests/atk/chunk_kda_bwd_prepare_a3/scripts/run_native.sh mssanitizer
```

正式精度执行 `accuracy` 和 CPU benchmark；性能执行 `performance_device` 并保存 profile。
性能只读取 msprof `op_summary` 中 ChunkKdaBwdPrepare 的 `Task Duration(us)`，不采用 Python wall time。
内存检测必须使用 `--sanitizer` 编译并安装的包；性能结论使用无 debug/sanitizer 的正式包。
可分 scope 执行，但每项均执行完整对应 JSON。
sanitizer 的 `--log-file` 与 ATK `-msl` 使用同一文件，保证报告读取实际检测日志。
内存检测仅观察待测算子，不额外启动有限值检测 kernel；精度和确定性保留有限值检查。
确定性默认使用 8 个 CPU 线程加速 ATK 原生大张量位比较，可通过
`KDA_PREPARE_ATK_CPU_THREADS` 配置；每条仍执行 100 次，比较逻辑不变。

## 覆盖

| key | 模板条件 | 基础用例 | 边界用例 | MSS/DC 用例 | SoC | 选择证据 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | BNSD / fixed | 2 | 0、4、6 | 0..24 中偶数、198 | A2/A3 | host tiling UT 实选 0 |
| 1 | BSND / fixed | 3 | 3、5、7 | 0..24 中奇数、199 | A2/A3 | host tiling UT 实选 1 |
| 6 | NTD / variable | 100 | 98、100、116 | 100..122 中偶数 | A2/A3 | host tiling UT 实选 6 |
| 7 | TND / variable | 101 | 99、101、117 | 100..122 中奇数 | A2/A3 | host tiling UT 实选 7 |

每个布局均覆盖两种状态方向；T=1/63/64/65、非整块、多个 batch/head、空序列、
多个 chunk、scale=0/负数均在基础矩阵中。补充边界脚本覆盖 32 对齐边界、大行距及 record 跨任务复用。

Host UT 覆盖每个 key 的两个 `state_v_first` 分支、block/workspace 计算及非法属性拒绝：

```bash
g++ -std=c++17 -Icommon/include -I"$ASCEND_HOME_PATH/include" \
  -I"$ASCEND_HOME_PATH/include/op_common" \
  fla/ops/ascendc/kda/chunk_kda_bwd_prepare/op_host/tests/tiling_contract_test.cpp \
  -o build/tiling_contract_test
./build/tiling_contract_test
```

布局对比补充用例为 packed TND case 123 与用户 BSND 模型 case 199；输入、seed、scale、
state 均不改变。`scripts/compare_layouts.py --output-dir <output>` 生成 `direct.json` 和
`transpose.json`。前者使用原 executor，后者使用 `scripts/executor_transpose.py`；后者在
每次调用内将三份物理输入显式转换为 head-major，再调用同一算子。两份 JSON 均可通过 ATK
`performance_device --save_data profile -sp` 采集，基线取三次实际转换与算子 task duration 之和。
基线 CPU 精度对齐继续使用原 FP64 标杆。

## 验收记录

本轮从 main 的 `90176b44` 实现并验证。A2 正式 wheel、A3 run 包及 sanitizer run 包均构建通过；
A3 包单独保留。本轮设备验证采用 A2，A3 尚未进行实机验证。

开发参考采用仓内 `CHUNK_INDEPENDENT V1`，提炼来源 commit
`a78efa9512dfa64c20b2a58a75e84e88a054d694`。复用独立 task 跨步调度和同步闭环的工程组织；
四 head 窗口、四 record、双 slot、资源常量及 A3 GM 中转由本算子设计确定。

| 测试项 | 结果 |
| --- | --- |
| Host tiling contract | 4 个模板 key × 2 种状态方向及非法属性检查通过 |
| Stage 0 中间结果 | 50/50 通过，A/Q/D 使用可追踪整数输入逐路精确比较 |
| 补充边界 | 106/106 通过，覆盖尾块、长序列复用及大行距 fallback |
| 非法输入拒绝 | 20/20 通过 |
| 原生混合容差精度 | 600/600 通过，执行失败 0、精度失败 0 |
| 确定性 | 50/50 通过，每条执行 100 次，输出逐位一致 |
| 内存检测 | 50/50 通过，实际 memcheck 日志无 ERROR，覆盖 key 0/1/6/7 |
| 补充边界 memcheck | 106/106 通过，包含 H=512/1024 大行距回退路径，实际日志无 ERROR |
| 布局转换基线精度 | TND case 123、BSND case 199 均通过 |
| 模型性能采集 | 原始 case 198、199 均成功生成 msprof CSV |
| 最终安装验证 | wheel 中的 wrapper、OPP、opapi、kernel 校验一致，公开入口调用通过 |

内存检测日志有 `FFTS_BASE_ADDR` 未复位警告，保留原始日志；ATK 原生内存检查结果为 Pass。
不将警告描述为已消除。

全量运行中 case 534 的额外 GM 初始化遇到显存资源竞争，随后算子执行及精度比较均通过。
该用例已单独重新执行 GM 初始化与原生精度比较，均成功，复测保存输出且无 ERROR。
全程保留原始输入分布、种子和精度标准。

模型性能取一次采集的第一个算子 task，并给出该次 ATK 原生 20 个 profile task 的范围：

| case_id | 输入布局 / T / H | 首个 Task Duration(us) | 同次采集范围(us) |
| --- | --- | --- | --- |
| 198 | BNSD / 8192 / 96 | 1502.860 | 1502.860..1512.800 |
| 199 | BSND / 16384 / 96 | 3263.360 | 3259.120..3270.120 |

布局对比在相同设备、相同输入与属性下分别采集一次，按 task 顺序读取首组数据。
基线为三次 Transpose 与 head-major 算子的 `Task Duration(us)` 之和；
降低比例为 `(基线 - 直接读取) / 基线`。这些数值说明本轮采集结果，不声明通用性能目标达标。

| case_id | 布局 | 直接读取(us) | 转换后调用(us) | task 耗时降低 |
| --- | --- | --- | --- | --- |
| 123 | TND | 5.180 | 41.361 | 87.48% |
| 199 | BSND | 3268.925 | 5241.245 | 37.63% |

最终 wheel 的 kernel、opapi、proto 和 Python native runtime 与正式验收包逐字节一致。
补回原有版权注释后，tiling 库仅有五处日志行号常量各增加 4 及 ELF build-id 变化，
计算和分发逻辑不变。最终安装使用无 sanitizer、无 debug 编译参数的正式包。
