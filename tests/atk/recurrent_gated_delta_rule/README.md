# RecurrentGDN A2/A3 ATK 交付件

本工程对齐本任务的 A2/A3 V2 设计和 2026-10-30 CPU 双标杆。测试通过公开入口 `fla_npu.ops.ascendc.recurrent_gated_delta_rule` 调用 Stable ABI；每次 NPU 正向执行后强制核验 `ascendc.BACKENDS['npu_recurrent_gated_delta_rule']=='stable'`，缺失或回退到 ctypes 均报 ERROR。CPU 标杆独立计算，不调用 DUT 或 kernel 辅助函数。本文说明测试资产和复现方法，硬件通过结论以本轮实际报告为准。

## 接口与输入

- Q/K 为 `[T,HK,128]` BF16，V/out 为 `[T,HV,128]` BF16；`HV%HK=0`，ratio 为 1～16，head 总数不限制为 256。
- State 为 `[Cstate,HV,128,128]` BF16/FP32，原地更新；比较完整 State pool，包括未写槽。State padding 和 storage offset 前后的 guard 另外检查不变。
- Beta 和 scalar g 分别独立支持 BF16/FP32。g 可空；gk 保留但非空应拒绝。raw gate 模式要求 g/A_log，dt_bias 可空；A_log/dt_bias 为 `[HV]` FP32。
- Q/K L2Norm、raw gate 和 Beta sigmoid 三个开关独立生效；`allow_neg_eigval` 仅在 sigmoid 开启时将结果乘 2。归一化后的 Q/K 保持 FP32，不重新量化到 BF16。
- lengths 为 INT32 `[prefix,L0,...]`，总和为 T；checkpoint 为 `[T]` 或 `[B,W]`。二维要求 `W>0` 且 `W>=max(L_b)`，全空请求也不接受零宽表，与原 CPU 标杆一致。二维 accepted 可大于当前长度且不超过 W；空请求跳过 accepted。同请求重复槽合法，不同请求不得发生写/写或初始读/写竞争。
- QKV 保留末轴 stride=1 的外轴 padding、transpose、混合 QKV slice、offset 和合法只读重叠。State 保留行/head/slot padding、置换和 offset。末轴 stride=2 的 QKV 是反向用例，不冒充合法布局。
- out 的无效 prefix 在测试比较前统一置零；kernel 不因此承担 prefix 清零要求。State 没有无效区屏蔽。

三条执行路径均先在 CPU 按相同 seed 生成产品 dtype 的输入；迁移 DUT 时按完整 backing storage 搬运，再以原 shape/stride/offset 重建视图，保留 QKV 只读别名。golden 不把产品输入改成 FP64；仅递推内部使用 FP64。

## 原生 CPU 双标杆

`reference.py` 原样来自用户提供的 2026-10-30 标杆，提供固定 FP64 golden 和 FP32 bench。两条路径均在 checkpoint 写回时转换到公开 dtype，下一 token 接续未舍入的工作 State。`reference_recurrent_gated_delta_rule.py` 只做兼容导入。

精度唯一标准为 YAML/JSON 中 ATK 原生 `cv_fused_double_benchmark`：`max_re_ratio=5`、`avg_re_ratio=1.5`、`root_mean_squared_ratio=1.5`。executor 不自定义误差阈值，也不以混合容差或单标杆替代双标杆。

ATK 原生标准还保留未显式覆盖的默认小值域判据。已安装版本的官方类和边界探针确认：FP32 输出的小值域为 `abs(golden)<=2^-14`，其中 `abs(actual-golden)>2^-30` 的元素计为错误点；错误点比例为 `countActual/max(countBench,1)`，默认 `number_count_radio=2`，等于2通过，大于2失败。官方 `NumberCountResult.do_summary` 会将 benchmark 的零计数改成1，因此报告中的 Benchmark 计数1不能区分原始0或1。以上规则未作修改。

ATK 26.9.24 的纯 CPU 探针已确认：两个显式节点配合 `--bm_device cpu` 自动形成三次计算。普通 CPU invocation 的 `task_result.is_benchmark_task=False`，执行 FP32 bench；ATK 自动追加的 CPU invocation 为 True，执行 FP64 golden。不能只按 device 或节点名决定精度。探针位于 `scripts/probe_atk_dual_roles.py`，不调用实际算子。

非对称常量探针进一步使用 NPU 节点（只初始化 marker，执行器返回 CPU 常量，不调用 DUT）：golden=1、bench=1.01、DUT=2 时原生比较 FAIL，Actual/Benchmark 误差约为 1/0.01，误差比约100；golden=1、bench=2、DUT=1.01 时 PASS，误差比约0.01。日志确认使用 `CvFusedDoubleBenchmarkAccuracyCompare`。ATK 将结果记在第二个 CPU 节点行，但 `Actual` 对应第一个 NPU DUT，不能只根据行名交换节点。探针公开输出使用 FP32，避免 FP64 输出触发不同比较路径。

```bash
python scripts/sample_accuracy_cases.py --source atk_recurrent_gated_delta_rule.json \
  --output atk_output/accuracy/atk_recurrent_gated_delta_rule_random20.json
atk node --name npu_dut --backend npu --devices 0 --is_bm False \
  node --name cpu_bench --backend cpu --is_bm True \
  task -c ./atk_output/accuracy/atk_recurrent_gated_delta_rule_random20.json \
    -p ./executor_recurrent_gated_delta_rule.py \
    --task accuracy --bm_device cpu -to 2000
```

每轮精度按当前任务要求随机执行 **20 条**，不再默认执行全部 906 条。`bash scripts/run_dual_accuracy.sh 0` 和统一入口的 `-scope=accuracy` 均采用该默认值；其余 ATK 参数可追加。抽样不放回，保存 sampling seed、原始 case ID、候选池和子集 SHA256；不改变任何 case 内容、输入 seed、值域或精度标准。`ATK_SAMPLE_SEED=<seed>` 可复现抽样，`ATK_CASE_FILE=<saved.json>` 可重跑已保存子集。性能始终执行固定前 10 条（case 0～9），不随机抽样。每次源码修改后的默认回归为随机20条精度、固定10条性能和1条内存检测，取消默认确定性测试。内存检测固定使用 MSS case 7（B=1、T=9、HK=2、HV=4、K=V=128，State/Beta/g 为FP32，三项融合开启）。发生失败时先加 `--save_data output` 保存失败用例并做 CT 定位；不改阈值、range、分布、shape 或覆盖来制造通过。

入口在 ATK 命令结束后调用仓内公共结果检查器，读取本轮新生成的 xlsx 精度通过字段。任意精度失败、执行 ERROR、缺少报告或读到旧报告均返回非零；ATK 自身 CLI 退出0不代表精度通过。

复检使用 `ATK_TASK=accuracy_lt bash scripts/run_dual_accuracy.sh 0 -wl '[63,399]' --loop_nums 50 --disable_id_seed -mt 8`，不加 `-sp`，并以 `ct dual analyze <report.xlsx>` 汇总。仅在 `disable_id_seed` 且 `loop_nums>1` 时，executor 将 ATK 每轮运行时 `case_config.id` 混入数值 seed；三角色共享该 id。shape、indices、accepted、layout、stride、offset、属性和数值范围保持冻结；正常 accuracy 与确定性继续使用原 seed。不能使用跨轮保持不变的 `torch.initial_seed()` 代替该运行时 id。

两轮原生 `accuracy_lt` 探针已核验六次执行：三角色在同一轮的输入数值 hash 相同，下一轮数值 hash 改变；六次 metadata/stride/offset/属性 hash 完全相同。CPU 自检另对二维 checkpoint、accepted 大于有效长度、空请求和重复槽执行了相同断言。

CT0.9.1 的 `dual analyze` 官方聚合仅覆盖 MARE/MERE/RMSE 三个 ratio，不定义小值域错误点数量比的聚合判定。复检记录必须同时保留每轮原生通过状态、小值域 Actual/Benchmark 计数和比例分布，不能用三 ratio 聚合结果覆盖其未覆盖的指标。CT CLI 不传播 `process_excel_and_check` 的失败数，聚合状态应读取该官方 API 的 `failed_cases`；现有 accuracy 报告检查脚本仅检查 ATK 原生汇总，不能替代 CT 统计。

## 覆盖与映射

以下 906 条为候选池和覆盖映射；当前每轮精度从中随机抽取 20 条，抽样清单随报告保存。历史全矩阵报告保留原始结果。

精度矩阵共 **906** 条，每个逻辑 profile 包含固定种子 20261022、20261023、20261024。以历史 V2 834 条矩阵为基础：822 条符合新契约的 profile 原样保留 head、layout 和数值范围；另 12 条历史 ratio>16 的输入原 shape 迁入反向矩阵，未将 head 夹到 16。新增 84 条验证本次 Beta/g dtype、窗口和 checkpoint 契约。历史 gk/both 标签仅作来源追溯，当前正向计算统一使用设计要求的 scalar gate；gk 非空有独立反向用例。

| 精度 case id | 覆盖 |
| --- | --- |
| 0～587 | 历史 P/G 合法 shape、GQA、prefix、accepted、State/input view、稀疏索引和连续调用；head 数保持原值 |
| 588～821 | V2 的四 bool 16 组合、BF16/FP32 State、L=1/3/4/8/9/17/33、二维表、无 g、全空请求、布局、raw 极值、softplus 尾部、零 QK、257/258 heads |
| 822～845 | State/Beta/g 的 8 个独立 dtype 组合，三个融合开关与 allow_neg_eigval 开启 |
| 846～857、876～887 | BF16/FP32 State 的 L=7/16/32/65 窗口与尾部 |
| 858～860、888～890 | 混合 QKV backing storage 的 slice/offset 与 prefix/变长 |
| 861～866、891～896 | 一维重复写槽；二维 accepted>L、空请求与重复槽 |
| 867～875、897～905 | ratio=16、复合 padding/offset 和 State slot/head 置换 |

`python gen_recurrent_gated_delta_rule.py --summary` 输出逐条 design_id、shape 和布局，作为完整映射。

原值域保持：普通 QKV 标准差 0.05、State 标准差 0.01；Beta 默认 [0.1,0.9]；已处理 g 默认 -[0.001,0.02]，另有已固化 near_zero/strong_decay/per_head 分支。raw g 标准差 2，raw Beta 标准差 3，A_log/dt_bias 标准差 0.5/0.2；独立极值覆盖 ±100 和 A_log=12～25/g=-A_log。未因 NPU 输出调整输入范围。

MSS 人工候选池为 **24** 条，当前默认只执行 case 7：0～7 一一对应 State/Beta/g 的 8 组模板 dtype；8～9 为 L=65；10～11 为混合 QKV；12～13 为二维 accepted/重复槽；14～17 为 offset/padding/置换；18～19 为空请求；20～21 为 softplus 尾部；22～23 为 257/258 heads。实际 TilingKey 选择仍需从本轮 kernel 记录核验。

反向矩阵 **57** 条：0～11 保留历史越界 head ratio 原 shape；12～56 覆盖 gk、QKV 尾轴 stride、State 重叠/输入别名、跨请求竞争、slot、accepted、bool、scale、raw gate 依赖和 dtype。以 `--task run` 执行独立 negative executor，要求实际 wrapper 抛出 ValueError/TypeError，且 State 数值和版本未改变。JSON 显式配置 `expected_return_code=0`，表示反向验证 executor 完成预期拒绝和 State 不变检查后的成功返回码；导入、ACL 等运行时错误不会被捕获或当作通过。

性能候选矩阵共 54 条，按 BS、HV、TOKEN_PER_BATCH 的列出顺序排列；当前默认只执行前 **10** 条（case 0～9）。候选范围为 `BS={32,64,72}`、`HK=16`、`HV={32,48,128}`、`TOKEN_PER_BATCH={1,2,4,8,16,32}` 和 `K=V=128`。`T=BS×TOKEN_PER_BATCH`，`block_num=T`，`state_indices=list(range(T))`，每个 token 使用独立 State slot，State shape 为 `[T,HV,V,K]`。固定性能用例统一使用 QKV bf16、State bf16、Beta bf16、g fp32，并开启 Q/K L2 norm、gate、beta sigmoid 融合路径。模型 case 和正式性能结论仍需按真实调用条件单独核验。共同接口可直接比较 V1/V2；新融合能力的 V1 基线应是等价的显式预处理调用链，不可规约输入后宣称同条件对比。性能只使用 msprof `op_summary` 中目标算子的 `Task Duration(us)`，各 pipe 耗时不相加。此前 `block_num=BS`、batch 内重复 slot 的 54 条结果仅作历史记录，不代表本矩阵。

性能 JSON 使用 `executor_recurrent_gated_delta_rule_perf` 入口，仍由同一个 executor 文件注册。该入口按固定规格缓存输入，每次调用前恢复初始 State，只返回 attention 输出，避免重复构造和导出完整 State；精度入口继续比较完整输出与 State。前 10 条用例的规格、seed、shape 和 range 与原矩阵相同。性能入口要求 NPU 节点，执行时指定 `--bm_device npu`；可用下列命令采集单轮性能：

```bash
atk node --backend npu --devices 0 --is_bm False task \
  -c atk_recurrent_gated_delta_rule_perf.json -p executor_recurrent_gated_delta_rule.py \
  --task performance_device --bm_device npu --performance_data 1,1,1 \
  --save_data profile -sp -mt 1 -to 2000 --save_report_json
```

## 生成与执行

在本目录执行：

```bash
# ATK 实际生成全矩阵（YAML dtype_numbers=906）
atk case -f recurrent_gated_delta_rule.yaml -p gen_recurrent_gated_delta_rule.py -dt 906 -en 0 -s 20261030
python scripts/check_atk_generated_json.py result/recurrent_gated_delta_rule/json/all_recurrent_gated_delta_rule.json
cp result/recurrent_gated_delta_rule/json/all_recurrent_gated_delta_rule.json atk_recurrent_gated_delta_rule.json
# Python materializer 仅作辅助逐条核验；不替代原生生成的正式输入
python gen_recurrent_gated_delta_rule.py --output recurrent_gated_delta_rule_materialized.json
# 生成独立手工 MSS、性能和反向矩阵
python gen_recurrent_gated_delta_rule.py --kind mss
python gen_recurrent_gated_delta_rule.py --kind perf
python gen_recurrent_gated_delta_rule.py --kind negative
# CPU 契约和双标杆 smoke，无 NPU 调用
python scripts/check_atk_cpu_contract.py
python test_reference.py
# 精度双标杆
bash scripts/run_dual_accuracy.sh 0
# 每次修改后不执行确定性测试
atk node --backend npu --devices 0 task -c atk_recurrent_gated_delta_rule_negative.json \
  -p executor_recurrent_gated_delta_rule.py --task run -to 2000
```

每次源码修改后取消确定性测试；统一入口的 RecurrentGDN `all` 跳过该阶段。内存检测默认执行 MSS case 7 一条，使用 `run_test_cpu.sh -scope=mssanitizer`，默认白名单为 `[7]`；候选池保留，显式范围仅用于定位。性能使用统一 `-scope=performance` 和 `_perf.json`，保留 V1 与 V2 各自的原始 msprof CSV。

内存检测必须将外层 `mssanitizer --log-file` 和 ATK `-msl` 指向同一文件，并检查该日志非空。只看执行 SUCCESS 或读取空日志得出的 PASS 不能证明内存检测通过；还需核对目标 kernel 的真实诊断。正式编译包未添加 sanitizer 编译选项，本轮采用 GM 访问快速定界，不将它描述为 UB、未初始化或竞争检测全覆盖；检测能力范围见 [MSS 官方指南](https://github.com/Ascend/mssanitizer/blob/master/docs/zh/user_guide/mssanitizer_user_guide.md)。

使用官方 kernel 名过滤，当前默认检测 case 7 对应的目标 kernel。可在本目录复现：

```bash
MSS_LOG_PATH="$PWD/recurrent_gdn_mss.log"
mssanitizer --tool=memcheck --kernel-name=RecurrentGatedDeltaRule \
  --log-file="$MSS_LOG_PATH" -- \
  atk node --backend npu --devices 0 task \
    -c atk_recurrent_gated_delta_rule_mss.json \
    -p executor_recurrent_gated_delta_rule.py \
    --task run --mssanitizer -msl "$MSS_LOG_PATH" -wl '[7]' -sp -to 2000
test -s "$MSS_LOG_PATH"
```

行 padding 的 BF16/FP32 补充使用原完整 JSON 和 `-wl '[678,774]'`，另设独立日志文件；不改 shape、stride、offset 或输入范围。

建议首轮白名单：`[0,657,663,807,822,825,828,831,834,837,840,843,855,858,864,870,885,888,894,903]`。白名单只用于烟测和定位；当前每轮回归仍按随机20条精度执行。

本次 CPU 自检已验证906条正向结构、57条反向拦截、20个代表 case 的40次FP64/FP32执行、相同数值/stride/offset及ATK角色分流。这些结果不代表 NPU 精度、性能、确定性或内存检查通过。

原生 `atk case` 已实际生成906条 JSON；独立检查确认 id=0～905、每条 `case_spec`、默认 seed 和精度标准均与冻结矩阵一致。交付件中的 `atk_recurrent_gated_delta_rule.json` 使用这份原生输出作为精度候选池及历史完整验收输入，SHA256 为 `6fccb2d346cdb696c90643647739ec6d65fed5df44acbe2f0bb000873db547b6`。Python materializer 只作辅助核验，负测/MSS/性能继续采用各自人工矩阵。

原生生成 JSON 只包含两个执行设备 marker 和一个完整 `case_spec`，ATK 的静态张量形状统计只描述 marker；真实算子 shape、layout 和 dtype 以 `case_spec` 与矩阵映射为准。

## 历史 A2 验收记录

以下为窗口向量化及本次用例修正前的历史结果，不能替代当前 V2 的重新验证。尤其旧 54 条性能用例使用了 batch 内重复 State slot，现已作废；当前固定矩阵采用上一节规定的 `block_num=T` 与完整索引。

单算子编译、安装和包内/安装后 kernel 身份校验通过，Stable 入口命中已核验。原生 `atk case` 生成的 906 条完整矩阵已执行，首次精度 904 PASS、2 FAIL、0 ERROR；case_id 63、399 **需要复检**。保存输出和 CT 定位未发现结构异常后，两例各进行 50 轮复检，100/100 轮原生四项判据全部通过，官方 CT 三 ratio 聚合失败数为 0。按统计复检流程，精度验收通过；原单轮 FAIL 保留在配套交付件中，不描述为首次 906 PASS。

负向 57/57 通过且异常前 State 不变；最终 host 修复包完整确定性 906 条每例 50 次全部通过，旧包结果另存历史材料，不替代最终包重跑。内存/MSS 的正式编译包 GM 访问检测 24 条主矩阵及原 case_id 678、774 两条行 padding 用例通过，目标算子零非法诊断；历史 C ABI 16/16 执行契约检查通过。ATK 扩展性能矩阵 54/54 执行成功，单 case 重跑后 54 条均具备可用 Task Duration。相同输入的 V1/V2 共同 6 场景均出现回退，V2 耗时为 V1 的 2.748～5.811 倍，未取得加速；正式耗时来源为 msprof `Task Duration(us)`。

复检未调整输入范围、阈值、shape、布局或覆盖；CT 官方三项聚合与原生第四项分别核验。完整报告、原始 profiler CSV、结果工作簿与 SHA256 清单位于配套测试交付件。本轮仅完成 A2 真机验证，不代表 A3 真机或 wheel 安装验证完成。

## 2026-11-02 用例与窗口预处理修正（repeat 优化前）

本节为后述 repeat 优化前的历史测试结果；不代替当前版本的随机20条验证。

固定性能矩阵已修正为 `block_num=T`、`state_indices=list(range(T))`，54 条 shape、顺序和 seed 保持固定。V2 的 UB 区域采用具名尺寸与偏移推导；短窗口 beta sigmoid 改为不依赖压缩 mask 的向量公式，softplus 增加向量舍入残差校正。V1 接口和计算实现未修改。

当前正式包随机 20 条原生双标杆精度 20/20 通过。完整 906 条为 898 PASS、8 FAIL、0 ERROR；未通过的 `case_id=63,275,399,574,722,725,734,802` 均为 FP32 State 小值域计数问题。保存输出、CT 对比后各做 50 轮数值复检：368/400 轮原生 PASS，32 轮仍未通过小值域计数判据；官方 CT 三 ratio 聚合失败数为 0，但该聚合不覆盖小值域计数。完整精度尚未达到全量原生通过，原始失败和每轮判据均保留。

正式编译包 GM 内存检测 24/24 通过，核对到 24 次目标 kernel 的完整开始/结束记录，0 非法访问、0 目标告警；该结果不代表 UB、未初始化或竞争检测全覆盖。随机 20 条确定性每条 50 次全部通过，0 ERROR。

历史 ACLNN 的内部启动列表补齐共享 kernel 新增的两个空输入槽，公开签名和 V1 计算实现保持原样；旧 C ABI 16 条执行兼容检查全部通过。该适配后的包再次执行随机 20 条原生双标杆精度，20 PASS、0 FAIL、0 ERROR。适配前后的 4 个 kernel 二进制 SHA256 全部相同，V2 L0 路径未修改。

共同接口的 6 条固定场景为 `T=1/4/8`、`HK=2`、`HV=8`、`K=V=128`、State BF16/FP32，融合属性关闭。输入 SHA256、每次调用前的 State 重置、5 次预热与 1 次正式采样一致，所有 pipe 字段完整；V2 耗时为 V1 的 2.664～5.701 倍，存在性能回退。该对比仅覆盖共同能力，不代表上述 54 条融合场景的等价 V1 基线。后续范围已调整为固定前10条，54条扩展采集已停止；对应原始数据保留为历史。

## 2026-11-02 repeat 批处理优化（历史版本）

V2 两次点积改用 repeat=128 批量处理128行，复用回收后的旧 State bank 作为完整乘积 scratch；两次点积合计8次向量 API 调用、6次 V barrier。BF16 路径整块衰减一次供 prediction 和更新共享。UB总量仍为162240 B，V1公共接口和计算头文件未修改。

当前单算子编译、安装、头文件/二进制身份校验通过。随机20条原生双标杆精度20 PASS、0 FAIL、0 ERROR；sampling seed=7090966332440272974，原始case ID和输入SHA随报告保存。精度候选池、输入范围和原生阈值保持原样。本轮仅按要求执行随机20条，未重跑906条。

性能固定执行case 0～9，使用相同输入JSON、每次调用前恢复初始State，耗时来自msprof目标算子的Task Duration(us)。优化前10条和优化后10条分别归档，结果以配套交付件为准。

本轮固定10条性能全部执行成功，目标算子相对 repeat 优化前加速 3.730～5.243 倍；两轮输入 JSON SHA256 相同，原始 CSV 与逐例对比随交付件保存。当前按任务要求取消确定性；内存检测仅执行 MSS case 7。

## V1/V2固定10条性能复现（外积修改前）

固定10条中8条有有效V1基线，V2相对V1加速1.168～1.209倍。V1的MAX_MTP=8，TPB=16/32会直接返回，约9us的空返回记录不作为计算性能，逐例标记为不支持。逐例耗时与原始CSV见配套交付件 comparison_fixed10_v1_v2/。两版使用相同CPU预处理结果，Q/K与beta按V1接口统一转为BF16、g转为已处理的FP32 log gate，再关闭融合。每例5次预热、1次正式采样，每次调用前恢复初始State。比较不包含CPU预处理耗时，不等同于FP32归一化的V2融合路径，也不作为端到端预处理调用链性能。

```bash
for version in v1 v2; do
  msprof --output="profile_fixed10/$version" python scripts/profile_v1_v2_fixed10.py \
    --version "$version" --cases atk_recurrent_gated_delta_rule_perf.json \
    --manifest "profile_fixed10/$version/case_manifest.json"
done
```

脚本核验固定case 0～9，保存逐输入SHA256。目标CSV中每例6次调用，按时间排序后取第6次，其余5次为预热；只读取目标算子的Task Duration(us)。历史6条小shape对比保留为优化前结果，不代替本次10条。

## 2026-11-02 外积统一使用 MulAddDst

按本轮要求，BF16和FP32 State的外积更新均使用MulAddDst：K=128分为两个64元素半行，每次repeat=128批量更新128行。
已删除FP32外积补偿函数、调用分支、乘积拆分/误差恢复、有限值选择、tile scratch常量及断言；外积不采用TwoProd/TwoSum算法。
原有窗口预处理、门控计算、整块一次衰减、批量点积和V1接口保持原样。此处变更仅涉及外积，raw gate的数值预处理仍按设计执行。
每次源码修改后执行随机20条原生双标杆精度、固定前10条性能和1条GM内存检测，不执行确定性。本轮单算子编译、安装和头文件/二进制身份校验通过。

本轮随机20条精度为19 PASS、1 FAIL、0 ERROR，sampling seed=14451288798797891062。case 492的State输出小值域数错误差比例为3，超过原生阈值2，需要复检；保存输出的CT图贴近对角，最大绝对误差7.45e-9。固定结构仅随机数值复检50轮，49 PASS、1 FAIL、0 ERROR，仍有小值域计数未达标。CT三项ratio聚合失败数为0，但该聚合不定义小值域判据，不能替代原生第四项。本轮精度不声明全部通过，未修改输入范围或精度阈值。

固定前10条融合性能10/10执行成功，目标Task Duration为142.123～2935.699 us；本轮和修改前均为BF16 State，逐条耗时比（修改前/修改后）为0.988～1.010，不能据此声称FP32收益。1条GM内存检测通过，目标kernel完整开始/结束各1次，0错误、0告警。原始CSV、随机样本及原生报告的数值保真导出随本轮交付件保存。

当前V1/V2共同能力对比已完成，两版逐输入SHA256一致，8条具有有效V1基线，TPB=16/32仍标为不支持。case 6/7本轮V1耗时分别为1201.524/687.614 us，明显高于前轮204.804/348.407 us；对应7.073/2.331倍仅作单轮观测，不作为稳定性能收益。其余6条本轮加速1.167～1.329倍。原始逐例数据全部保留，未删去异常点或追加采样改变结果。
