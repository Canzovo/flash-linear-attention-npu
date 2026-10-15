# ChunkKdaBwdPrepare

BF16、C=64、K=V=128 的 KDA backward prepare 算子，支持 A2/A3。
输入支持 BNSD、BSND、NTD、TND，三路输出固定为 head-major：FP32 dAqk、BF16 dv、FP32 dq_raw。

- [公开接口及数学定义](docs/api.md)
- [A3 完整设计](docs/design_a3.md)
- [当前实现及资源](docs/design.md)
- [ATK 用例及验收结果](../../../../../tests/atk/chunk_kda_bwd_prepare_a3/README.md)

arch22 实现采用 AIC Cube 与 AIV Vector 的 GM 中转；arch35 源码保留供后续 A5 适配，
本轮不注册 Ascend950。统一 ACLNN/L0 接口使用显式 layout 参数。
