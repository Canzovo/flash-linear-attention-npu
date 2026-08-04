#include "register/op_def_registry.h"

namespace ops {
class ChunkKdaFwdFusedA5 : public OpDef {
public:
    explicit ChunkKdaFwdFusedA5(const char *name) : OpDef(name)
    {
        const std::initializer_list<ge::DataType> dataTypes = {
            ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
            ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16
        };
        const std::initializer_list<ge::DataType> gateTypes = {
            ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
            ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT
        };
        const std::initializer_list<ge::DataType> rawGateTypes = {
            ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_BF16, ge::DT_BF16,
            ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_BF16, ge::DT_BF16
        };
        const std::initializer_list<ge::DataType> betaTypes = {
            ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT, ge::DT_BF16,
            ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT, ge::DT_BF16
        };
        const std::initializer_list<ge::DataType> stateTypes = {
            ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
            ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT
        };
        const std::initializer_list<ge::DataType> intTypes = {
            ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
            ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64
        };
        const std::initializer_list<ge::Format> formats = {
            ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
            ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND
        };

        this->Input("q").ParamType(REQUIRED).DataType(dataTypes).Format(formats).UnknownShapeFormat(formats);
        this->Input("k").ParamType(REQUIRED).DataType(dataTypes).Format(formats).UnknownShapeFormat(formats);
        this->Input("v").ParamType(REQUIRED).DataType(dataTypes).Format(formats).UnknownShapeFormat(formats);
        this->Input("gk").ParamType(REQUIRED).DataType(gateTypes).Format(formats).UnknownShapeFormat(formats);
        this->Input("raw_g").ParamType(REQUIRED).DataType(rawGateTypes).Format(formats).UnknownShapeFormat(formats);
        this->Input("a_log").ParamType(OPTIONAL).DataType(stateTypes).Format(formats).UnknownShapeFormat(formats);
        this->Input("dt_bias").ParamType(OPTIONAL).DataType(stateTypes).Format(formats).UnknownShapeFormat(formats);
        this->Input("beta").ParamType(REQUIRED).DataType(betaTypes).Format(formats).UnknownShapeFormat(formats);
        this->Input("initial_state").ParamType(OPTIONAL).DataType(stateTypes).Format(formats).UnknownShapeFormat(formats);
        this->Input("cu_seqlens").ParamType(OPTIONAL).ValueDepend(OPTIONAL)
            .DataType(intTypes)
            .Format(formats).UnknownShapeFormat(formats);
        this->Input("chunk_indices").ParamType(OPTIONAL).ValueDepend(OPTIONAL)
            .DataType(intTypes)
            .Format(formats).UnknownShapeFormat(formats);

        this->Output("attn_out").ParamType(REQUIRED).DataType(dataTypes).Format(formats).UnknownShapeFormat(formats);
        this->Output("final_state").ParamType(REQUIRED).DataType(stateTypes).Format(formats).UnknownShapeFormat(formats);
        this->Output("Aqk").ParamType(REQUIRED).DataType(dataTypes).Format(formats).UnknownShapeFormat(formats);
        this->Output("Akk").ParamType(REQUIRED).DataType(dataTypes).Format(formats).UnknownShapeFormat(formats);
        this->Output("w").ParamType(REQUIRED).DataType(dataTypes).Format(formats).UnknownShapeFormat(formats);
        this->Output("u").ParamType(REQUIRED).DataType(dataTypes).Format(formats).UnknownShapeFormat(formats);
        this->Output("qg").ParamType(REQUIRED).DataType(dataTypes).Format(formats).UnknownShapeFormat(formats);
        this->Output("kg").ParamType(REQUIRED).DataType(dataTypes).Format(formats).UnknownShapeFormat(formats);
        this->Output("v_new").ParamType(REQUIRED).DataType(dataTypes).Format(formats).UnknownShapeFormat(formats);
        this->Output("h").ParamType(REQUIRED).DataType(dataTypes).Format(formats).UnknownShapeFormat(formats);

        this->Attr("scale").AttrType(REQUIRED).Float(1.0);
        this->Attr("chunk_size").AttrType(REQUIRED).Int(64);
        this->Attr("safe_gate").AttrType(REQUIRED).Bool(false);
        this->Attr("logical_batch").AttrType(REQUIRED).Int(1);
        this->Attr("logical_seqlen").AttrType(REQUIRED).Int(1);
        this->Attr("logical_q_heads").AttrType(REQUIRED).Int(1);
        this->Attr("logical_v_heads").AttrType(REQUIRED).Int(1);
        this->Attr("logical_k_dim").AttrType(REQUIRED).Int(1);
        this->Attr("logical_v_dim").AttrType(REQUIRED).Int(1);
        this->Attr("logical_total_chunks").AttrType(REQUIRED).Int(1);
        this->Attr("use_gate_in_kernel").AttrType(REQUIRED).Bool(false);
        this->Attr("lower_bound").AttrType(REQUIRED).Float(-5.0);
        this->Attr("defer_gate_cumsum").AttrType(REQUIRED).Bool(false);
        this->Attr("output_final_state").AttrType(REQUIRED).Bool(false);
        this->Attr("store_qg").AttrType(REQUIRED).Bool(false);
        this->Attr("store_v_new").AttrType(REQUIRED).Bool(false);
        this->Attr("store_h").AttrType(REQUIRED).Bool(false);

        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("prebuildPattern.value", "Opaque")
            .ExtendCfgInfo("coreType.value", "AiCore")
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
        this->AICore().AddConfig("ascend950", config);
    }
};

OP_ADD(ChunkKdaFwdFusedA5);
} // namespace ops
