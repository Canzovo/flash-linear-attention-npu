#include "chunk_kda_fwd_fused_a5_tiling.h"

#include <register/op_impl_registry.h>
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {
constexpr size_t INPUT_Q_IDX = 0;
constexpr size_t INPUT_GK_IDX = 3;
constexpr size_t INPUT_RAW_G_IDX = 4;
constexpr size_t INPUT_A_LOG_IDX = 5;
constexpr size_t INPUT_DT_BIAS_IDX = 6;
constexpr size_t INPUT_INITIAL_STATE_IDX = 8;
constexpr size_t INPUT_CU_SEQLENS_IDX = 9;
constexpr size_t INPUT_CHUNK_INDICES_IDX = 10;
constexpr size_t ATTR_SCALE_IDX = 0;
constexpr size_t ATTR_CHUNK_SIZE_IDX = 1;
constexpr size_t ATTR_SAFE_GATE_IDX = 2;
constexpr size_t ATTR_LOGICAL_BATCH_IDX = 3;
constexpr size_t ATTR_LOGICAL_SEQLEN_IDX = 4;
constexpr size_t ATTR_LOGICAL_Q_HEADS_IDX = 5;
constexpr size_t ATTR_LOGICAL_V_HEADS_IDX = 6;
constexpr size_t ATTR_LOGICAL_K_DIM_IDX = 7;
constexpr size_t ATTR_LOGICAL_V_DIM_IDX = 8;
constexpr size_t ATTR_LOGICAL_TOTAL_CHUNKS_IDX = 9;
constexpr size_t ATTR_USE_GATE_IDX = 10;
constexpr size_t ATTR_LOWER_BOUND_IDX = 11;
constexpr size_t ATTR_DEFER_GATE_CUMSUM_IDX = 12;
constexpr size_t ATTR_OUTPUT_FINAL_STATE_IDX = 13;
constexpr size_t ATTR_STORE_QG_IDX = 14;
constexpr size_t ATTR_STORE_V_NEW_IDX = 15;
constexpr size_t ATTR_STORE_H_IDX = 16;

constexpr uint64_t KDA_ALIGN = 512;
constexpr uint64_t KDA_SOLVE_SCRATCH_SLOTS = 5;
constexpr uint64_t KDA_SOLVE_PIPELINE_DEPTH = 4;
constexpr uint64_t KDA_SCORE_QUEUE_SLOTS = 4;
constexpr uint64_t KDA_SCORE_SCRATCH_PLANES = 3;
constexpr uint64_t KDA_GDN_PIPELINE_DEPTH = 2;
constexpr uint32_t KDA_BATCH_MODE = 1;

uint64_t AlignWorkspace(uint64_t bytes)
{
    return (bytes + KDA_ALIGN - 1) / KDA_ALIGN * KDA_ALIGN;
}

bool ResolveSequenceInfo(gert::TilingContext *context, int64_t chunkSize, int64_t totalChunks,
                         int64_t batch, bool &isVarLen, int64_t &seqNum)
{
    isVarLen = context->GetOptionalInputTensor(INPUT_CU_SEQLENS_IDX) != nullptr;
    seqNum = batch;
    if (!isVarLen) {
        return true;
    }
    auto cuTensor = context->GetOptionalInputTensor(INPUT_CU_SEQLENS_IDX);
    auto chunkMetadata = context->GetOptionalInputTensor(INPUT_CHUNK_INDICES_IDX);
    seqNum = cuTensor->GetStorageShape().GetDim(0) - 1;
    if (seqNum <= 0 || chunkMetadata == nullptr ||
        chunkMetadata->GetStorageShape().GetShapeSize() != totalChunks * 4) {
        return false;
    }
    const int64_t *cu = cuTensor->GetData<int64_t>();
    if (cu == nullptr) {
        return false;
    }
    int64_t chunkCount = 0;
    for (int64_t seq = 0; seq < seqNum; ++seq) {
        if (cu[seq] < 0 || cu[seq + 1] < cu[seq]) {
            return false;
        }
        chunkCount += (cu[seq + 1] - cu[seq] + chunkSize - 1) / chunkSize;
    }
    return chunkCount == totalChunks;
}
} // namespace

ge::graphStatus Tiling4ChunkKdaFwdFusedA5(gert::TilingContext *context)
{
    const auto qDesc = context->GetInputDesc(INPUT_Q_IDX);
    const auto rawGDesc = context->GetInputDesc(INPUT_RAW_G_IDX);
    const auto gkDesc = context->GetInputDesc(INPUT_GK_IDX);
    const auto attrs = context->GetAttrs();
    const auto qShapePtr = context->GetInputShape(INPUT_Q_IDX);
    if (qDesc == nullptr || rawGDesc == nullptr || gkDesc == nullptr ||
        attrs == nullptr || qShapePtr == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const float scale = static_cast<float>(*(attrs->GetAttrPointer<double>(ATTR_SCALE_IDX)));
    const int64_t chunkSize = *(attrs->GetAttrPointer<int64_t>(ATTR_CHUNK_SIZE_IDX));
    const bool safeGate = *(attrs->GetAttrPointer<bool>(ATTR_SAFE_GATE_IDX));
    const int64_t batch = *(attrs->GetAttrPointer<int64_t>(ATTR_LOGICAL_BATCH_IDX));
    const int64_t seqlen = *(attrs->GetAttrPointer<int64_t>(ATTR_LOGICAL_SEQLEN_IDX));
    const int64_t qHeads = *(attrs->GetAttrPointer<int64_t>(ATTR_LOGICAL_Q_HEADS_IDX));
    const int64_t vHeads = *(attrs->GetAttrPointer<int64_t>(ATTR_LOGICAL_V_HEADS_IDX));
    const int64_t kDim = *(attrs->GetAttrPointer<int64_t>(ATTR_LOGICAL_K_DIM_IDX));
    const int64_t vDim = *(attrs->GetAttrPointer<int64_t>(ATTR_LOGICAL_V_DIM_IDX));
    const int64_t totalChunks = *(attrs->GetAttrPointer<int64_t>(ATTR_LOGICAL_TOTAL_CHUNKS_IDX));
    const bool useGateInKernel = *(attrs->GetAttrPointer<bool>(ATTR_USE_GATE_IDX));
    const float lowerBound = *(attrs->GetAttrPointer<float>(ATTR_LOWER_BOUND_IDX));
    const bool deferGateCumsum = *(attrs->GetAttrPointer<bool>(ATTR_DEFER_GATE_CUMSUM_IDX));
    const bool outputFinalState = *(attrs->GetAttrPointer<bool>(ATTR_OUTPUT_FINAL_STATE_IDX));
    const bool storeQG = *(attrs->GetAttrPointer<bool>(ATTR_STORE_QG_IDX));
    const bool storeVNew = *(attrs->GetAttrPointer<bool>(ATTR_STORE_V_NEW_IDX));
    const bool storeH = *(attrs->GetAttrPointer<bool>(ATTR_STORE_H_IDX));
    if (batch <= 0 || seqlen <= 0 || qHeads <= 0 || vHeads <= 0 || kDim <= 0 ||
        vDim <= 0 || totalChunks <= 0 || vHeads % qHeads != 0) {
        return ge::GRAPH_FAILED;
    }

    bool isVarLen = false;
    int64_t seqNum = 0;
    if (!ResolveSequenceInfo(context, chunkSize, totalChunks, batch, isVarLen, seqNum)) {
        return ge::GRAPH_FAILED;
    }
    const auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t blockDim = platform.GetCoreNumAic() == 0 ? 1 : platform.GetCoreNumAic();
    const bool hasALog = context->GetOptionalInputDesc(INPUT_A_LOG_IDX) != nullptr;
    const bool hasDtBias = context->GetOptionalInputDesc(INPUT_DT_BIAS_IDX) != nullptr;
    const bool hasInitialState = context->GetOptionalInputDesc(INPUT_INITIAL_STATE_IDX) != nullptr;
    const auto &qShape = qShapePtr->GetStorageShape();
    const bool inputSequenceMajor =
        qShape.GetDimNum() == 4 && qShape.GetDim(0) == batch &&
        qShape.GetDim(1) == seqlen && qShape.GetDim(2) == qHeads && qShape.GetDim(3) == kDim;
    const bool computeGateInPrepare =
        deferGateCumsum && platform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND950 &&
        qDesc->GetDataType() == ge::DT_BF16 && rawGDesc->GetDataType() == ge::DT_FLOAT &&
        gkDesc->GetDataType() == ge::DT_FLOAT && useGateInKernel && safeGate && hasALog &&
        !isVarLen && chunkSize == 64 && kDim == 128 && vDim == 128 &&
        vHeads % 2 == 0 && seqlen % chunkSize == 0;
    const bool fuseRecurrentPostWu =
        computeGateInPrepare && !storeQG && !storeVNew && !storeH;
    const bool fusePostWu = computeGateInPrepare && !fuseRecurrentPostWu;

    const uint64_t dataBytes = qDesc->GetDataType() == ge::DT_FLOAT ? sizeof(float) : sizeof(uint16_t);
    const uint64_t tokenHeads = static_cast<uint64_t>(batch) * vHeads * seqlen;
    const uint64_t matrixBytes = tokenHeads * chunkSize * sizeof(float);
    const uint64_t aqkFp32Offset = 0;
    const uint64_t akkFp32Offset = AlignWorkspace(aqkFp32Offset + matrixBytes);
    const uint64_t prepareScratchOffset = AlignWorkspace(akkFp32Offset + matrixBytes);
    const uint64_t solveDepth = safeGate ? KDA_SOLVE_PIPELINE_DEPTH : 1;
    const uint64_t solveBytes = static_cast<uint64_t>(blockDim) * solveDepth *
        KDA_SOLVE_SCRATCH_SLOTS * chunkSize * chunkSize * sizeof(float);
    const uint64_t scoreBytes = static_cast<uint64_t>(blockDim) * KDA_SCORE_QUEUE_SLOTS *
        KDA_SCORE_SCRATCH_PLANES * chunkSize * kDim * dataBytes;
    const uint64_t qgScaledOffset = AlignWorkspace(
        prepareScratchOffset + AlignWorkspace(solveBytes) + scoreBytes);
    const uint64_t qgScaledBytes = tokenHeads * kDim * dataBytes;

    const uint64_t postWuScratchOffset = AlignWorkspace(qgScaledOffset + qgScaledBytes);
    const uint64_t postWuScratchBytes = (fusePostWu || fuseRecurrentPostWu) ? 0 :
        tokenHeads * static_cast<uint64_t>(kDim) * sizeof(float);
    uint64_t gdnOffset = AlignWorkspace(postWuScratchOffset + postWuScratchBytes);
    const uint64_t vWorkspaceOffset = gdnOffset;
    gdnOffset = AlignWorkspace(gdnOffset + static_cast<uint64_t>(blockDim) * chunkSize *
        vDim * sizeof(float) * KDA_GDN_PIPELINE_DEPTH);
    const uint64_t vUpdateWorkspaceOffset = gdnOffset;
    gdnOffset = AlignWorkspace(gdnOffset + static_cast<uint64_t>(blockDim) * chunkSize *
        vDim * sizeof(float) * KDA_GDN_PIPELINE_DEPTH);
    const uint64_t kDecayWorkspaceOffset = gdnOffset;
    gdnOffset = AlignWorkspace(gdnOffset + static_cast<uint64_t>(blockDim) * chunkSize *
        kDim * sizeof(float) * KDA_GDN_PIPELINE_DEPTH);
    const uint64_t hWorkspaceOffset = gdnOffset;
    gdnOffset = AlignWorkspace(gdnOffset + static_cast<uint64_t>(blockDim) * kDim *
        vDim * sizeof(float) * KDA_GDN_PIPELINE_DEPTH);
    const uint64_t numSeqWorkspaceOffset = gdnOffset;
    const uint64_t tokenBatch = isVarLen ? static_cast<uint64_t>(seqNum) : 1;
    gdnOffset = AlignWorkspace(gdnOffset + (tokenBatch + 1) * sizeof(int64_t));
    const uint64_t numChunksWorkspaceOffset = gdnOffset;
    gdnOffset = AlignWorkspace(gdnOffset + (tokenBatch + 1) * sizeof(int64_t));

    const uint64_t outputScratchOffset = gdnOffset;
    const uint64_t outputElements = tokenHeads * vDim;
    const uint64_t totalWorkspace = AlignWorkspace(outputScratchOffset + 2 * outputElements * sizeof(float));
    context->SetBlockDim(blockDim);
    context->SetTilingKey(2);
    context->SetScheduleMode(KDA_BATCH_MODE);
    context->GetWorkspaceSizes(1)[0] = platform.GetLibApiWorkSpaceSize() + totalWorkspace;

    ChunkKdaFwdFusedA5TilingData tiling;
    tiling.set_batch(batch);
    tiling.set_seqNum(seqNum);
    tiling.set_qHeadNum(qHeads);
    tiling.set_vHeadNum(vHeads);
    tiling.set_seqlen(seqlen);
    tiling.set_kHeadDim(kDim);
    tiling.set_vHeadDim(vDim);
    tiling.set_chunkSize(chunkSize);
    tiling.set_totalChunks(totalChunks);
    tiling.set_scale(scale);
    tiling.set_hasInitialState(hasInitialState);
    tiling.set_isVarLen(isVarLen);
    tiling.set_safeGate(safeGate);
    tiling.set_inputSequenceMajor(inputSequenceMajor);
    tiling.set_fusePostWu(fusePostWu);
    tiling.set_fuseRecurrentPostWu(fuseRecurrentPostWu);
    tiling.set_computeGateInPrepare(computeGateInPrepare);
    tiling.set_hasALog(hasALog);
    tiling.set_hasDtBias(hasDtBias);
    tiling.set_lowerBound(lowerBound);
    tiling.set_prepareUsedCoreNum(blockDim);
    tiling.set_postWuUsedCoreNum(blockDim);
    tiling.set_prepareAqkFp32Offset(aqkFp32Offset);
    tiling.set_prepareAkkFp32Offset(akkFp32Offset);
    tiling.set_prepareScratchOffset(prepareScratchOffset);
    tiling.set_qgScaledOffset(qgScaledOffset);
    tiling.set_postWuScratchOffset(postWuScratchOffset);
    tiling.set_outputUsedCoreNum(blockDim);
    tiling.set_outputScratchOffset(outputScratchOffset);

    tiling.set_kNumHead(qHeads);
    tiling.set_vNumHead(vHeads);
    tiling.set_useInitialState(hasInitialState);
    tiling.set_storeFinalState(outputFinalState);
    tiling.set_storeQG(storeQG);
    tiling.set_storeVNew(storeVNew);
    tiling.set_storeH(storeH);
    tiling.set_isVariedLen(isVarLen ? 1 : 0);
    tiling.set_shapeBatch(isVarLen ? 1 : batch);
    tiling.set_tokenBatch(isVarLen ? seqNum : 1);
    tiling.set_vWorkspaceOffset(vWorkspaceOffset);
    tiling.set_vUpdateWorkspaceOffset(vUpdateWorkspaceOffset);
    tiling.set_kDecayWorkspaceOffset(kDecayWorkspaceOffset);
    tiling.set_hWorkspaceOffset(hWorkspaceOffset);
    tiling.set_numSeqWorkspaceOffset(numSeqWorkspaceOffset);
    tiling.set_numChunksWorkspaceOffset(numChunksWorkspaceOffset);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepare4ChunkKdaFwdFusedA5(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ChunkKdaFwdFusedA5)
    .Tiling(Tiling4ChunkKdaFwdFusedA5)
    .TilingParse<ChunkKdaFwdFusedA5CompileInfo>(TilingPrepare4ChunkKdaFwdFusedA5);
} // namespace optiling
