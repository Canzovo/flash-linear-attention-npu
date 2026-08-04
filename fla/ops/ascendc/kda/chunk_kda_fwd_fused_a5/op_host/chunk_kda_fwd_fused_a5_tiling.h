#pragma once

#include <cstdint>
#include <register/tilingdata_base.h>

namespace optiling {

BEGIN_TILING_DATA_DEF(ChunkKdaFwdFusedA5TilingData)
TILING_DATA_FIELD_DEF(int64_t, batch);
TILING_DATA_FIELD_DEF(int64_t, seqNum);
TILING_DATA_FIELD_DEF(int64_t, qHeadNum);
TILING_DATA_FIELD_DEF(int64_t, vHeadNum);
TILING_DATA_FIELD_DEF(int64_t, seqlen);
TILING_DATA_FIELD_DEF(int64_t, kHeadDim);
TILING_DATA_FIELD_DEF(int64_t, vHeadDim);
TILING_DATA_FIELD_DEF(int64_t, chunkSize);
TILING_DATA_FIELD_DEF(int64_t, totalChunks);
TILING_DATA_FIELD_DEF(float, scale);
TILING_DATA_FIELD_DEF(bool, hasInitialState);
TILING_DATA_FIELD_DEF(bool, isVarLen);
TILING_DATA_FIELD_DEF(bool, safeGate);
TILING_DATA_FIELD_DEF(bool, inputSequenceMajor);
TILING_DATA_FIELD_DEF(bool, fusePostWu);
TILING_DATA_FIELD_DEF(bool, fuseRecurrentPostWu);
TILING_DATA_FIELD_DEF(bool, computeGateInPrepare);
TILING_DATA_FIELD_DEF(bool, hasALog);
TILING_DATA_FIELD_DEF(bool, hasDtBias);
TILING_DATA_FIELD_DEF(float, lowerBound);
TILING_DATA_FIELD_DEF(int64_t, prepareUsedCoreNum);
TILING_DATA_FIELD_DEF(int64_t, postWuUsedCoreNum);
TILING_DATA_FIELD_DEF(int64_t, prepareAqkFp32Offset);
TILING_DATA_FIELD_DEF(int64_t, prepareAkkFp32Offset);
TILING_DATA_FIELD_DEF(int64_t, prepareScratchOffset);
TILING_DATA_FIELD_DEF(int64_t, qgScaledOffset);
TILING_DATA_FIELD_DEF(int64_t, postWuScratchOffset);
TILING_DATA_FIELD_DEF(int64_t, outputUsedCoreNum);
TILING_DATA_FIELD_DEF(int64_t, outputScratchOffset);

TILING_DATA_FIELD_DEF(int64_t, kNumHead);
TILING_DATA_FIELD_DEF(int64_t, vNumHead);
TILING_DATA_FIELD_DEF(bool, useInitialState);
TILING_DATA_FIELD_DEF(bool, storeFinalState);
TILING_DATA_FIELD_DEF(bool, storeQG);
TILING_DATA_FIELD_DEF(bool, storeVNew);
TILING_DATA_FIELD_DEF(bool, storeH);
TILING_DATA_FIELD_DEF(int64_t, isVariedLen);
TILING_DATA_FIELD_DEF(int64_t, shapeBatch);
TILING_DATA_FIELD_DEF(int64_t, tokenBatch);
TILING_DATA_FIELD_DEF(int64_t, vWorkspaceOffset);
TILING_DATA_FIELD_DEF(int64_t, vUpdateWorkspaceOffset);
TILING_DATA_FIELD_DEF(int64_t, kDecayWorkspaceOffset);
TILING_DATA_FIELD_DEF(int64_t, hWorkspaceOffset);
TILING_DATA_FIELD_DEF(int64_t, numSeqWorkspaceOffset);
TILING_DATA_FIELD_DEF(int64_t, numChunksWorkspaceOffset);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ChunkKdaFwdFusedA5, ChunkKdaFwdFusedA5TilingData)

struct ChunkKdaFwdFusedA5CompileInfo {};
} // namespace optiling
