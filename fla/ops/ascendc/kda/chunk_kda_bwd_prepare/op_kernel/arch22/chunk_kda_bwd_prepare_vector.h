#ifndef CHUNK_KDA_BWD_PREPARE_ARCH22_VECTOR_H
#define CHUNK_KDA_BWD_PREPARE_ARCH22_VECTOR_H
#include "chunk_kda_bwd_prepare_common.h"
namespace KDA {
class ChunkKdaBwdPrepareVector {
public:
    __aicore__ inline void Init(GM_ADDR cu, GM_ADDR indices, GM_ADDR out, GM_ADDR scratch,
        const ChunkKdaBwdPrepareTilingData *tiling, AscendC::TPipe *pipe)
    {
        cu_ = cu; indices_ = indices; t_ = tiling; pipe_ = pipe;
        raw_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(scratch));
        out_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(out));
        pipe_->InitBuffer(ub_, UB_BYTES);
        AllocateEvents<AscendC::HardEvent::V_MTE2>(pipe_, rawFree_, 2, true);
        AllocateEvents<AscendC::HardEvent::MTE2_V>(pipe_, rawReady_, 2, false);
        AllocateEvents<AscendC::HardEvent::MTE3_V>(pipe_, outFree_, 2, true);
        AllocateEvents<AscendC::HardEvent::V_MTE3>(pipe_, outReady_, 2, false);
        auto mask = ub_.Get<float>()[4 * RAW_ELEMENTS];
        AscendC::Duplicate(mask, 0.0f, RAW_ELEMENTS);
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t row = 0; row < 64; ++row)
            AscendC::Duplicate(mask[row * 64], t_->scale, row + 1);
        AscendC::PipeBarrier<PIPE_V>();
        for (uint16_t record = 0; record < 4; ++record)
            AscendC::CrossCoreSetFlag<CROSS_MODE, PIPE_MTE2>(FREE_BASE + record);
    }
    __aicore__ inline void Process()
    {
        const int64_t block = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        const int64_t blocks = AscendC::GetBlockNum();
        const uint32_t aiv = AscendC::GetSubBlockIdx();
        uint64_t generation = 0;
        for (int64_t work = block; work < t_->workTaskNum; work += blocks) {
            const int64_t begin = (work / t_->chunkTaskNum) * 4;
            const int64_t end = KdaMin(begin + 4, t_->NV);
            ChunkInfo chunk;
            ResolveChunk(work % t_->chunkTaskNum, cu_, indices_, *t_, chunk);
            if (!chunk.valid) continue;
            for (int64_t head = begin; head < end; ++head, ++generation) {
                const uint32_t target = generation & 1, slot = (generation >> 1) & 1;
                const uint32_t record = 2 * target + slot;
                AscendC::CrossCoreWaitFlag<CROSS_MODE, PIPE_MTE2>(READY_BASE + record);
                if (target != aiv) {
                    AscendC::CrossCoreSetFlag<CROSS_MODE, PIPE_MTE2>(FREE_BASE + record);
                    continue;
                }
                const uint32_t rows = chunk.validRows, aligned = (rows + 7) / 8 * 8;
                auto raw = ub_.Get<float>()[slot * RAW_ELEMENTS];
                auto dst = ub_.Get<float>()[2 * RAW_ELEMENTS + slot * RAW_ELEMENTS];
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(rawFree_[slot]);
                AscendC::DataCopyExtParams p{static_cast<uint16_t>(rows), rows * 4,
                                            (64 - rows) * 4, (64 - aligned) / 8, 0};
                AscendC::DataCopyPadExtParams<float> pad{true, 0,
                    static_cast<uint8_t>(aligned - rows), 0.0f};
                AscendC::DataCopyPad(raw, raw_[(block * 4 + record) * RECORD_ELEMENTS], p, pad);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(rawReady_[slot]);
                AscendC::CrossCoreSetFlag<CROSS_MODE, PIPE_MTE2>(FREE_BASE + record);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(rawReady_[slot]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outFree_[slot]);
                // Never read unwritten workspace columns. Build a safe tile
                // first, including exact zeros above the diagonal and in tails.
                AscendC::Duplicate(dst, 0.0f, RAW_ELEMENTS);
                AscendC::PipeBarrier<PIPE_V>();
                for (uint32_t row = 0; row < rows; ++row)
                    AscendC::Adds(dst[row * 64], raw[row * 64], 0.0f, row + 1);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(rawFree_[slot]);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Mul(dst, dst, ub_.Get<float>()[4 * RAW_ELEMENTS], rows * 64);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(outReady_[slot]);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(outReady_[slot]);
                AscendC::DataCopy(out_[TokenOutputOffset(*t_, chunk, head, 64)], dst, rows * 64);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outFree_[slot]);
            }
        }
        ReleaseEvents<AscendC::HardEvent::V_MTE2>(pipe_, rawFree_, 2, true);
        ReleaseEvents<AscendC::HardEvent::MTE2_V>(pipe_, rawReady_, 2, false);
        ReleaseEvents<AscendC::HardEvent::MTE3_V>(pipe_, outFree_, 2, true);
        ReleaseEvents<AscendC::HardEvent::V_MTE3>(pipe_, outReady_, 2, false);
    }
private:
    AscendC::TPipe *pipe_;
    const ChunkKdaBwdPrepareTilingData *t_;
    GM_ADDR cu_;
    GM_ADDR indices_;
    AscendC::GlobalTensor<float> raw_, out_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> ub_;
    AscendC::TEventID rawFree_[2], rawReady_[2], outFree_[2], outReady_[2];
};
}
#endif
