/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#ifndef RECURRENT_GDN_A2A3_V2_H
#define RECURRENT_GDN_A2A3_V2_H

#include "kernel_operator.h"
#include "recurrent_gated_delta_rule_struct.h"
#include "recurrent_gated_delta_rule_df.h"

namespace RecurrentGatedDeltaRule {
using namespace AscendC;

struct RGDRV2InitParams {
    GM_ADDR query;
    GM_ADDR key;
    GM_ADDR value;
    GM_ADDR beta;
    GM_ADDR state;
    GM_ADDR lengths;
    GM_ADDR indices;
    GM_ADDR gate;
    GM_ADDR accepted;
    GM_ADDR aLog;
    GM_ADDR dtBias;
    GM_ADDR out;
};

// Each physical region has one owner. In particular a BF16 snapshot lives in
// the consumed source bank, never in the FP32 bank used by the next recurrence.
template <typename State, typename Beta, typename Gate>
class RGDRV2 {
public:
    __aicore__ inline RGDRV2(const RecurrentGatedDeltaRuleTilingData *td) : td_(*td) {}

    __aicore__ inline void Init(const RGDRV2InitParams &p, TPipe *pipe)
    {
        pipe_ = pipe;
        q_.SetGlobalBuffer((__gm__ bfloat16_t *)p.query);
        k_.SetGlobalBuffer((__gm__ bfloat16_t *)p.key);
        v_.SetGlobalBuffer((__gm__ bfloat16_t *)p.value);
        beta_.SetGlobalBuffer((__gm__ Beta *)p.beta);
        gate_.SetGlobalBuffer((__gm__ Gate *)p.gate);
        state_.SetGlobalBuffer((__gm__ State *)p.state);
        lengths_.SetGlobalBuffer((__gm__ int32_t *)p.lengths);
        indices_.SetGlobalBuffer((__gm__ int32_t *)p.indices);
        accepted_.SetGlobalBuffer((__gm__ int32_t *)p.accepted);
        aLog_.SetGlobalBuffer((__gm__ float *)p.aLog);
        bias_.SetGlobalBuffer((__gm__ float *)p.dtBias);
        out_.SetGlobalBuffer((__gm__ bfloat16_t *)p.out);
        pipe_->InitBuffer(ub_, 162240);
        for (uint32_t bank = 0; bank < 2; ++bank) {
            for (uint32_t input = 0; input < 3; ++input) {
                ready_[bank][input] = pipe_->AllocEventID<HardEvent::MTE2_V>();
            }
            rawFree_[bank] = pipe_->AllocEventID<HardEvent::V_MTE2>();
            stateReady_[bank] = pipe_->AllocEventID<HardEvent::V_MTE3>();
            stateFree_[bank] = pipe_->AllocEventID<HardEvent::MTE3_V>();
            outReady_[bank] = pipe_->AllocEventID<HardEvent::V_MTE3>();
            outFree_[bank] = pipe_->AllocEventID<HardEvent::MTE3_V>();
        }
        scalarToVector_ = pipe_->AllocEventID<HardEvent::S_V>();
        taskUbFree_ = pipe_->AllocEventID<HardEvent::V_MTE2>();
        taskGmFree_ = pipe_->AllocEventID<HardEvent::MTE3_MTE2>();
    }

    __aicore__ inline void Process()
    {
        const uint64_t blocks = GetBlockNum();
        const uint64_t block = GetBlockIdx();
        const uint64_t quotient = td_.taskCount / blocks;
        const uint64_t remainder = td_.taskCount % blocks;
        const uint64_t begin = block * quotient + (block < remainder ? block : remainder);
        const uint64_t end = begin + quotient + (block < remainder ? 1 : 0);
        for (uint64_t task = begin; task < end; ++task) {
            const uint64_t batch = task / td_.valueHeads;
            const uint64_t head = task % td_.valueHeads;
            const int32_t length = lengths_.GetValue(batch + 1);
            if (length == 0) continue;
            uint64_t start = lengths_.GetValue(0);
            for (uint64_t b = 0; b < batch; ++b) start += lengths_.GetValue(b + 1);
            RunHead(batch, head, start, static_cast<uint32_t>(length));
        }
        ReleaseEvents();
    }

private:
    template <typename T>
    __aicore__ inline LocalTensor<T> At(uint32_t byteOffset, uint32_t count)
    {
        return ub_.GetWithOffset<T>(count, byteOffset);
    }
    __aicore__ inline LocalTensor<float> StateBank(uint32_t bank) { return At<float>(bank * 65536, 16384); }
    __aicore__ inline LocalTensor<State> Snapshot(uint32_t bank)
    {
        return At<State>(bank * 65536 + 32768, 16384);
    }
    __aicore__ inline LocalTensor<bfloat16_t> Raw(uint32_t bank, uint32_t input)
    {
        return At<bfloat16_t>(131072 + bank * 6208 + input * 2048, 1024);
    }

    __aicore__ inline void ReclaimState(uint32_t bank)
    {
        if (statePending_[bank]) {
            WaitFlag<HardEvent::MTE3_V>(stateFree_[bank]);
            statePending_[bank] = false;
        }
    }
    __aicore__ inline void ReclaimOut(uint32_t bank)
    {
        if (outPending_[bank]) {
            WaitFlag<HardEvent::MTE3_V>(outFree_[bank]);
            outPending_[bank] = false;
        }
    }
    __aicore__ inline void ReclaimRaw(uint32_t bank)
    {
        if (rawPending_[bank]) {
            WaitFlag<HardEvent::V_MTE2>(rawFree_[bank]);
            rawPending_[bank] = false;
        }
    }

    __aicore__ inline void CopyInput(uint32_t bank, uint32_t input, uint64_t start,
                                     uint64_t head, uint32_t count)
    {
        const uint64_t *strides = input == 0 ? td_.queryStrides :
                                  (input == 1 ? td_.keyStrides : td_.valueStrides);
        const uint64_t offset = start * strides[0] + head * strides[1];
        DataCopyExtParams params{static_cast<uint16_t>(count), 256,
                                 static_cast<uint32_t>((strides[0] - 128) * 2), 0, 0};
        DataCopyPadExtParams<bfloat16_t> padding{false, 0, 0, 0};
        auto dst = Raw(bank, input);
        if (input == 0) DataCopyPad(dst, q_[offset], params, padding);
        else if (input == 1) DataCopyPad(dst, k_[offset], params, padding);
        else DataCopyPad(dst, v_[offset], params, padding);
        SetFlag<HardEvent::MTE2_V>(ready_[bank][input]);
    }

    __aicore__ inline void PrefetchQ(uint32_t bank, uint64_t start, uint64_t head, uint32_t count)
    {
        ReclaimRaw(bank);
        CopyInput(bank, 0, start, head, count);
    }

    __aicore__ inline void CopyWindowMeta(uint32_t bank, uint64_t batch, uint64_t head,
                                          uint64_t start, uint32_t position, uint32_t count)
    {
        auto betaRaw = At<Beta>(131072 + bank * 6208 + 6144, 8);
        auto gateRaw = At<Gate>(131072 + bank * 6208 + 6176, 8);
        auto slots = At<int32_t>(162144, 8);
        for (uint32_t i = 0; i < 8; ++i) {
            betaRaw.SetValue(i, static_cast<Beta>(0.0f));
            if (td_.hasGama) gateRaw.SetValue(i, static_cast<Gate>(0.0f));
            slots.SetValue(i, 0);
        }
        // `start` is already the absolute GM token of this window.  Only the
        // checkpoint layout uses the request-local `position` axis.
        const uint64_t indexBase = td_.checkpointWidth ? batch * td_.checkpointWidth + position : start;
        for (uint32_t i = 0; i < count; ++i) {
            betaRaw.SetValue(i, static_cast<Beta>(beta_.GetValue((start + i) * td_.valueHeads + head)));
            if (td_.hasGama) {
                gateRaw.SetValue(i, static_cast<Gate>(gate_.GetValue((start + i) * td_.valueHeads + head)));
            }
            slots.SetValue(i, indices_.GetValue(indexBase + i));
        }
        // GM metadata is gathered once per window; all gate/beta arithmetic
        // below is vectorized over the padded window.
        SetFlag<HardEvent::S_V>(scalarToVector_);
    }

    __aicore__ inline void PrepareWindow(uint32_t bank, uint64_t batch, uint64_t head,
                                         uint64_t start, uint32_t position, uint32_t count,
                                         float aLog, float bias)
    {
        auto betaRaw = At<Beta>(131072 + bank * 6208 + 6144, 8);
        auto gateRaw = At<Gate>(131072 + bank * 6208 + 6176, 8);
        const uint64_t keyHead = head / (td_.valueHeads / td_.keyHeads);
        CopyWindowMeta(bank, batch, head, start, position, count);
        WaitFlag<HardEvent::MTE2_V>(ready_[bank][0]);
        CopyInput(bank, 1, start, keyHead, count);
        Cast(At<float>(143488, 1024), Raw(bank, 0), RoundMode::CAST_NONE, count * 128);
        WaitFlag<HardEvent::MTE2_V>(ready_[bank][1]);
        CopyInput(bank, 2, start, head, count);
        Cast(At<float>(147584, 1024), Raw(bank, 1), RoundMode::CAST_NONE, count * 128);
        WaitFlag<HardEvent::MTE2_V>(ready_[bank][2]);
        Cast(At<float>(151680, 1024), Raw(bank, 2), RoundMode::CAST_NONE, count * 128);
        WaitFlag<HardEvent::S_V>(scalarToVector_);
        auto beta = At<float>(161408, 8);
        auto gate = At<float>(161440, 8);
        auto tmp0 = At<float>(161472, 8);
        auto tmp1 = At<float>(161504, 8);
        auto tmp2 = At<float>(161536, 8);
        auto tmp3 = At<float>(161568, 8);
        // Pad short metadata windows to the A2 FP32 vector granularity.  The
        // token loop consumes only [0, count), so the padded tail is ignored.
        const uint32_t vectorCount = 8;
        if constexpr (std::is_same<Beta, float>::value) DataCopy(beta, betaRaw, vectorCount);
        else Cast(beta, betaRaw, RoundMode::CAST_NONE, vectorCount);
        if (td_.useBetaSigmoidInKernel) {
            Abs(tmp0, beta, vectorCount);
            PipeBarrier<PIPE_V>();
            Muls(tmp1, tmp0, -1.0f, vectorCount);
            Exp(tmp1, tmp1, vectorCount);
            PipeBarrier<PIPE_V>();
            Adds(tmp2, tmp1, 1.0f, vectorCount);
            PipeBarrier<PIPE_V>();
            Duplicate(tmp0, 1.0f, vectorCount);
            Div(tmp0, tmp0, tmp2, vectorCount);
            Div(tmp2, tmp1, tmp2, vectorCount);
            PipeBarrier<PIPE_V>();
            auto betaMask = tmp3.template ReinterpretCast<uint8_t>();
            CompareScalar(betaMask, beta, 0.0f, CMPMODE::GE, vectorCount);
            PipeBarrier<PIPE_V>();
            Select(beta, betaMask, tmp0, tmp2, SELMODE::VSEL_TENSOR_TENSOR_MODE, vectorCount);
            if (td_.allowNegEigval) {
                PipeBarrier<PIPE_V>();
                Muls(beta, beta, 2.0f, vectorCount);
            }
        }
        if (td_.hasGama) {
            if constexpr (std::is_same<Gate, float>::value) DataCopy(gate, gateRaw, vectorCount);
            else Cast(gate, gateRaw, RoundMode::CAST_NONE, vectorCount);
            if constexpr (std::is_same<State, float>::value) {
                if (td_.useGateInKernel) {
                    DataCopy(tmp3, gate, vectorCount);
                    PipeBarrier<PIPE_V>();
                }
            }
            if (td_.useGateInKernel) {
                Adds(gate, gate, bias, vectorCount);
                PipeBarrier<PIPE_V>();
                Maxs(tmp0, gate, 0.0f, vectorCount);
                Abs(tmp1, gate, vectorCount);
                Muls(tmp1, tmp1, -1.0f, vectorCount);
                Exp(tmp1, tmp1, vectorCount);
                PipeBarrier<PIPE_V>();
                Adds(tmp2, tmp1, 1.0f, vectorCount);
                Ln(tmp2, tmp2, vectorCount);
                PipeBarrier<PIPE_V>();
                Add(gate, tmp0, tmp2, vectorCount);
                Duplicate(tmp1, aLog, vectorCount);
                Exp(tmp1, tmp1, vectorCount);
                PipeBarrier<PIPE_V>();
                Mul(gate, gate, tmp1, vectorCount);
                Muls(gate, gate, -1.0f, vectorCount);
                PipeBarrier<PIPE_V>();
                Exp(gate, gate, vectorCount);
            } else {
                Exp(gate, gate, vectorCount);
            }
        }
        // The raw bank is only reusable after all MTE2-produced values have
        // been consumed by V.  The vectorized casts above are the last raw
        // readers; the remaining stages operate on typed UB vectors.
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE2>(rawFree_[bank]);
        rawPending_[bank] = true;
        auto q = At<float>(143488, 1024);
        auto k = At<float>(147584, 1024);
        if (td_.useQkL2NormInKernel) {
            NormalizeWindow(q, count);
            NormalizeWindow(k, count);
        }
        Muls(q, q, td_.scale, count * 128);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void NormalizeWindow(LocalTensor<float> vector, uint32_t rows)
    {
        auto product = At<float>(157312, 1024);
        auto sum = At<float>(155776, 128);
        auto inverse = At<float>(156800, 128);
        auto correction = At<float>(161472, 8);
        Mul(product, vector, vector, rows * 128);
        PipeBarrier<PIPE_V>();
        for (uint32_t row = 0; row < rows; ++row) {
            Add(product[row * 128 + 64], product[row * 128], product[row * 128 + 64], 64);
            WholeReduceSum(sum[row], product[row * 128 + 64], 64, 1, 1, 1, 8);
        }
        PipeBarrier<PIPE_V>();
        const uint32_t vectorRows = 8;
        Adds(sum, sum, 1.0e-6f, vectorRows);
        PipeBarrier<PIPE_V>();
        Adds(inverse, sum, 0.0f, vectorRows);
        Rsqrt(inverse, inverse, vectorRows);
        PipeBarrier<PIPE_V>();
        for (uint32_t iteration = 0; iteration < 3; ++iteration) {
            Mul(correction, inverse, inverse, vectorRows);
            PipeBarrier<PIPE_V>();
            Mul(correction, correction, sum, vectorRows);
            PipeBarrier<PIPE_V>();
            Muls(correction, correction, -0.5f, vectorRows);
            Adds(correction, correction, 1.5f, vectorRows);
            PipeBarrier<PIPE_V>();
            Mul(inverse, inverse, correction, vectorRows);
            PipeBarrier<PIPE_V>();
        }
        // Broadcast one inverse norm to each 128-element token row and scale
        // the whole window in one VEC pass.  This removes the former scalar
        // read/scale loop over tokens while retaining an extra Newton step for
        // the A2 rsqrt approximation.
        auto broadcast = At<float>(157312, 1024);
        Brcb(broadcast, inverse, static_cast<uint8_t>((rows + 7U) / 8U), {1, 8});
        PipeBarrier<PIPE_V>();
        // Brcb stores one 8-scalar block per repeat.  The repeat form below
        // reuses those row scalars across all 128 features without reading
        // beyond the compact broadcast block.
        for (uint32_t half = 0; half < 128; half += 64) {
            Mul(vector[half], vector[half], broadcast, 64, rows,
                {1, 1, 0, 16, 16, 1});
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline uint64_t StateOffset(uint64_t slot, uint64_t head)
    {
        return slot * td_.stateStrides[0] + head * td_.stateStrides[1];
    }

    __aicore__ inline void LoadState(uint64_t slot, uint64_t head)
    {
        SetFlag<HardEvent::V_MTE2>(taskUbFree_);
        WaitFlag<HardEvent::V_MTE2>(taskUbFree_);
        SetFlag<HardEvent::MTE3_MTE2>(taskGmFree_);
        WaitFlag<HardEvent::MTE3_MTE2>(taskGmFree_);
        DataCopyExtParams params{128, static_cast<uint32_t>(128 * sizeof(State)),
            static_cast<uint32_t>((td_.stateStrides[2] - 128) * sizeof(State)), 0, 0};
        DataCopyPadExtParams<State> padding{false, 0, 0, 0};
        if constexpr (std::is_same<State, float>::value) {
            DataCopyPad(StateBank(0), state_[StateOffset(slot, head)], params, padding);
        } else {
            DataCopyPad(Snapshot(1), state_[StateOffset(slot, head)], params, padding);
        }
        SetFlag<HardEvent::MTE2_V>(ready_[0][0]);
        WaitFlag<HardEvent::MTE2_V>(ready_[0][0]);
        if constexpr (!std::is_same<State, float>::value) {
            Cast(StateBank(0), Snapshot(1), RoundMode::CAST_NONE, 16384);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void DotRows(LocalTensor<float> dst, LocalTensor<float> matrix,
                                    LocalTensor<float> vector, float alpha)
    {
        auto product = At<float>(157312, 1024);
        // Keep the reduction destination isolated for every row.  The vector
        // multiply is still used for the complete 128-element row; the
        // reduction instruction has an implementation-defined 8-lane write
        // footprint, so batching rows into one destination window can overlap
        // the next row and corrupt adjacent UB state.
        for (uint32_t row = 0; row < 128; ++row) {
            if (alpha != 1.0f) {
                Muls(product, matrix[row * 128], alpha, 128);
                PipeBarrier<PIPE_V>();
                Mul(product, product, vector, 128);
            } else {
                Mul(product, matrix[row * 128], vector, 128);
            }
            PipeBarrier<PIPE_V>();
            Add(product[64], product, product[64], 64);
            WholeReduceSum(dst[row], product[64], 64, 1, 1, 1, 8);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void EmitState(uint32_t bank, uint64_t slot, uint64_t head)
    {
        SetFlag<HardEvent::V_MTE3>(stateReady_[bank]);
        WaitFlag<HardEvent::V_MTE3>(stateReady_[bank]);
        DataCopyExtParams params{128, static_cast<uint32_t>(128 * sizeof(State)), 0,
            static_cast<uint32_t>((td_.stateStrides[2] - 128) * sizeof(State)), 0};
        if constexpr (std::is_same<State, float>::value) {
            DataCopyPad(state_[StateOffset(slot, head)], StateBank(bank), params);
        } else {
            DataCopyPad(state_[StateOffset(slot, head)], Snapshot(bank), params);
        }
        SetFlag<HardEvent::MTE3_V>(stateFree_[bank]);
        statePending_[bank] = true;
    }

    __aicore__ inline void CompensatedOuter(LocalTensor<float> target, LocalTensor<float> scratch,
                                            LocalTensor<float> key, LocalTensor<float> delta)
    {
        // Split each multiplier into two exact FP32 parts without scaling.
        // Clearing twelve mantissa bits avoids splitter multiplication overflow.
        auto mask = At<uint16_t>(156800, 256);
        Duplicate(mask, static_cast<uint16_t>(0xffff), 256);
        PipeBarrier<PIPE_V>();
        uint64_t alternating[2] = {0x5555555555555555ULL, 0x5555555555555555ULL};
        Duplicate(mask, static_cast<uint16_t>(0xf000), alternating, 2, 1, 8);
        PipeBarrier<PIPE_V>();
        auto keyHigh = scratch;
        auto keyLow = scratch[128];
        auto deltaHigh = scratch[256];
        auto deltaLow = scratch[384];
        And(keyHigh.ReinterpretCast<uint16_t>(), key.ReinterpretCast<uint16_t>(), mask, 256);
        And(deltaHigh.ReinterpretCast<uint16_t>(), delta.ReinterpretCast<uint16_t>(), mask, 256);
        PipeBarrier<PIPE_V>();
        Sub(keyLow, key, keyHigh, 128);
        Sub(deltaLow, delta, deltaHigh, 128);
        PipeBarrier<PIPE_V>();
        auto highBroadcast = scratch[512];
        auto lowBroadcast = scratch[1536];
        Brcb(highBroadcast, deltaHigh, 16, {1, 8});
        Brcb(lowBroadcast, deltaLow, 16, {1, 8});
        PipeBarrier<PIPE_V>();
        auto product = scratch[2560];
        auto productError = scratch[3584];
        auto temp = scratch[4608];
        auto error = scratch[5632];
        auto previous = scratch[6656];
        auto sum = scratch[7680];
        auto recovered = scratch[8704];
        auto finite = scratch[9728].ReinterpretCast<uint8_t>();
        auto broadcast = At<float>(157312, 1024);
        for (uint32_t row = 0; row < 128; row += 8) {
            DataCopy(previous, target[row * 128], 1024);
            for (uint32_t half = 0; half < 128; half += 64) {
                Mul(product[half], key[half], broadcast[row * 8], 64, 8, {1, 1, 0, 16, 0, 1});
                Mul(productError[half], keyHigh[half], highBroadcast[row * 8], 64, 8,
                    {1, 1, 0, 16, 0, 1});
            }
            PipeBarrier<PIPE_V>();
            Sub(productError, productError, product, 1024);
            for (uint32_t half = 0; half < 128; half += 64) {
                Mul(temp[half], keyHigh[half], lowBroadcast[row * 8], 64, 8, {1, 1, 0, 16, 0, 1});
            }
            PipeBarrier<PIPE_V>();
            Add(productError, productError, temp, 1024);
            PipeBarrier<PIPE_V>();
            for (uint32_t half = 0; half < 128; half += 64) {
                Mul(temp[half], keyLow[half], highBroadcast[row * 8], 64, 8, {1, 1, 0, 16, 0, 1});
            }
            PipeBarrier<PIPE_V>();
            Add(productError, productError, temp, 1024);
            PipeBarrier<PIPE_V>();
            for (uint32_t half = 0; half < 128; half += 64) {
                Mul(temp[half], keyLow[half], lowBroadcast[row * 8], 64, 8, {1, 1, 0, 16, 0, 1});
            }
            PipeBarrier<PIPE_V>();
            Add(productError, productError, temp, 1024);
            Add(sum, product, previous, 1024);
            PipeBarrier<PIPE_V>();
            Sub(recovered, sum, product, 1024);
            PipeBarrier<PIPE_V>();
            Sub(temp, sum, recovered, 1024);
            PipeBarrier<PIPE_V>();
            Sub(error, product, temp, 1024);
            PipeBarrier<PIPE_V>();
            Sub(temp, previous, recovered, 1024);
            PipeBarrier<PIPE_V>();
            Add(error, error, temp, 1024);
            PipeBarrier<PIPE_V>();
            Add(error, error, productError, 1024);
            PipeBarrier<PIPE_V>();
            Add(temp, sum, error, 1024);
            Abs(recovered, sum, 1024);
            PipeBarrier<PIPE_V>();
            CompareScalar(finite, recovered, 3.402823466e+38f, CMPMODE::LE, 1024);
            PipeBarrier<PIPE_V>();
            Select(target[row * 128], finite, temp, sum, SELMODE::VSEL_TENSOR_TENSOR_MODE, 1024);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void Token(uint32_t token, uint64_t head,
                                  uint64_t globalToken, uint32_t src, uint32_t dst,
                                  float aLog, float bias)
    {
        auto query = At<float>(143488 + token * 512, 128);
        auto key = At<float>(147584 + token * 512, 128);
        auto value = At<float>(151680 + token * 512, 128);
        const float beta = At<float>(161408, 8).GetValue(token);
        float alpha = td_.hasGama ? At<float>(161440, 8).GetValue(token) : 1.0f;
        if constexpr (std::is_same<State, float>::value) {
            if (td_.hasGama && td_.useGateInKernel) {
                const float rawGate = At<float>(161568, 8).GetValue(token);
                const bool compensatedGate = aLog >= -80.0f && aLog <= 80.0f &&
                    rawGate >= -1.0e20f && rawGate <= 1.0e20f && bias >= -1.0e20f && bias <= 1.0e20f;
                if (compensatedGate) alpha = RecurrentGdnDf::RawAlpha(rawGate, aLog, bias);
            }
        }
        auto source = StateBank(src);
        auto target = StateBank(dst);
        auto delta = At<float>(155776, 128);
        if constexpr (std::is_same<State, float>::value) {
            ReclaimState(dst);
            Muls(target, source, alpha, 16384);
            PipeBarrier<PIPE_V>();
            DotRows(delta, target, key, 1.0f);
        } else {
            DotRows(delta, source, key, alpha);
        }
        Sub(delta, value, delta, 128);
        PipeBarrier<PIPE_V>();
        Muls(delta, delta, beta, 128);
        PipeBarrier<PIPE_V>();
        auto broadcast = At<float>(157312, 1024);
        Brcb(broadcast, delta, 16, {1, 8});
        PipeBarrier<PIPE_V>();
        if constexpr (!std::is_same<State, float>::value) {
            ReclaimState(dst);
            Muls(target, source, alpha, 16384);
            PipeBarrier<PIPE_V>();
        }
        if constexpr (std::is_same<State, float>::value) {
            // The old source has no remaining V readers. Reclaim its previous
            // copyout before using its storage for TwoProd/TwoSum scratch.
            // Neither A2 vmla nor vmadd guarantees IEEE single-round FP32 FMA.
            ReclaimState(src);
            CompensatedOuter(target, source, key, delta);
        } else {
            for (uint32_t half = 0; half < 128; half += 64) {
                MulAddDst(target[half], key[half], broadcast, 64, 128, {1, 1, 0, 16, 0, 1});
            }
        }
        PipeBarrier<PIPE_V>();
        auto output = At<float>(156288, 128);
        DotRows(output, target, query, 1.0f);
        const uint32_t outBank = globalToken % 2;
        ReclaimOut(outBank);
        auto outputBf16 = At<bfloat16_t>(161600 + outBank * 256, 128);
        Cast(outputBf16, output, RoundMode::CAST_RINT, 128);
        const int32_t writeSlot = At<int32_t>(162144, 8).GetValue(token);
        if constexpr (std::is_same<State, float>::value) {
            EmitState(dst, static_cast<uint64_t>(writeSlot), head);
        } else {
            // Source was reclaimed on its preceding turn as a destination.
            // All V reads above must finish before its upper half is reused.
            PipeBarrier<PIPE_V>();
            Cast(Snapshot(src), target, RoundMode::CAST_RINT, 16384);
            EmitState(src, static_cast<uint64_t>(writeSlot), head);
        }
        SetFlag<HardEvent::V_MTE3>(outReady_[outBank]);
        WaitFlag<HardEvent::V_MTE3>(outReady_[outBank]);
        DataCopy(out_[(globalToken * td_.valueHeads + head) * 128], outputBf16, 128);
        SetFlag<HardEvent::MTE3_V>(outFree_[outBank]);
        outPending_[outBank] = true;
    }

    __aicore__ inline void RunHead(uint64_t batch, uint64_t head, uint64_t start, uint32_t length)
    {
        const int32_t accepted = td_.hasAcceptedTokens ? accepted_.GetValue(batch) : 1;
        const uint64_t indexBase = td_.checkpointWidth ? batch * td_.checkpointWidth : start;
        const int32_t initial = indices_.GetValue(indexBase + static_cast<uint64_t>(accepted - 1));
        LoadState(static_cast<uint64_t>(initial), head);
        const float aLog = td_.hasALog ? aLog_.GetValue(head) : 0.0f;
        const float bias = td_.hasDtBias ? bias_.GetValue(head) : 0.0f;
        const uint64_t keyHead = head / (td_.valueHeads / td_.keyHeads);
        uint32_t src = 0;
        PrefetchQ(0, start, keyHead, length < 8 ? length : 8);
        for (uint32_t position = 0; position < length; position += 8) {
            const uint32_t bank = (position / 8) % 2;
            const uint32_t count = length - position < 8 ? length - position : 8;
            PrepareWindow(bank, batch, head, start + position, position, count, aLog, bias);
            if (position + count < length) {
                const uint32_t nextCount = length - position - count < 8 ? length - position - count : 8;
                PrefetchQ(1 - bank, start + position + count, keyHead, nextCount);
            }
            for (uint32_t token = 0; token < count; ++token) {
                Token(token, head, start + position + token, src, 1 - src, aLog, bias);
                src = 1 - src;
            }
        }
        for (uint32_t bank = 0; bank < 2; ++bank) {
            ReclaimState(bank);
            ReclaimOut(bank);
            ReclaimRaw(bank);
        }
    }

    __aicore__ inline void ReleaseEvents()
    {
        for (uint32_t bank = 0; bank < 2; ++bank) {
            for (uint32_t i = 0; i < 3; ++i) pipe_->ReleaseEventID<HardEvent::MTE2_V>(ready_[bank][i]);
            pipe_->ReleaseEventID<HardEvent::V_MTE2>(rawFree_[bank]);
            pipe_->ReleaseEventID<HardEvent::V_MTE3>(stateReady_[bank]);
            pipe_->ReleaseEventID<HardEvent::MTE3_V>(stateFree_[bank]);
            pipe_->ReleaseEventID<HardEvent::V_MTE3>(outReady_[bank]);
            pipe_->ReleaseEventID<HardEvent::MTE3_V>(outFree_[bank]);
        }
        pipe_->ReleaseEventID<HardEvent::S_V>(scalarToVector_);
        pipe_->ReleaseEventID<HardEvent::V_MTE2>(taskUbFree_);
        pipe_->ReleaseEventID<HardEvent::MTE3_MTE2>(taskGmFree_);
    }

    RecurrentGatedDeltaRuleTilingData td_;
    TPipe *pipe_;
    TBuf<TPosition::VECCALC> ub_;
    GlobalTensor<bfloat16_t> q_, k_, v_, out_;
    GlobalTensor<Beta> beta_;
    GlobalTensor<Gate> gate_;
    GlobalTensor<State> state_;
    GlobalTensor<int32_t> lengths_, indices_, accepted_;
    GlobalTensor<float> aLog_, bias_;
    TEventID ready_[2][3], rawFree_[2];
    TEventID stateReady_[2], stateFree_[2], outReady_[2], outFree_[2];
    TEventID scalarToVector_, taskUbFree_, taskGmFree_;
    bool rawPending_[2] = {false, false};
    bool statePending_[2] = {false, false};
    bool outPending_[2] = {false, false};
};
} // namespace RecurrentGatedDeltaRule
#endif
