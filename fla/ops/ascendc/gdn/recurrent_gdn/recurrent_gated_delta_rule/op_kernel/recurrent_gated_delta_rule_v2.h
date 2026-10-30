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
            scalarReady_[bank] = pipe_->AllocEventID<HardEvent::S_V>();
            scalarFree_[bank] = pipe_->AllocEventID<HardEvent::V_S>();
            stateReady_[bank] = pipe_->AllocEventID<HardEvent::V_MTE3>();
            stateFree_[bank] = pipe_->AllocEventID<HardEvent::MTE3_V>();
            outReady_[bank] = pipe_->AllocEventID<HardEvent::V_MTE3>();
            outFree_[bank] = pipe_->AllocEventID<HardEvent::MTE3_V>();
        }
        scalarToVector_ = pipe_->AllocEventID<HardEvent::S_V>();
        vectorToScalar_ = pipe_->AllocEventID<HardEvent::V_S>();
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
    __aicore__ inline LocalTensor<float> Fp(uint32_t input) { return At<float>(143488 + input * 4096, 1024); }
    __aicore__ inline LocalTensor<float> Tmp(uint32_t i) { return At<float>(161472 + i * 32, 8); }

    template <HardEvent Event>
    __aicore__ inline void Sync(TEventID id)
    {
        SetFlag<Event>(id);
        WaitFlag<Event>(id);
    }
    __aicore__ inline void VBarrier() { PipeBarrier<PIPE_V>(); }

    __aicore__ inline float ExpScalar(float value)
    {
        if (value >= -0.125f && value <= 0.125f) {
            // Keep the small gate's expm1 tail until the final addition to one.
            // Native VEC Exp can be one ULP away here, which is amplified by
            // cancellation in the FP32 State update.
            const float tail = 1.0f + value * (0.5f + value * (1.0f / 6.0f +
                value * (1.0f / 24.0f + value * (1.0f / 120.0f +
                value * (1.0f / 720.0f + value * (1.0f / 5040.0f + value / 40320.0f))))));
            return 1.0f + value * tail;
        }
        auto tmp = Tmp(0);
        Duplicate(tmp, value, 8);
        VBarrier();
        Exp(tmp, tmp, 8);
        Sync<HardEvent::V_S>(vectorToScalar_);
        return tmp.GetValue(0);
    }

    __aicore__ inline float Log1pScalar(float value)
    {
        // Avoid losing the exp(-abs(x)) tail in the addition to one.
        if (value < 0.015625f) {
            return value * (1.0f + value * (-0.5f + value * (1.0f / 3.0f - value * 0.25f)));
        }
        auto tmp = Tmp(1);
        Duplicate(tmp, value, 8);
        VBarrier();
        Adds(tmp, tmp, 1.0f, 8);
        VBarrier();
        Ln(tmp, tmp, 8);
        Sync<HardEvent::V_S>(vectorToScalar_);
        return tmp.GetValue(0);
    }

    __aicore__ inline float Normalize(LocalTensor<float> vector)
    {
        auto product = At<float>(156800, 128);
        auto sum = Tmp(2);
        Duplicate(sum, 0.0f, 8);
        Mul(product, vector, vector, 128);
        VBarrier();
        Add(product[64], product, product[64], 64);
        VBarrier();
        WholeReduceSum(sum, product[64], 64, 1, 1, 1, 8);
        VBarrier();
        Adds(sum, sum, 1.0e-6f, 8);
        VBarrier();
        auto inverseBlock = Tmp(3);
        auto correction = Tmp(0);
        Rsqrt(inverseBlock, sum, 8);
        VBarrier();
        // A2/A3's native rsqrt is an approximation (about 1e-3 relative
        // error). Two Newton steps restore FP32 accuracy before either Q or
        // K is used; checkpoint casts must not conceal this recurrence error.
        for (uint32_t iteration = 0; iteration < 2; ++iteration) {
            Mul(correction, inverseBlock, inverseBlock, 8);
            VBarrier();
            Mul(correction, correction, sum, 8);
            VBarrier();
            Muls(correction, correction, -0.5f, 8);
            VBarrier();
            Adds(correction, correction, 1.5f, 8);
            VBarrier();
            Mul(inverseBlock, inverseBlock, correction, 8);
            VBarrier();
        }
        Sync<HardEvent::V_S>(vectorToScalar_);
        float inverse = inverseBlock.GetValue(0);
        if constexpr (std::is_same<State, float>::value) {
            // Retain the Newton residual through its final rounding. The
            // ordinary FP32 Newton expression can finish one ULP low.
            const float x = sum.GetValue(0);
            if (x > 0.0f && x < 3.402823466e+38f && inverse > 0.0f) {
                const RecurrentGdnDf::Df r{inverse, 0.0f};
                const auto residual = RecurrentGdnDf::Sub({1.0f, 0.0f},
                    RecurrentGdnDf::Mul({x, 0.0f}, RecurrentGdnDf::Mul(r, r)));
                const auto refined = RecurrentGdnDf::Add(r,
                    RecurrentGdnDf::Mul({0.5f, 0.0f}, RecurrentGdnDf::Mul(r, residual)));
                inverse = refined.hi + refined.lo;
            }
        }
        Muls(vector, vector, inverse, 128);
        VBarrier();
        return inverse;
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

    __aicore__ inline void PrepareWindow(uint32_t bank, uint64_t batch, uint64_t head,
                                         uint64_t start, uint32_t position, uint32_t count)
    {
        if (scalarPending_[bank]) {
            WaitFlag<HardEvent::V_S>(scalarFree_[bank]);
            scalarPending_[bank] = false;
        }
        auto betaRaw = At<Beta>(131072 + bank * 6208 + 6144, 8);
        auto gateRaw = At<Gate>(131072 + bank * 6208 + 6176, 8);
        auto slots = At<int32_t>(162144, 8);
        for (uint32_t i = 0; i < 8; ++i) {
            betaRaw.SetValue(i, static_cast<Beta>(0.0f));
            if (td_.hasGama) gateRaw.SetValue(i, static_cast<Gate>(0.0f));
        }
        for (uint32_t i = 0; i < count; ++i) {
            betaRaw.SetValue(i, beta_.GetValue((start + i) * td_.valueHeads + head));
            if (td_.hasGama) gateRaw.SetValue(i, gate_.GetValue((start + i) * td_.valueHeads + head));
            const uint64_t index = td_.checkpointWidth ? batch * td_.checkpointWidth + position + i : start + i;
            slots.SetValue(i, indices_.GetValue(index));
        }
        SetFlag<HardEvent::S_V>(scalarReady_[bank]);
        const uint64_t keyHead = head / (td_.valueHeads / td_.keyHeads);
        WaitFlag<HardEvent::MTE2_V>(ready_[bank][0]);
        CopyInput(bank, 1, start, keyHead, count);
        Cast(Fp(0), Raw(bank, 0), RoundMode::CAST_NONE, count * 128);
        WaitFlag<HardEvent::MTE2_V>(ready_[bank][1]);
        CopyInput(bank, 2, start, head, count);
        Cast(Fp(1), Raw(bank, 1), RoundMode::CAST_NONE, count * 128);
        WaitFlag<HardEvent::MTE2_V>(ready_[bank][2]);
        Cast(Fp(2), Raw(bank, 2), RoundMode::CAST_NONE, count * 128);
        VBarrier();
        SetFlag<HardEvent::V_MTE2>(rawFree_[bank]);
        rawPending_[bank] = true;
        WaitFlag<HardEvent::S_V>(scalarReady_[bank]);
        // Small tensors are scalar-gathered, then vector-cast as a whole block.
        if constexpr (std::is_same<Beta, float>::value) DataCopy(At<float>(161408, 8), betaRaw, 8);
        else Cast(At<float>(161408, 8), betaRaw, RoundMode::CAST_NONE, count);
        if (td_.hasGama) {
            if constexpr (std::is_same<Gate, float>::value) DataCopy(At<float>(161440, 8), gateRaw, 8);
            else Cast(At<float>(161440, 8), gateRaw, RoundMode::CAST_NONE, count);
        }
        VBarrier();
        SetFlag<HardEvent::V_S>(scalarFree_[bank]);
        scalarPending_[bank] = true;
        Sync<HardEvent::V_S>(vectorToScalar_);
    }

    __aicore__ inline uint64_t StateOffset(uint64_t slot, uint64_t head)
    {
        return slot * td_.stateStrides[0] + head * td_.stateStrides[1];
    }

    __aicore__ inline void LoadState(uint64_t slot, uint64_t head)
    {
        Sync<HardEvent::V_MTE2>(taskUbFree_);
        Sync<HardEvent::MTE3_MTE2>(taskGmFree_);
        DataCopyExtParams params{128, static_cast<uint32_t>(128 * sizeof(State)),
            static_cast<uint32_t>((td_.stateStrides[2] - 128) * sizeof(State)), 0, 0};
        DataCopyPadExtParams<State> padding{false, 0, 0, 0};
        if constexpr (std::is_same<State, float>::value) {
            DataCopyPad(StateBank(0), state_[StateOffset(slot, head)], params, padding);
        } else {
            DataCopyPad(Snapshot(1), state_[StateOffset(slot, head)], params, padding);
        }
        Sync<HardEvent::MTE2_V>(ready_[0][0]);
        if constexpr (!std::is_same<State, float>::value) {
            Cast(StateBank(0), Snapshot(1), RoundMode::CAST_NONE, 16384);
            VBarrier();
        }
    }

    __aicore__ inline void DotRows(LocalTensor<float> dst, LocalTensor<float> matrix,
                                    LocalTensor<float> vector, float alpha)
    {
        auto product = At<float>(156800, 128);
        for (uint32_t row = 0; row < 128; ++row) {
            if (alpha != 1.0f) {
                Muls(product, matrix[row * 128], alpha, 128);
                VBarrier();
                Mul(product, product, vector, 128);
            } else {
                Mul(product, matrix[row * 128], vector, 128);
            }
            VBarrier();
            Add(product[64], product, product[64], 64);
            VBarrier();
            WholeReduceSum(dst[row], product[64], 64, 1, 1, 1, 8);
            VBarrier();
        }
    }

    __aicore__ inline void EmitState(uint32_t bank, uint64_t slot, uint64_t head)
    {
        Sync<HardEvent::V_MTE3>(stateReady_[bank]);
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
        VBarrier();
        uint64_t alternating[2] = {0x5555555555555555ULL, 0x5555555555555555ULL};
        Duplicate(mask, static_cast<uint16_t>(0xf000), alternating, 2, 1, 8);
        VBarrier();
        auto keyHigh = scratch;
        auto keyLow = scratch[128];
        auto deltaHigh = scratch[256];
        auto deltaLow = scratch[384];
        And(keyHigh.ReinterpretCast<uint16_t>(), key.ReinterpretCast<uint16_t>(), mask, 256);
        And(deltaHigh.ReinterpretCast<uint16_t>(), delta.ReinterpretCast<uint16_t>(), mask, 256);
        VBarrier();
        Sub(keyLow, key, keyHigh, 128);
        Sub(deltaLow, delta, deltaHigh, 128);
        VBarrier();
        auto highBroadcast = scratch[512];
        auto lowBroadcast = scratch[1536];
        Brcb(highBroadcast, deltaHigh, 16, {1, 8});
        Brcb(lowBroadcast, deltaLow, 16, {1, 8});
        VBarrier();
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
            VBarrier();
            Sub(productError, productError, product, 1024);
            for (uint32_t half = 0; half < 128; half += 64) {
                Mul(temp[half], keyHigh[half], lowBroadcast[row * 8], 64, 8, {1, 1, 0, 16, 0, 1});
            }
            VBarrier();
            Add(productError, productError, temp, 1024);
            VBarrier();
            for (uint32_t half = 0; half < 128; half += 64) {
                Mul(temp[half], keyLow[half], highBroadcast[row * 8], 64, 8, {1, 1, 0, 16, 0, 1});
            }
            VBarrier();
            Add(productError, productError, temp, 1024);
            VBarrier();
            for (uint32_t half = 0; half < 128; half += 64) {
                Mul(temp[half], keyLow[half], lowBroadcast[row * 8], 64, 8, {1, 1, 0, 16, 0, 1});
            }
            VBarrier();
            Add(productError, productError, temp, 1024);
            Add(sum, product, previous, 1024);
            VBarrier();
            Sub(recovered, sum, product, 1024);
            VBarrier();
            Sub(temp, sum, recovered, 1024);
            VBarrier();
            Sub(error, product, temp, 1024);
            VBarrier();
            Sub(temp, previous, recovered, 1024);
            VBarrier();
            Add(error, error, temp, 1024);
            VBarrier();
            Add(error, error, productError, 1024);
            VBarrier();
            Add(temp, sum, error, 1024);
            Abs(recovered, sum, 1024);
            VBarrier();
            CompareScalar(finite, recovered, 3.402823466e+38f, CMPMODE::LE, 1024);
            VBarrier();
            Select(target[row * 128], finite, temp, sum, SELMODE::VSEL_TENSOR_TENSOR_MODE, 1024);
            VBarrier();
        }
    }

    __aicore__ inline void Token(uint32_t rawBank, uint32_t token, uint64_t head,
                                  uint64_t globalToken, uint32_t src, uint32_t dst,
                                  float aLog, float bias)
    {
        auto query = Fp(0)[token * 128];
        auto key = Fp(1)[token * 128];
        auto value = Fp(2)[token * 128];
        const float rawBeta = At<float>(161408, 8).GetValue(token);
        const float rawGate = td_.hasGama ? At<float>(161440, 8).GetValue(token) : 0.0f;
        float beta = rawBeta;
        if (td_.useBetaSigmoidInKernel) {
            const float exponential = ExpScalar(beta >= 0.0f ? -beta : beta);
            beta = beta >= 0.0f ? 1.0f / (1.0f + exponential) : exponential / (1.0f + exponential);
            if (td_.allowNegEigval) beta *= 2.0f;
        }
        float alpha = 1.0f;
        if (td_.hasGama) {
            bool compensatedGate = false;
            if constexpr (std::is_same<State, float>::value) {
                compensatedGate = td_.useGateInKernel && aLog >= -80.0f && aLog <= 80.0f &&
                    rawGate >= -1.0e20f && rawGate <= 1.0e20f && bias >= -1.0e20f && bias <= 1.0e20f;
            }
            if (compensatedGate) {
                alpha = RecurrentGdnDf::RawAlpha(rawGate, aLog, bias);
            } else {
                float logGate = rawGate;
                if (td_.useGateInKernel) {
                    const float x = rawGate + bias;
                    const float absX = x >= 0.0f ? x : -x;
                    const float tail = ExpScalar(-absX);
                    const float softplus = (x > 0.0f ? x : 0.0f) + Log1pScalar(tail);
                    logGate = -ExpScalar(aLog) * softplus;
                }
                alpha = ExpScalar(logGate);
            }
        }
        if (td_.useQkL2NormInKernel) { Normalize(query); Normalize(key); }
        Muls(query, query, td_.scale, 128);
        VBarrier();
        auto source = StateBank(src);
        auto target = StateBank(dst);
        auto delta = At<float>(155776, 128);
        if constexpr (std::is_same<State, float>::value) {
            ReclaimState(dst);
            Muls(target, source, alpha, 16384);
            VBarrier();
            DotRows(delta, target, key, 1.0f);
        } else {
            DotRows(delta, source, key, alpha);
        }
        Sub(delta, value, delta, 128);
        VBarrier();
        Muls(delta, delta, beta, 128);
        VBarrier();
        auto broadcast = At<float>(157312, 1024);
        Brcb(broadcast, delta, 16, {1, 8});
        VBarrier();
        if constexpr (!std::is_same<State, float>::value) {
            ReclaimState(dst);
            Muls(target, source, alpha, 16384);
            VBarrier();
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
        VBarrier();
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
            VBarrier();
            Cast(Snapshot(src), target, RoundMode::CAST_RINT, 16384);
            EmitState(src, static_cast<uint64_t>(writeSlot), head);
        }
        Sync<HardEvent::V_MTE3>(outReady_[outBank]);
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
            PrepareWindow(bank, batch, head, start + position, position, count);
            if (position + count < length) {
                const uint32_t nextCount = length - position - count < 8 ? length - position - count : 8;
                PrefetchQ(1 - bank, start + position + count, keyHead, nextCount);
            }
            for (uint32_t token = 0; token < count; ++token) {
                Token(bank, token, head, start + position + token, src, 1 - src, aLog, bias);
                src = 1 - src;
            }
        }
        for (uint32_t bank = 0; bank < 2; ++bank) {
            ReclaimState(bank);
            ReclaimOut(bank);
            ReclaimRaw(bank);
            if (scalarPending_[bank]) {
                WaitFlag<HardEvent::V_S>(scalarFree_[bank]);
                scalarPending_[bank] = false;
            }
        }
    }

    __aicore__ inline void ReleaseEvents()
    {
        for (uint32_t bank = 0; bank < 2; ++bank) {
            for (uint32_t i = 0; i < 3; ++i) pipe_->ReleaseEventID<HardEvent::MTE2_V>(ready_[bank][i]);
            pipe_->ReleaseEventID<HardEvent::V_MTE2>(rawFree_[bank]);
            pipe_->ReleaseEventID<HardEvent::S_V>(scalarReady_[bank]);
            pipe_->ReleaseEventID<HardEvent::V_S>(scalarFree_[bank]);
            pipe_->ReleaseEventID<HardEvent::V_MTE3>(stateReady_[bank]);
            pipe_->ReleaseEventID<HardEvent::MTE3_V>(stateFree_[bank]);
            pipe_->ReleaseEventID<HardEvent::V_MTE3>(outReady_[bank]);
            pipe_->ReleaseEventID<HardEvent::MTE3_V>(outFree_[bank]);
        }
        pipe_->ReleaseEventID<HardEvent::S_V>(scalarToVector_);
        pipe_->ReleaseEventID<HardEvent::V_S>(vectorToScalar_);
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
    TEventID ready_[2][3], rawFree_[2], scalarReady_[2], scalarFree_[2];
    TEventID stateReady_[2], stateFree_[2], outReady_[2], outFree_[2];
    TEventID scalarToVector_, vectorToScalar_, taskUbFree_, taskGmFree_;
    bool rawPending_[2] = {false, false};
    bool scalarPending_[2] = {false, false};
    bool statePending_[2] = {false, false};
    bool outPending_[2] = {false, false};
};
} // namespace RecurrentGatedDeltaRule
#endif
