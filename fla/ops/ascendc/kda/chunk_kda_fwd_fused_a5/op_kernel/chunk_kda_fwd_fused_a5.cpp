#include "kernel_operator.h"
#include "../../chunk_kda_fwd_prepare/op_kernel/chunk_kda_fwd_prepare_kernel.hpp"

#if __has_include("../../../gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/op_kernel/chunk_gated_delta_rule_fwd_h_struct.h")
#include "../../../gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/op_kernel/chunk_gated_delta_rule_fwd_h_struct.h"
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "../../../gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/op_kernel/arch35/gemm/kernel/gdn_fwd_h_kernel.hpp"
#else
#include "../../../gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/op_kernel/gemm/kernel/gdn_fwd_h_kernel.hpp"
#endif
#else
#include "../../chunk_gated_delta_rule_fwd_h/op_kernel/chunk_gated_delta_rule_fwd_h_struct.h"
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "../../chunk_gated_delta_rule_fwd_h/op_kernel/arch35/gemm/kernel/gdn_fwd_h_kernel.hpp"
#else
#include "../../chunk_gated_delta_rule_fwd_h/op_kernel/gemm/kernel/gdn_fwd_h_kernel.hpp"
#endif
#endif
#include "../../chunk_kda_fwd_finalize/op_kernel/chunk_kda_fwd_finalize_kernel.hpp"
#include "chunk_kda_fwd_recurrent_a5.hpp"
#include "lib/matmul_intf.h"

namespace KdaFusedA5 {

template <bool SAFE_GATE, typename T, typename GK_T, typename BETA_T,
          typename TilingData>
__aicore__ inline void Run(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR gk, GM_ADDR rawG,
    GM_ADDR aLog, GM_ADDR dtBias, GM_ADDR beta, GM_ADDR initialState,
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR attnOut,
    GM_ADDR finalState, GM_ADDR aqk, GM_ADDR akk, GM_ADDR w, GM_ADDR u,
    GM_ADDR qg, GM_ADDR kg, GM_ADDR vNew, GM_ADDR h,
    GM_ADDR userWorkspace, const TilingData &tiling, AscendC::TPipe &pipe)
{
    GM_ADDR qgScaled = userWorkspace + tiling.qgScaledOffset;
    KdaPrepare::RunChunkKdaPrepare<SAFE_GATE, T, GK_T, BETA_T,
        TilingData, 64, 128, 128>(
        q, k, v, gk, rawG, aLog, dtBias, beta, initialState,
        cuSeqlens, chunkIndices, aqk, akk, qg, qgScaled, w, u, kg,
        userWorkspace, tiling, pipe, tiling.storeQG);

    AscendC::SyncAll<false>();
    pipe.Reset();

    if (!tiling.fusePostWu) {
        KdaPostWu::RunChunkKdaPostWu<T, GK_T, BETA_T>(
            q, k, v, gk, beta, initialState, cuSeqlens, chunkIndices,
            w, akk, u, w, u, kg, vNew, userWorkspace, tiling, pipe);
        AscendC::SyncAll<false>();
        pipe.Reset();
    }

    ChunkKdaFwdRecurrentA5<T, GK_T, TilingData> recurrent;
    recurrent.Init(
        gk, initialState, attnOut, finalState, aqk, w, u, qgScaled,
        kg, vNew, h, tiling);
    recurrent.Process();
}

} // namespace KdaFusedA5

extern "C" __global__ __aicore__ void chunk_kda_fwd_fused_a5(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR gk, GM_ADDR raw_g,
    GM_ADDR a_log, GM_ADDR dt_bias, GM_ADDR beta, GM_ADDR initial_state,
    GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR attn_out,
    GM_ADDR final_state, GM_ADDR aqk, GM_ADDR akk, GM_ADDR w, GM_ADDR u,
    GM_ADDR qg, GM_ADDR kg, GM_ADDR v_new, GM_ADDR h,
    GM_ADDR workspace, GM_ADDR tiling)
{
    GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
    GET_TILING_DATA_WITH_STRUCT(ChunkKdaFwdFusedA5TilingData, tilingData, tiling);
    if (TILING_KEY_IS(2)) {
        KERNEL_TASK_TYPE(2, KERNEL_TYPE_MIX_AIC_1_2);
        AscendC::TPipe pipe;
        if (tilingData.safeGate) {
            KdaFusedA5::Run<true, DTYPE_Q, DTYPE_GK, DTYPE_BETA>(
                q, k, v, gk, raw_g, a_log, dt_bias, beta, initial_state,
                cu_seqlens, chunk_indices, attn_out, final_state, aqk, akk,
                w, u, qg, kg, v_new, h, userWorkspace, tilingData, pipe);
        } else {
            KdaFusedA5::Run<false, DTYPE_Q, DTYPE_GK, DTYPE_BETA>(
                q, k, v, gk, raw_g, a_log, dt_bias, beta, initial_state,
                cu_seqlens, chunk_indices, attn_out, final_state, aqk, akk,
                w, u, qg, kg, v_new, h, userWorkspace, tilingData, pipe);
        }
    }
}
