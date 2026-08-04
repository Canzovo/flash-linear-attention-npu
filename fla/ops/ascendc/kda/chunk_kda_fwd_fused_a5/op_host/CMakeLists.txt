add_op_to_compiled_list()
set(chunk_kda_fwd_fused_a5_depends
    "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h;fla/ops/ascendc/kda/chunk_kda_fwd_prepare;fla/ops/ascendc/kda/chunk_kda_fwd_post_wu;fla/ops/ascendc/kda/chunk_kda_fwd_finalize"
    CACHE STRING "Kernel source dependencies for the private fused KDA forward path" FORCE)
if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        chunk_kda_fwd_fused_a5_def.cpp
    )
endif()

add_modules_sources(OPTYPE chunk_kda_fwd_fused_a5 ACLNNTYPE aclnn_exclude)
add_ops_compile_options(
    OP_NAME ChunkKdaFwdFusedA5
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
)
if("${ASCEND_COMPUTE_UNIT}" STREQUAL "ascend950")
    add_ops_compile_options(
        OP_NAME ChunkKdaFwdFusedA5
        COMPUTE_UNIT Ascend950PR_9599
        OPTIONS -DENABLE_CV_COMM_VIA_SSBUF=true
    )
endif()
