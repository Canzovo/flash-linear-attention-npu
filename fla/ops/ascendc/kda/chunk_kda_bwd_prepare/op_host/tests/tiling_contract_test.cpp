#include <cassert>
#include <iostream>
#include "tiling_base/tiling_templates_registry.h"
#undef OP_LOGD
#undef OP_LOGE
#define OP_LOGD(...) ((void)0)
#define OP_LOGE(...) ((void)0)
#include "../op_tiling/chunk_kda_bwd_prepare_tiling_processor.h"

int main()
{
    constexpr const char *layouts[] = {"BNSD", "BSND", "NTD", "TND"};
    constexpr uint64_t keys[] = {0, 1, 6, 7};
    for (uint32_t layout = 0; layout < 4; ++layout) {
        const bool packed = layout >= 2, seq = (layout & 1) != 0;
        gert::StorageShape aqk, value, state;
        if (packed) {
            aqk = seq ? gert::StorageShape({65, 5, 64}, {65, 5, 64})
                      : gert::StorageShape({5, 65, 64}, {5, 65, 64});
            value = seq ? gert::StorageShape({65, 5, 128}, {65, 5, 128})
                        : gert::StorageShape({5, 65, 128}, {5, 65, 128});
            state = gert::StorageShape({2, 5, 128, 128}, {2, 5, 128, 128});
        } else {
            aqk = seq ? gert::StorageShape({2, 65, 5, 64}, {2, 65, 5, 64})
                      : gert::StorageShape({2, 5, 65, 64}, {2, 5, 65, 64});
            value = seq ? gert::StorageShape({2, 65, 5, 128}, {2, 65, 5, 128})
                        : gert::StorageShape({2, 5, 65, 128}, {2, 5, 65, 128});
            state = gert::StorageShape({2, 2, 5, 128, 128}, {2, 2, 5, 128, 128});
        }
        gert::StorageShape cu({3}, {3}), indices({4}, {4});
        for (bool direction : {false, true}) {
            KDA::ChunkKdaBwdPrepareTilingData data{};
            optiling::ChunkKdaBwdPrepareTilingContext ctx{
                "contract", &aqk, &value, &value, &state,
                packed ? &cu : nullptr, packed ? &indices : nullptr,
                ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16,
                0.125, 64, direction, layouts[layout], 24, 4096};
            optiling::ChunkKdaBwdPrepareTilingProcessor processor(ctx, data);
            assert(processor.Process() == ge::GRAPH_SUCCESS);
            assert(processor.GetTilingKey() == keys[layout]);
            assert(processor.GetBlockDim() == (packed ? 4 : 8));
            assert(processor.GetWorkspaceSize() == 4096 + processor.GetBlockDim() * 65536);
            assert(data.NV == 5 && data.T == 65 && data.B == (packed ? 1 : 2));
            std::cout << layouts[layout] << " state=" << direction << " key="
                      << processor.GetTilingKey() << " blocks=" << processor.GetBlockDim() << '\n';
            ctx.layout = "INVALID";
            assert(processor.Process() == ge::GRAPH_FAILED);
            ctx.layout = layouts[layout];
            ctx.chunkSize = 32;
            assert(processor.Process() == ge::GRAPH_FAILED);
            ctx.chunkSize = 64;
            ctx.cuSeqlensShape = packed ? nullptr : &cu;
            assert(processor.Process() == ge::GRAPH_FAILED);
        }
    }
}
