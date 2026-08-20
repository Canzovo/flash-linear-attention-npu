// Canonical aclnn route contract for RecurrentKda.
// Runtime inputs are defined by ../recurrent_kda.json.
#include "aclnn_recurrent_kda.h"

namespace {
[[maybe_unused]] auto *const kGetWorkspace = &aclnnRecurrentKdaGetWorkspaceSize;
[[maybe_unused]] auto *const kRun = &aclnnRecurrentKda;
} // namespace
