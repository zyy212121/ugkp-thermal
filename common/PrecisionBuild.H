#pragma once
// Direct source builds retain the original FP64 path. The dual-build tool
// emits this header separately inside each materialized source tree.
namespace ugkp { namespace precision {
inline constexpr int compiledBits = 64;
inline constexpr const char* policy = "legacy64-v1";
}}
