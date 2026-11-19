#pragma once
#include "AdvanceState.H"
#include "AdvanceExtension.H"
extern "C" int ugkwpGpuResidentStrictConfigureAdvance(void*,const ugkp::gpu::AdvanceState*,int pureGas);
extern "C" int ugkwpGpuResidentStrictEnqueueAdvance(void*,const ugkp::gpu::AdvanceRequest*);
extern "C" int ugkwpGpuResidentStrictCollectAdvance(void*,ugkp::gpu::AdvanceState*);
extern "C" int ugkwpGpuResidentStrictCollectAdvanceProgress(void*,ugkp::gpu::AdvanceState*,ugkp::gpu::AdvanceProgress*,bool (*)(const void*),const void*);
extern "C" void ugkwpGpuResidentStrictReleaseAdvance(void*) noexcept;
