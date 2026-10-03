#pragma once

// Mobile-packing launch/error protocol. Requires GasHostPolicy::Time/Weight,
// DeviceState and the mobile-packing kernels. The cooperative argument keeps
// the original time type; this header does not own the device projection math.
int applyMobilePackingProjection
(
    DeviceState* s,
    const GasHostPolicy::Time dt,
    const int block
)
{
    if (!s->jammingPressureEnabled || s->particleWorkGrid <= 0)
    {
        return 0;
    }
    if (!(dt > static_cast<GasHostPolicy::Weight>(0.0)) || !std::isfinite(dt))
    {
        setLastErrorText("invalid mobile packing-projection dt");
        return 1;
    }
    const int cellGrid = (s->nCells + block - 1)/block;
    cudaError_t err = cudaSuccess;

#define PACKING_LAUNCH(CALL, NAME) \
    CALL; \
    err = cudaGetLastError(); \
    if (err != cudaSuccess) \
    { \
        setLastError(NAME, err); \
        return 1; \
    }

    PACKING_LAUNCH
    (
        (clearMobilePackingMomentsKernel<<<cellGrid, block>>>(s->deviceState)),
        "clear mobile packing moments launch"
    );
    PACKING_LAUNCH
    (
        (clearMobilePackingActivityCountsKernel<<<1, 1>>>(s->deviceState)),
        "clear mobile packing activity counts launch"
    );
    PACKING_LAUNCH
    (
        (accumulateMobilePackingMomentsKernel<<<s->particleWorkGrid, s->particleBlockThreads>>>
        (s->deviceState)),
        "accumulate mobile packing moments launch"
    );
    PACKING_LAUNCH
    (
        (normalizeMobilePackingMomentsKernel<<<cellGrid, block>>>(s->deviceState)),
        "normalize mobile packing moments launch"
    );
    PACKING_LAUNCH
    (
        (prepareMobilePackingProjectionKernel<<<cellGrid, block>>>
        (s->deviceState, dt)),
        "prepare mobile packing projection launch"
    );

    if (s->mobilePackingCooperativeGrid <= 0)
    {
        setLastErrorText("mobile packing cooperative grid is unavailable");
        return 1;
    }
    GasHostPolicy::Time kernelDt = dt;
    void* cooperativeArguments[] = {&s->deviceState, &kernelDt};
    err = cudaLaunchCooperativeKernel
    (
        reinterpret_cast<const void*>
        (completeMobilePackingProjectionCooperativeKernel),
        dim3(s->mobilePackingCooperativeGrid),
        dim3(block),
        cooperativeArguments,
        0,
        nullptr
    );
    if (err != cudaSuccess)
    {
        setLastError("complete mobile packing projection launch", err);
        return 1;
    }
#undef PACKING_LAUNCH
    return 0;
}
