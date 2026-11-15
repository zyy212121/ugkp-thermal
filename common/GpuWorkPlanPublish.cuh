#pragma once
// Startup-selected geometry: never time physical kernels to choose a width.
int publishStartupWorkPlan(DeviceState* s,
    const GasHostPolicy::Time, const GasHostPolicy::Time)
{
    if (s->fixedWorkBlockTuned) return 0;
    s->fixedWorkBlockTuned=1;
    // Widths were uploaded at creation; this flag is host-only. A full
    // publication here would overwrite device-owned runtime state.
    return 0;
}
