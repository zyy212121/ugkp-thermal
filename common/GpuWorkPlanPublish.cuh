#pragma once

int publishStartupWorkPlan(DeviceState* s,
    const GasHostPolicy::Time, const GasHostPolicy::Time)
{
    if (s->fixedWorkBlockTuned) return 0;
    s->fixedWorkBlockTuned=1;


    return 0;
}
