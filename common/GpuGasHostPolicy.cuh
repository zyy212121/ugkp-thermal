#pragma once

// Host-only policies: retain the application's time and RK storage precision.
// These types never enter DeviceState or a device-kernel parameter list.
// Requires DeviceState; the wall-energy callback is supplied by the capable TU.
template<class TimeValue, class WeightValue>
struct GasHostWithoutWallEnergy
{
    using Time = TimeValue;
    using Weight = WeightValue;

    static Time firstLedgerDt(const DeviceState*, const Time) { return Time(0); }
    static Time finalLedgerDt(const Time) { return Time(0); }
    static int accumulateWallEnergy(DeviceState*, const Time) { return 0; }
};

template
<
    class TimeValue,
    class WeightValue,
    int (*AccumulateWallEnergy)(DeviceState*, TimeValue)
>
struct GasHostWithWallEnergy
{
    using Time = TimeValue;
    using Weight = WeightValue;

    static Time firstLedgerDt(const DeviceState* s, const Time dt)
    {
        return dt*(s->hostGasTimeIntegrator == 2 ? 0.5 : 1.0/6.0);
    }

    static Time finalLedgerDt(const Time dt)
    {
        return dt*(2.0/3.0);
    }

    static int accumulateWallEnergy(DeviceState* s, const Time ledgerDt)
    {
        return AccumulateWallEnergy(s, ledgerDt);
    }
};
