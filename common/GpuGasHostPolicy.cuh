#pragma once

// Host-only policies: retain the application's time and RK storage precision.
// These types never enter DeviceState or a device-kernel parameter list.
// The caller supplies its own host-state type; neither policy owns gas or
// particle storage. CUDA consumers already compile as C++17.
template<class TimeValue, class WeightValue>
struct GasHostWithoutWallEnergy
{
    using Time = TimeValue;
    using Weight = WeightValue;

    template<class HostState>
    static int applyFaceSources(HostState*, const Time, const Time) { return 0; }

    template<class HostState>
    static Time firstLedgerDt(const HostState*, const Time) { return Time(0); }
    static Time finalLedgerDt(const Time) { return Time(0); }
    template<class HostState>
    static int accumulateWallEnergy(HostState*, const Time) { return 0; }
};

template
<
    class TimeValue,
    class WeightValue,
    auto AccumulateWallEnergy
>
struct GasHostWithWallEnergy
{
    using Time = TimeValue;
    using Weight = WeightValue;

    template<class HostState>
    static int applyFaceSources(HostState*, const Time, const Time) { return 0; }

    template<class HostState>
    static Time firstLedgerDt(const HostState* s, const Time dt)
    {
        return dt*(s->hostGasTimeIntegrator == 2 ? 0.5 : 1.0/6.0);
    }

    static Time finalLedgerDt(const Time dt)
    {
        return dt*(2.0/3.0);
    }

    template<class HostState>
    static int accumulateWallEnergy(HostState* s, const Time ledgerDt)
    {
        return AccumulateWallEnergy(s, ledgerDt);
    }
};
