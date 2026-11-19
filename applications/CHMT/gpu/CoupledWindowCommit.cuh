

namespace {
struct WindowReceipt {
    MaterialWindowCommit window{};
    std::uint64_t gasGeometry=0,solidGeometry=0;
    Real dt=0,nextDt=0,radiationTime=0;
    decltype(Budget::radiation) radiation=0;
};
struct AcceptedWindowSlot {WindowReceipt receipt;Budget budget;};
__global__ void closeCoupledWindow(AcceptedWindowSlot* slot,WindowReceipt* completed,
    WindowReceipt base,WindowReceipt endpoint,Budget budget,Real externalHeat,
    bool continuing,ugkp::diagnostics::DeviceContext* diagnostics) {
    *completed={};
    if(ugkp::diagnostics::hasFatal(diagnostics))return;
    const auto previous=continuing?slot->receipt:base;
    const auto step=previous.window.acceptedStep+1,seq=previous.window.sequence+1;
    if(previous.window.time!=base.window.time||previous.window.sequence!=base.window.sequence||
       !(endpoint.window.time>previous.window.time)||!::isfinite(endpoint.nextDt)||!(endpoint.nextDt>0)||
       !::isfinite(externalHeat)||endpoint.window.sequence!=seq||
       (endpoint.window.committed&&endpoint.window.acceptedStep!=step)){
        ugkp::diagnostics::recordFatal(diagnostics,ugkp::diagnostics::makeRecord(ugkp::diagnostics::Code::InvalidPhysicalState,
            26041,26,0,ugkp::diagnostics::EntityKind::None,-1,-1,endpoint.window.time,previous.window.time));
        return;
    }
    budget.radiation+=externalHeat;
    if(!::isfinite(budget.radiation)){
        ugkp::diagnostics::recordFatal(diagnostics,ugkp::diagnostics::makeRecord(ugkp::diagnostics::Code::InvalidPhysicalState,
            26041,26,0,ugkp::diagnostics::EntityKind::None,-1,-1,budget.radiation,0));
        return;
    }
    endpoint.window.acceptedStep=step;endpoint.window.committed=true;
    endpoint.dt=endpoint.window.time-previous.window.time;endpoint.radiation=budget.radiation;
    slot->budget=budget;slot->receipt=endpoint;
    *completed=endpoint;
}
}
