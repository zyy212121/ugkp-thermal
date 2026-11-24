

namespace {
struct WindowReceipt {
    MaterialWindowCommit window{};
    std::uint64_t gasGeometry=0,solidGeometry=0;
    Real dt=0,nextDt=0;
};
__global__ void closeCoupledWindow(WindowReceipt* slot,WindowReceipt* completed,
    WindowReceipt base,WindowReceipt endpoint,
    bool continuing,ugkp::diagnostics::DeviceContext* diagnostics) {
    *completed={};
    if(ugkp::diagnostics::hasFatal(diagnostics))return;
    const auto previous=continuing?*slot:base;
    const auto step=previous.window.acceptedStep+1,seq=previous.window.sequence+1;
    if(previous.window.time!=base.window.time||previous.window.sequence!=base.window.sequence||
       !(endpoint.window.time>previous.window.time)||!::isfinite(endpoint.nextDt)||!(endpoint.nextDt>0)||
       endpoint.window.sequence!=seq||
       (endpoint.window.committed&&endpoint.window.acceptedStep!=step)){
        ugkp::diagnostics::recordFatal(diagnostics,ugkp::diagnostics::makeRecord(ugkp::diagnostics::Code::InvalidPhysicalState,
            26041,26,0,ugkp::diagnostics::EntityKind::None,-1,-1,endpoint.window.time,previous.window.time));
        return;
    }
    endpoint.window.acceptedStep=step;endpoint.window.committed=true;
    endpoint.dt=endpoint.window.time-previous.window.time;
    *slot=endpoint;
    *completed=endpoint;
}
}
