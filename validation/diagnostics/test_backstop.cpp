#include "DiagnosticsHost.H"
static void returnsIncorrectly(void*,const ugkp::diagnostics::Record&,const ugkp::diagnostics::SourceDescriptor*) {}
int main(int argc,char**) {
    using namespace ugkp::diagnostics;
    if(argc>1)recordFatal(nullptr,makeRecord(Code::NonFinite));
    DeviceContext context{};reset(context,1,2);recordFatal(&context,makeRecord(Code::NegativeTemperature));
    HostSink sink; sink.fatal=returnsIncorrectly;
    dispatch(takeSnapshot(context),sink);
    return 99;
}
