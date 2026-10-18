#include "DiagnosticsFoam.H"
#include <cassert>
#include <string>
#include <thread>
#include <chrono>
using namespace ugkp::diagnostics;
int main(int argc, char** argv) {
    const bool mpi=argc>1 && std::string(argv[1])=="mpi";
    if(mpi)Foam::UPstream::init(argc,argv,false);
    DeviceContext context{}; reset(context, 7, 42, 5, 0.125);
    recordWarning(&context,makeRecord(Code::ConservationResidual,301,2,3,EntityKind::Cell,8,-1,0.01,0.001));
    recordFatal(&context,makeRecord(Code::NegativeTemperature,302,2,3,EntityKind::Cell,9,-1,-4));
    const auto snapshot=takeSnapshot(context,{{301,"common/source.cuh","conservation",11,"mass residual"},{302,"common/source.cuh","temperature",12,"temperature"}},"test");
    const auto sink=makeFoamSink();
    if(std::getenv("UGKP_DIAGNOSTIC_WARNINGS")) assert(!sink.warningsEnabled);
    if(mpi) {
        auto warningOnly=snapshot;warningOnly.hasFatal=false;dispatch(warningOnly,sink);
        if(!Foam::Pstream::master()) {
            Foam::FatalError.throwExceptions();
            dispatch(snapshot,sink); // MPI must terminate even in exception mode.
            return 98;
        }
        std::this_thread::sleep_for(std::chrono::seconds(10));return 97;
    }
    if(argc>1 && std::string(argv[1])=="exit") {dispatch(snapshot,sink);return 99;}
    Foam::FatalError.throwExceptions();
    try {dispatch(snapshot,sink);assert(false);} catch(const Foam::error& error) {
        assert(error.functionName()=="temperature");
        assert(error.sourceFileName()=="common/source.cuh");
        assert(error.sourceFileLineNumber()==12);
        assert(error.message().find("observed=-4")!=std::string::npos);
        assert(error.message().find("step=5")!=std::string::npos);
    }
    return 0;
}
