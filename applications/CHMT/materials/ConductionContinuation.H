#ifndef CHMT_MATERIALS_CONDUCTIONCONTINUATION_H
#define CHMT_MATERIALS_CONDUCTIONCONTINUATION_H
#include "materials/CpuMaterialDriver.H"
namespace chmt { namespace detail {
// Caller-private completion route; never a configuration or user selector.
struct ConductionContinuation {
    static bool predict(const CpuMaterialDriver& driver,const HostState& base,const CouplingInterval& interval,
        WallProgram& output,std::string& error,CpuMaterialReport* report) {
        return driver.predictWallImpl(base,interval,output,error,report,true);
    }
    static bool complete(CpuMaterialDriver& driver,const HostState& base,const HostState& gas,
        const IntervalHistory& history,const CpuMaterialControls& controls,HostState& output,
        WallProgram& corrected,CpuMaterialReport& report,std::string& error,const WallProgram* executed) {
        return driver.advanceCandidateImpl(base,gas,history,controls,output,corrected,report,error,executed,true);
    }
};
}}
#endif
