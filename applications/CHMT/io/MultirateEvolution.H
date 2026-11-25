#ifndef CHMT_IO_MULTIRATEEVOLUTION_H
#define CHMT_IO_MULTIRATEEVOLUTION_H
#include "coupling/IntervalContracts.H"
#include "materials/CpuMaterialDriver.H"

namespace chmt {
struct Backend;
struct WindowRadiation {
    virtual ~WindowRadiation()=default;

    virtual bool prepareParent(Backend&,const WallProgram&,const HostState&,std::string&){return true;}
    virtual bool advance(Backend&,const WallProgram&,HostState&,std::string&)=0;
};
struct MicrostepGeometryReport {
    std::uint64_t microSequence=0;
    Real begin=0,end=0;
    std::array<HostStageGeometry,2> gas{},solid{};
};
struct WindowEvolutionReport {
    std::uint64_t acceptedGasMicrosteps=0,materialSolves=0;
    Real nextGasDt=0,radiationInterval=0;
    bool physicalOutput=false;
    std::vector<MicrostepGeometryReport> geometry{};
};



bool advanceCoupledWindow(Backend&,CpuMaterialDriver&,const ModelConfig&,
    const CouplingControls&,const CpuMaterialControls&,Real targetTime,
    bool retainStageGeometry,HostState& accepted,WindowEvolutionReport&,std::string& error,WindowRadiation* radiation=nullptr,bool publishHostState=true,
    bool forceFinalOutput=true,bool windowOutput=false);
}
#endif
