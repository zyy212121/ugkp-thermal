#pragma once
#include "FoamAdvanceBoundary.H"
#include "FoamPrecisionIO.H"

namespace ugkp { namespace gpu {
template<class Time,class Freeze,class Enqueue,class Metadata>
void writeGasOutput(Time& time,bool force,Freeze freeze,Enqueue enqueue,Metadata metadata)
{
    precision::ScopedRestartWritePrecision precisionScope;
    const std::string directory(time.timePath().c_str());
    precision::beginPrecisionWrite(directory);
    auto publishParticles=freeze();
    if(!force && foamAdvancePrefetchAllowed(time))enqueue();
    const bool fields=force ? time.writeNow() : time.write();
    publishParticles();
    const bool extra=metadata();
    if(!fields||!extra){
        throw diagnostics::RecordedFailure(diagnostics::makeRecord(diagnostics::Code::OutputData,36004,0),
            {36004,__FILE__,__func__,__LINE__,"incomplete gas output at "+directory});
    }
    precision::writeRestartPrecision(directory);
}

template<class Resident,class Time,class Field>
void mirrorGasTurbulence(Resident& resident,const Time& time,int model,Field& k,Field& omega,Field& nut)
{
    if(model==3)resident.downloadSstToHostMirror(time,k,omega,nut);
    else resident.downloadNutToHostMirror(time,nut);
}
}}
