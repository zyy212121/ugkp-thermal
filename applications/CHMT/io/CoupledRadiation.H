#pragma once
#include "io/MultirateEvolution.H"
#include "gpu/RadiationWarmStart.H"
#include "fvMesh.H"
#include "Time.H"
#include <memory>
namespace chmt {

class CoupledRadiation final:public WindowRadiation {
    struct Data;std::unique_ptr<Data> data_;
public:
    CoupledRadiation(Foam::Time&,Foam::fvMesh&,const ModelConfig&,const HostState&);
    ~CoupledRadiation();
    void freezeWarmStart(ugkp::gpu::RadiationWarmStart&,void* nextWriter) const;
    bool readWarmStart(const std::string&);
    bool writeWarmStart(const std::string&) const;
    bool prepareParent(Backend&,const WallProgram&,const HostState&,std::string&) override;
    bool advance(Backend&,const WallProgram&,HostState&,std::string&) override;
};
}
