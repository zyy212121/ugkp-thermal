#pragma once
#include "io/MultirateEvolution.H"
#include "fvMesh.H"
#include "Time.H"
#include <memory>
namespace chmt {
// Application adapter for the common DO/particle material owners.
class CoupledRadiation final:public WindowRadiation {
    struct Data;std::unique_ptr<Data> data_;
public:
    CoupledRadiation(Foam::Time&,Foam::fvMesh&,const ModelConfig&,const HostState&,int frequency);
    ~CoupledRadiation();
    bool advance(Backend&,const WallProgram&,HostState&,Real&,std::string&) override;
};
}
