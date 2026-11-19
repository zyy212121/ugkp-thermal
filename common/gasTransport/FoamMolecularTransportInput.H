#pragma once
#include "FoamMolecularTransportIO.H"
namespace Foam {
class MolecularTransportInput {
    ugkwp::MolecularTransportConfiguration config_;
    autoPtr<IOdictionary> identity_;
public:
    MolecularTransportInput(const Time& time,const fvMesh& mesh,const ugkwp::GasModelConfiguration& gasModel){
        try{
            config_=readFoamMolecularTransport(time,mesh,gasModel.speciesNames);
            if(gasModel.mode!=ugkwp::GasMode::SingleLegacy)
                ugkwp::validateTransportMolarMasses(config_,gasModel.species);


            identity_.reset(new IOdictionary(IOobject("gasTransportIdentity",time.timeName(),mesh,IOobject::NO_READ,IOobject::AUTO_WRITE)));
            identity_().set("transportHash",Foam::word(("v1_"+std::to_string(config_.hash))));
        }catch(const std::exception& e){FatalErrorInFunction<<e.what()<<exit(FatalError);}
    }
    const ugkwp::MolecularTransportConfiguration& config()const{return config_;}
};
}
