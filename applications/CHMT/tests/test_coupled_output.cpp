#include "io/CoupledOutput.H"
#include "tests/TestSupport.H"
#include <sstream>
#include <limits>
using namespace chmt;
using chmt_test::check;
int main(){
    ModelConfig model;model.condensedNames={{"C0","C1"}};model.physics=chmt_test::physics();HostState state;
    state.gasMesh.volumes={1};state.solidMesh.volumes={1};
    state.gas={conservativeGas(chmt_test::gas(model.physics),1,model.physics)};
    SolidQ solid;solid.condensed[0]=1000;solid.energy=1000*condensedE(model.physics.condensed[0],300);state.solid={solid};
    state.gasMesh.owner={0};state.gasMesh.neighbour={-1};state.gasMesh.boundaryKind={BoundaryKind::Interface};
    state.solidMesh.owner={0};state.solidMesh.neighbour={-1};
    state.surface.area={1};state.surface.gasFace={0};state.surface.solidFace={0};state.surface.solidCell={0};state.surface.persistentId={11};
    ExchangePacket packet;packet.kind=ExchangeKind::GasSolid;packet.gasCell=0;packet.solidCell=0;packet.face=11;
    packet.energy=packet.conductive=-2;packet.consumerMask=ConsumeGas|ConsumeSolid;state.ledger={packet};
    using Writer=bool(*)(std::ostream&,const HostState&,const ModelConfig&,std::string&);
    const Writer writers[]={io::writeAcceptedCoupledSummary,io::writeAcceptedGasFields,io::writeAcceptedMaterialFields,io::writeAcceptedExchangeFields};
    std::string error;
    for(auto writer:writers){std::ostringstream out;check(writer(out,state,model,error),error);check(!out.str().empty(),"accepted output nonempty");}
    std::ostringstream material,exchange,summary;
    check(io::writeAcceptedMaterialFields(material,state,model,error),error);
    check(material.str().find("mass_C0")!=std::string::npos&&material.str().find("mass_pore_S0")!=std::string::npos,"material species inventories exported");
    check(io::writeAcceptedExchangeFields(exchange,state,model,error),error);
    check(exchange.str().find("consumer_mask")!=std::string::npos&&exchange.str().find("conductive")!=std::string::npos,"exchange owners and energy decomposition exported");
    check(io::writeAcceptedCoupledSummary(summary,state,model,error),error);
    check(summary.str().find("commit_sequence")!=std::string::npos,"summary synchronized commit sequence");
    for(int bad=0;bad<4;++bad){auto invalid=state;
        if(bad==0)invalid.commitSequence=1;
        if(bad==1)invalid.ledger[0].consumerMask=ConsumeGas;
        if(bad==2)invalid.time=std::numeric_limits<double>::quiet_NaN();
        if(bad==3)invalid.filmStorage.resize(1);
        for(auto writer:writers){std::ostringstream out;check(!writer(out,invalid,model,error),"every writer rejects unaccepted state");check(out.str().empty(),"failed output leaves no partial field");}
    }
    auto invalid=state;invalid.solid[0].energy=std::numeric_limits<double>::quiet_NaN();std::ostringstream out;
    check(!io::writeAcceptedMaterialFields(out,invalid,model,error)&&out.str().empty(),"material validation precedes writing");
    std::cout<<"PASS: coupled output writers export material/exchange and reject unsynchronized state.\n";
}
