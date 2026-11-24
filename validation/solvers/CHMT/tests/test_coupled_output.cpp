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
    using Writer=bool(*)(std::ostream&,const HostState&,const ModelConfig&,std::string&);
    const Writer writers[]={io::writeAcceptedCoupledSummary,io::writeAcceptedGasFields,io::writeAcceptedMaterialFields};
    std::string error;
    for(auto writer:writers){std::ostringstream out;check(writer(out,state,model,error),error);check(!out.str().empty(),"accepted output nonempty");}
    std::ostringstream material,heat,summary;
    check(io::writeAcceptedMaterialFields(material,state,model,error),error);
    check(material.str().find("mass_C0")!=std::string::npos&&material.str().find("mass_pore_S0")!=std::string::npos,"material species inventories exported");
    check(io::writeAcceptedWallHeatFields(heat,state,0.25,{{2,3,4}},error),error);
    check(heat.str().find("convection_J")!=std::string::npos&&heat.str().find(",2,3,4")!=std::string::npos,
        "wall energy components exported");
    check(io::writeAcceptedCoupledSummary(summary,state,model,error),error);
    check(summary.str().find("commit_sequence")!=std::string::npos,"summary synchronized commit sequence");
    check(summary.str().find("sst_chemistry_integral_audit_collection")!=std::string::npos&&summary.str().find("NOT_COLLECTED_LEGACY_CHECKPOINT_SLOTS_ONLY")!=std::string::npos,"legacy audit payload is not reported as current collection");
    for(int bad=0;bad<4;++bad){auto invalid=state;
        if(bad==0)invalid.commitSequence=1;
        if(bad==1)invalid.nextDt=1;
        if(bad==2)invalid.time=std::numeric_limits<double>::quiet_NaN();
        if(bad==3)invalid.filmStorage.resize(1);
        for(auto writer:writers){std::ostringstream out;check(writer(out,invalid,model,error),"every writer consumes supplied metadata");check(!out.str().empty(),"supplied output is written");}
    }
    auto invalid=state;invalid.solid[0].energy=std::numeric_limits<double>::quiet_NaN();std::ostringstream out;
    check(!io::writeAcceptedMaterialFields(out,invalid,model,error)&&!error.empty(),"invalid material energy rejected");
    std::ostringstream failed;failed.setstate(std::ios::badbit);
    check(!io::writeAcceptedMaterialFields(failed,state,model,error)&&!error.empty(),"material stream failure propagates");
    std::cout<<"PASS: coupled output writers export material/wall heat and consume supplied metadata.\n";
}
