import pytest
from test_mixture_state import compile_probe
from test_mixture_transport import fixture


def wall_fixture():
    source=fixture()
    old='struct State:ugkwp::GasStateView<Real,Time,ugkwp::SstCoefficients>{ugkwp::GasSpeciesState<Real,2> gasSpecies;};'
    new='#include "gasTransport/GasBoundaryLayerModelState.H"\nstruct State:ugkwp::GasStateView<Real,Time,ugkwp::SstCoefficients>{ugkwp::GasSpeciesState<Real,2> gasSpecies;ugkwp::GasBoundaryLayerModelState<Real,2> gasBoundaryLayerModel;};'
    assert source.count(old)==1
    return source.replace(old,new)


@pytest.mark.parametrize('native',[False,True])
@pytest.mark.parametrize('route',[0,1,2])
def test_selected_wall_slot_requires_allocated_extent(tmp_path,native,route):
    code=r"""
int main(){for(int mode=0;mode<3;++mode){
 State s;initialise(s);s.gasReconstruction=0;s.riemannBoundaryKind[1]=2;
 const int count=mode==2?0:1,slot=mode==0?count:mode==1?count+4:0;
 int slots[3]={-1,slot,-1};auto&w=s.gasBoundaryLayer;w.enabled=true;w.count=count;w.faceSlot=slots;
 std::vector<ugkwp::GasBoundaryLayerExchange<Real>> exchanges(count);
 std::vector<ugkwp::gaswall::WallInput<Real,2>> inputs(count);
 std::vector<Real> species(2*count,Real(3.5));
 for(int k=0;k<count;++k){exchanges[k].ready=exchanges[k].massReady=true;exchanges[k].matchingPressure=s.p[0];inputs[k].massFlux[0]=.25;inputs[k].massFlux[1]=.75;}
 w.exchange=ROUTE==2?nullptr:exchanges.data();w.speciesFlux=species.data();
 s.gasBoundaryLayerModel.useSuppliedWallState=ROUTE==2;s.gasBoundaryLayerModel.input=inputs.data();
 auto inventory=[&](){return std::vector<Real>{s.rho[0],s.rho[1],s.rhoE[0],s.rhoE[1],s.gasSpecies.rho[0],s.gasSpecies.rho[1],s.gasSpecies.rho[2],s.gasSpecies.rho[3],s.gasSpecies.flux[0],s.gasSpecies.flux[1],s.gasSpecies.flux[2],s.gasSpecies.flux[3],s.gasSpecies.flux[4],s.gasSpecies.flux[5]};};
 const auto before=inventory();Real out[5]={11,12,13,14,15};
 const bool ok=computeRiemannGasFaceFluxDevice<false,MASS_ONLY>(s,1,out[0],out[1],out[2],out[3],out[4],NATIVE);
 ck(!ok&&s.gasSpecies.faceStatus[1]==int(ugkwp::GasTransportCode::InvalidStorage),"unavailable selected slot did not report storage failure");
 ck(before==inventory(),"slot resource failure changed conserved fields or species flux");
 for(Real value:out)ck(value==0,"slot resource failure published partial face flux");
}}
""".replace('ROUTE',str(route)).replace('MASS_ONLY','true' if route else 'false').replace('NATIVE','true' if native else 'false')
    compile_probe(tmp_path,wall_fixture()+code,sanitizers=True)


@pytest.mark.parametrize('native',[False,True])
@pytest.mark.parametrize('route',[0,1,2])
def test_last_allocated_wall_slot_preserves_flux_and_optional_resources(tmp_path,native,route):
    code=r"""
int main(){State s;initialise(s);s.gasReconstruction=0;s.riemannBoundaryKind[1]=2;
 int slots[3]={-1,1,-1};auto&w=s.gasBoundaryLayer;w.enabled=true;w.count=2;w.faceSlot=slots;
 ugkwp::GasBoundaryLayerExchange<Real> exchanges[2];
 ugkwp::gaswall::WallInput<Real,2> inputs[2];Real species[4]={99,3.5,101,7.25};
 auto&e=exchanges[1];e.ready=e.massReady=true;e.matchingPressure=s.p[0];e.mass=2.25;e.momentumX=4.5;e.momentumY=-6.75;e.momentumZ=9;e.energy=13.5;
 inputs[1].massFlux[0]=.25;inputs[1].massFlux[1]=.75;
 w.exchange=ROUTE==2?nullptr:exchanges;w.speciesFlux=ROUTE==0?species:nullptr;w.status=nullptr;w.sst=nullptr;w.ownerSlot=nullptr;
 s.gasBoundaryLayerModel.useSuppliedWallState=ROUTE==2;s.gasBoundaryLayerModel.input=ROUTE==2?inputs:nullptr;
 Real out[5]={11,12,13,14,15};const bool ok=computeRiemannGasFaceFluxDevice<false,MASS_ONLY>(s,1,out[0],out[1],out[2],out[3],out[4],NATIVE);
 ck(ok&&s.gasSpecies.faceStatus[1]==0,"last valid selected slot failed");
 ck(out[0]==(ROUTE==2?-s.magSf[1]:e.mass),"selected wall mass expression changed");
 if(ROUTE==0){ck(out[1]==e.momentumX&&out[2]==e.momentumY&&out[3]==e.momentumZ&&out[4]==e.energy,"selected wall momentum or energy changed");ck(s.gasSpecies.flux[1]==3.5&&s.gasSpecies.flux[4]==7.25,"selected wall species indexing changed");}
 else for(int k=1;k<5;++k)ck(out[k]==0,"mass-only consumer read unused flux");
}
""".replace('ROUTE',str(route)).replace('MASS_ONLY','true' if route else 'false').replace('NATIVE','true' if native else 'false')
    compile_probe(tmp_path,wall_fixture()+code,sanitizers=True)
