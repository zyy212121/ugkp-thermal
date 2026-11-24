from test_diffusive_courant_transport import execute, precision_tree
from test_legacy_ale_transport import fixture as legacy_fixture
from test_mixture_transport import fixture as mixture_fixture
from test_current_gas_courant import launches, CUDA
from validation.tools.paths import SOURCE_ROOT


def test_preparation_lease_and_state_update(tmp_path, precision_tree):
    helper = launches((SOURCE_ROOT / "common/GpuGasCourant.cuh").read_text())
    execute(tmp_path, precision_tree, legacy_fixture().split("void moving(", 1)[0] + CUDA + helper + r"""
struct TimedHost:Host {
    ugkwp::GasResidentInterval gasResidentInterval;
    ugkwp::GasBoundaryLayerState<Real> gasBoundaryLayer;
    TimedHost* deviceState=this;
    ugkwp::GasTransportPreparation* gasCourantPreparation=nullptr;
};
int main(){
    for(int model=1;model<=3;++model){
        TimedHost s;initialise(s);s.hostTurbulenceModel=s.turbulenceModel=model;s.sstConfigured=model==3;
        s.lesDeltaCoeff=1;s.smagorinskyCs=Real(.17);s.waleCw=Real(.325);
        s.gasResidentInterval={.001,2.,2.001,2.0005,true,false};
        ugkwp::GasTransportPreparation prepared;s.gasCourantPreparation=&prepared;
        using P=ugkwp::GasTransportPreparation;
        ck(ugkwp::gasTransportPreparationEligible(s),"stationary nonreactive state excluded");
        s.gasSpecies.mode=ugkwp::GasMode::MixtureChemistry;
        ck(!ugkwp::gasTransportPreparationEligible(s),"chemistry half-step reused old preparation");
        s.gasSpecies.mode=ugkwp::GasMode::SingleLegacy;
        s.gasGeometry.enabled=true;
        ck(!ugkwp::gasTransportPreparationEligible(s),"moving geometry reused old preparation");
        s.gasGeometry.enabled=false;
        s.gasBoundaryLayer.enabled=true;
        ck(!ugkwp::gasTransportPreparationEligible(s),"wall-layer update reused old preparation");
        s.gasBoundaryLayer.enabled=false;
        for(int c=0;c<2;++c){s.rhoUx[c]=Real(2+4*c);s.rhoE[c]=Real(250000)+s.rhoUx[c]*s.rhoUx[c]/Real(4);}
        ck(computeGasCurrentCourantField(&s,.001,0.,4,4,nullptr,true,&prepared)==0,"CFL failed");
        ck(prepared.time==2.,"preparation stamped wrong boundary time");
        ck(ugkwp::gasTransportPrepared(s,&prepared,P::gradient|P::eddy),"preparation unavailable");
        const Real beforeNut=s.nut[0],beforeGrad=s.gradUxX[0];
        ck(prepareGasTransportCoefficients(&s,4,nullptr,false,false,true,&prepared)==0,"retained preparation failed");
        ck(s.nut[0]==beforeNut&&s.gradUxX[0]==beforeGrad,"retained preparation changed");
        s.gasResidentInterval.begin=3.;
        for(int c=0;c<2;++c)s.Ux[c]=Real(7+12*c);
        ck(!ugkwp::gasTransportPrepared(s,&prepared,P::gradient),"changed boundary time reused preparation");
        ck(prepareGasTransportCoefficients(&s,4,nullptr,false,false,true,&prepared)==0,"changed-time preparation failed");
        ck(s.gradUxX[0]!=beforeGrad,"changed-time gradients not refreshed");
        s.gasResidentInterval.begin=2.;
        ck(computeGasCurrentCourantField(&s,.001,0.,4,4,nullptr,true,&prepared)==0,"second CFL failed");
        for(int f=0;f<3;++f){
            s.gasPhiRho[f]=s.gasPhiRhoUx[f]=s.gasPhiRhoUy[f]=s.gasPhiRhoUz[f]=s.gasPhiRhoE[f]=Real(0);
        }
        emulateLaunch([&]{applyGasFluxDivergenceByCellKernel(&s,0.,false,&prepared);},1,4);
        ck(prepared.fields==0,"physical update retained old derivatives for next RK stage");
        for(int c=0;c<2;++c)s.Ux[c]=Real(13+15*c);
        ck(prepareGasTransportCoefficients(&s,4,nullptr,false,false,true,&prepared)==0,"next RK preparation failed");
        ck(s.gradUxX[0]!=beforeGrad,"RK stage reused previous state");
    }
}
""")


def test_inlet_outlet_zero_flux_invalidates_sst_only(tmp_path, precision_tree):
    execute(tmp_path, precision_tree, mixture_fixture().split("void transport(", 1)[0] + r"""
int main(){
    State s;initialise(s);s.turbulenceModel=3;s.sstConfigured=1;
    using P=ugkwp::GasTransportPreparation;
    P prepared;prepared.fields=P::primitive|P::gradient|P::limiter|P::eddy|P::sst;
    s.riemannBoundaryKind[2]=0;s.sstBoundaryKMode[2]=s.sstBoundaryOmegaMode[2]=2;
    Real scales[4]={1,0,1,1};s.gasSpecies.positivityScale=scales;
    s.gasFluxPositivityScale[0]=s.gasFluxPositivityScale[1]=1;
    s.gasPhiRho[2]=-1;s.gasSpecies.flux[2]=1;s.gasSpecies.flux[5]=-2;
    threadIdx.x=2;
    applyGasFluxPositivityScaleKernel(&s,false,&prepared);
    ck(s.gasPhiRho[2]==0,"positive inventory limiter did not exercise flow-direction change");
    ck(!(prepared.fields&P::sst),"inletOutlet retained F1 from inflow after flow became zero");
    ck((prepared.fields&(P::primitive|P::gradient|P::eddy))==(P::primitive|P::gradient|P::eddy),"unaffected preparation discarded");
}
""")


def test_current_boundary_is_published_before_sst_preparation(tmp_path, precision_tree):
    helper = launches((SOURCE_ROOT / "common/GpuGasCourant.cuh").read_text())
    execute(tmp_path, precision_tree, legacy_fixture().split("void moving(", 1)[0] + CUDA + helper + r"""
template<class Term>auto tableTerm(double time,Real value,int)->decltype(Term::table(time,value)){return Term::table(time,value);}
template<class Term>Term tableTerm(double time,Real value,long){return {time,value};}
struct TimedHost:Host {
    ugkwp::GasResidentInterval gasResidentInterval;
    TimedHost* deviceState=this;
};
int main(){
    TimedHost s;initialise(s);s.hostTurbulenceModel=s.turbulenceModel=3;s.sstConfigured=1;
    s.sstWallTreatment=1;s.sstWallCmu=Real(.09);s.sstWallKappa=Real(.41);s.sstWallE=Real(9.8);
    s.gasMu=Real(.00001);
    s.riemannBoundaryKind[1]=s.gasBoundaryKind[1]=2;
    s.riemannBoundaryUFix[1]=s.gasBoundaryUFix[1]=1;
    s.riemannBoundaryUy[1]=s.gasBoundaryUy[1]=0;
    ugkp::boundary::Face faces[3];faces[1].function[ugkp::boundary::Uy]=0;
    ugkp::boundary::Function function;function.kind=ugkp::boundary::FunctionKind::Table;function.count=2;
    using Term=ugkp::boundary::Term;Term terms[2]={tableTerm<Term>(0,Real(0),0),tableTerm<Term>(10,Real(100),0)};
    s.gasBoundaryProgram={faces,&function,terms,3};
    s.gasResidentInterval={1,2,3,2.5,true,false};
    ugkwp::GasTransportPreparation prepared;
    ck(computeGasCurrentCourantField(&s,.001,0.,4,4,nullptr,true,&prepared)==0,"time-varying boundary CFL failed");
    near(s.riemannBoundaryUy[1],Real(20),"CFL used endpoint instead of accepted-state boundary");
    near(s.omega[0],sstDynamicOmegaWallValue(s,1,0),"SST preparation used previous boundary values");
    ck(prepared.time==2.,"current-state preparation has wrong clock");
}
""")
