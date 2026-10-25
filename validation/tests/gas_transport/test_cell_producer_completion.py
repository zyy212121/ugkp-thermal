"""B3 actual cell, RK, SST and current-Co terminal publication (CPU shim)."""
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

import pytest
from test_mixture_state import compile_probe
from test_mixture_transport import fixture
from test_legacy_ale_transport import fixture as legacy_fixture
from test_sst_ale_audit import AUDIT

BULK = {
    'stale_cell_status': r'''s.gasSpecies.cellStatus[0]=3;s.gasPhiRho[1]=-.2;s.gasSpecies.flux[1]=-.18;s.gasSpecies.flux[4]=-.02;threadIdx.x=0;applyGasFluxDivergenceByCellKernel(&s,.5);ck(s.rho[0]==1.1&&s.gasSpecies.rho[0]==.99&&s.gasSpecies.cellStatus[0]==3,"terminal cell status skipped actual inventory");''',
    'stale_face_status': r'''s.gasSpecies.faceStatus[1]=3;s.gasPhiRho[1]=-.2;s.gasSpecies.flux[1]=-.18;s.gasSpecies.flux[4]=-.02;threadIdx.x=0;applyGasFluxDivergenceByCellKernel(&s,.5);ck(s.rho[0]==1.1&&s.gasSpecies.rho[0]==.99&&s.gasSpecies.cellStatus[0]==3,"terminal face status skipped actual inventory");''',
    'species_nan_budget_prefix': r'''s.gasPhiRho[1]=-2;s.gasPhiRhoE[1]=-7;s.gasSpecies.flux[1]=-1;s.gasSpecies.flux[4]=std::numeric_limits<Real>::quiet_NaN();s.gasPhiRhoUx[1]=-3;s.gasPhiRhoUy[1]=-4;s.gasPhiRhoUz[1]=-5;const Real e=s.rhoE[0];threadIdx.x=0;applyGasFluxDivergenceByCellKernel(&s,.5);ck(s.rho[0]==2&&s.rhoUx[0]==1.5&&s.rhoUy[0]==2&&s.rhoUz[0]==2.5&&s.rhoE[0]==e+3.5,"terminal bulk writes incomplete after species numeric refusal");ck(s.gasSpecies.rho[0]==1.4&&std::isnan(s.gasSpecies.rho[2]),"terminal species candidate incomplete");ck(s.gasSpecies.cellStatus[0]!=0,"numeric failure certificate lost");''',
    'mass_nan': r'''s.gasPhiRho[1]=std::numeric_limits<Real>::quiet_NaN();s.gasPhiRhoUy[1]=-8;s.gasSpecies.flux[1]=-.3;s.gasSpecies.flux[4]=-.2;threadIdx.x=0;applyGasFluxDivergenceByCellKernel(&s,.5);ck(std::isnan(s.rho[0])&&s.rhoUy[0]==4&&s.gasSpecies.rho[0]==1.05&&s.gasSpecies.rho[2]==.2,"actual mass/species numeric tails not published");''',
    'candidate_overflow': r'''s.rhoUx[0]=std::numeric_limits<Real>::max();s.gasPhiRhoUx[1]=-std::numeric_limits<Real>::max();s.gasPhiRho[1]=-2;s.gasSpecies.flux[1]=-1.8;s.gasSpecies.flux[4]=-.2;threadIdx.x=0;applyGasFluxDivergenceByCellKernel(&s,1.);ck(std::isinf(s.rhoUx[0])&&s.rho[0]==3&&s.gasSpecies.rho[0]==2.7,"actual overflow tail not published");''',
    'zero_static_volume': r'''s.V[0]=0;s.gasPhiRho[1]=-1;s.gasSpecies.flux[1]=-.9;s.gasSpecies.flux[4]=-.1;threadIdx.x=0;applyGasFluxDivergenceByCellKernel(&s,.5);ck(std::isinf(s.rho[0])&&std::isinf(s.gasSpecies.rho[0])&&std::isnan(s.rhoE[0]),"zero supplied volume stopped actual divisions");''',
}

@pytest.mark.parametrize('scenario', BULK)
def test_actual_terminal_bulk_inventory(tmp_path, scenario):
    compile_probe(tmp_path, fixture()+'\nint main(){State s;initialise(s);'+BULK[scenario]+'}')

RK={
    'stale_status': r'''s.gasSpecies.cellStatus[0]=3;s.rho[0]=3;s.gasSpecies.rho[0]=2.7;s.gasSpecies.rho[2]=.3;s.rhoUx[0]=9;blendGasConservativeStateKernel(&s,.25,.75);ck(s.rho[0]==2.5&&s.rhoUx[0]==6.75&&s.gasSpecies.rho[0]==Real(.25)*Real(.9)+Real(.75)*Real(2.7)&&s.gasSpecies.cellStatus[0]==3,"terminal RK status suppressed actual blend");''',
    'species_nan': r'''s.gasSpecies.rho[0]=std::numeric_limits<Real>::quiet_NaN();s.rho[0]=3;s.rhoUx[0]=9;s.rhoK[0]=10;s.rhoOmega[0]=20;blendGasConservativeStateKernel(&s,.25,.75);ck(std::isnan(s.gasSpecies.rho[0])&&s.gasSpecies.rho[2]==.1&&s.rho[0]==2.5&&s.rhoUx[0]==6.75&&s.rhoK[0]==8&&s.rhoOmega[0]==16,"RK numeric refusal lost bulk/species/SST tail");''',
    'species_overflow': r'''s.gasSpecies.rho[0]=s.gasSpecies.initial[0]=std::numeric_limits<Real>::max();s.rho[0]=3;s.rhoUy[0]=9;blendGasConservativeStateKernel(&s,1.,1.);ck(std::isinf(s.gasSpecies.rho[0])&&s.rho[0]==4&&s.rhoUy[0]==9&&s.rhoK[0]==4,"RK overflow tail lost");''',
}

@pytest.mark.parametrize('scenario', RK)
def test_actual_terminal_rk_physical_channels(tmp_path, scenario):
    from test_sst_ale_audit import AUDIT as SST_AUDIT

    setup=r'''void near(Real a,Real b,const char*m){ck(std::abs(a-b)<1e-12,m);}'''
    body=r'''s.sstConfigured=1;s.rhoK[0]=2;s.rhoOmega[0]=4;audit(s);
threadIdx.x=0;saveGasConservativeStateKernel(&s);
ck(s.rhoNext[0]==s.rho[0]&&s.rhoENext[0]==s.rhoE[0]&&s.gasSpecies.initial[0]==.9&&s.rhoKInitial[0]==2&&s.rhoOmegaInitial[0]==4,"save incomplete");

'''
    compile_probe(tmp_path,fixture()+setup+SST_AUDIT+'\nint main(){State s;initialise(s);'+body+RK[scenario]+'}')

SST={
    'moving_stale_status': r'''s.gasSpecies.cellStatus[0]=3;s.sstPhiRhoK[0]=-6;s.sstPhiRhoOmega[0]=-8;s.gasPhiRho[0]=-2;const Real before=s.rhoK[0];threadIdx.x=0;applySstFluxAndSourceKernel(&s,.01);ck(s.rhoK[0]!=before&&observed.transportK[0]==.06&&observed.volume[0]==2,"SST old status skipped actual ledger/inventory");''',
    'moving_nan_flux': r'''s.sstPhiRhoK[0]=std::numeric_limits<Real>::quiet_NaN();s.sstPhiRhoOmega[0]=-8;threadIdx.x=0;applySstFluxAndSourceKernel(&s,.01);ck(std::isnan(s.rhoK[0])&&s.rhoOmega[0]!=8&&std::isnan(observed.transportK[0])&&observed.transportOmega[0]==.08&&observed.volume[0]==2,"SST numeric tail not evaluated");''',
    'moving_candidate_overflow': r'''s.rhoK[0]=std::numeric_limits<Real>::max();oldV[0]=2;newV[0]=1;threadIdx.x=0;applySstFluxAndSourceKernel(&s,.01);ck(std::isinf(s.rhoK[0])&&observed.volume[0]==1&&std::isnan(observed.constraintK[0]),"SST candidate overflow skipped audit tail");''',
}

@pytest.mark.parametrize('scenario', SST)
def test_actual_terminal_sst_updates_and_test_observation(tmp_path, scenario):
    setup=r'''s.sstConfigured=1;Real oldV[2]={1,1},newV[2]={2,2},sweeps[3]={0,0,0};moving(s,oldV,newV,sweeps,Real(.01));audit(s);'''
    compile_probe(tmp_path,legacy_fixture()+AUDIT+'\nint main(){State s;initialise(s);'+setup+SST[scenario]+'}')

def test_terminal_selected_omega_numeric_target_keeps_real_projection(tmp_path):
    from test_boundary_layer_operators import SETUP
    compile_probe(tmp_path,legacy_fixture()+AUDIT+SETUP+r'''
int main(){State s;initialise(s);wall(s);audit(s);s.gasBoundaryLayer.sst[0].ownerOmega=std::numeric_limits<Real>::infinity();threadIdx.x=0;applySstWallFunctionStateKernel(&s);ck(std::isinf(s.omega[0])&&std::isinf(s.rhoOmega[0])&&std::isinf(observed.constraintOmega[0]),"terminal wall omega actual projection missing");}
''')

@pytest.mark.parametrize('kernel',['convective','face'])
def test_current_co_numeric_geometry_is_evaluated_not_placeholder(tmp_path,kernel):
    calls={
        'convective':r'''s.gasPhiRhoE[0]=4;s.gasPhiRhoE[1]=2;computeGasConvectiveCourantByCellKernel(&s,.1);ck(s.gasFluxPositivityScale[0]==Real(.5)*Real(.1)*Real(6)/Real(2),"numeric geometry used placeholder instead of actual Co");''',
        'face':r'''oldV[0]=1;sweep[0]=std::numeric_limits<Real>::quiet_NaN();s.gasPhiRhoE[0]=-77;s.gasPhiRho[0]=-77;computeGasCourantFieldKernel(&s,.1);ck(std::isnan(s.gasPhiRhoE[0])&&std::isnan(s.gasPhiRho[0])&&s.gasSpecies.faceStatus[0]!=0,"nonfinite geometry arithmetic or its failure diagnostic was hidden");''',
    }
    compile_probe(tmp_path,legacy_fixture()+r'''
int main(){State s;initialise(s);s.sstConfigured=1;Real oldV[2]={std::numeric_limits<Real>::quiet_NaN(),1},newV[2]={2,2},sweep[3]={0,0,0};moving(s,oldV,newV,sweep,Real(.1));threadIdx.x=0;
'''+calls[kernel]+'}')

@pytest.mark.parametrize('case',['species_nan','face_status','negative_candidate','rk_nan','sst_flux_nan'])
def test_strict_retry_candidate_still_stops_before_publication(tmp_path,case):
    cases={
        'species_nan':r'''s.gasSpecies.flux[0]=std::numeric_limits<Real>::quiet_NaN();applyGasFluxDivergenceByCellKernel(&s,.1,true);ck(s.rho[0]==before&&s.gasSpecies.rho[0]==species&&s.gasSpecies.cellStatus[0]!=0,"strict species retry changed");''',
        'face_status':r'''s.gasSpecies.faceStatus[0]=3;applyGasFluxDivergenceByCellKernel(&s,.1,true);ck(s.rho[0]==before&&s.gasSpecies.rho[0]==species&&s.gasSpecies.cellStatus[0]==3,"strict status retry changed");''',
        'negative_candidate':r'''s.gasPhiRho[1]=20;s.gasSpecies.flux[1]=18;s.gasSpecies.flux[4]=2;applyGasFluxDivergenceByCellKernel(&s,.1,true);ck(s.rho[0]==before&&s.gasSpecies.rho[0]==species&&s.gasSpecies.cellStatus[0]!=0,"strict inventory retry changed");''',
        'rk_nan':r'''saveGasConservativeStateKernel(&s);s.gasSpecies.rho[0]=std::numeric_limits<Real>::quiet_NaN();blendGasConservativeStateKernel(&s,.5,.5,true);ck(s.rho[0]==before&&s.gasSpecies.cellStatus[0]!=0,"strict RK retry changed");''',
        'sst_flux_nan':r'''s.sstConfigured=1;s.k[0]=1;s.omega[0]=2;s.rhoK[0]=1;s.rhoOmega[0]=2;Real oldV[2]={1,1},newV[2]={2,2},sweep[3]={0,0,0};s.gasGeometry.enabled=true;s.gasGeometry.oldVolume=oldV;s.gasGeometry.newVolume=newV;s.gasGeometry.faceSweptVolume=sweep;s.gasGeometry.interval=.1;s.sstPhiRhoK[0]=std::numeric_limits<Real>::quiet_NaN();applySstFluxAndSourceKernel(&s,.1,true);ck(s.rhoK[0]==1&&s.rhoOmega[0]==2&&s.gasSpecies.cellStatus[0]!=0,"strict SST retry changed");''',
    }
    compile_probe(tmp_path,fixture()+'\nint main(){State s;initialise(s);threadIdx.x=0;const Real before=s.rho[0],species=s.gasSpecies.rho[0];'+cases[case]+'}')

@pytest.mark.parametrize('field',['flux','rho','initial'])
def test_missing_consumed_species_storage_remains_actual_error(tmp_path,field):
    action='blendGasConservativeStateKernel(&s,.5,.5);' if field=='initial' else 'applyGasFluxDivergenceByCellKernel(&s,.1);'
    compile_probe(tmp_path,fixture()+'\nint main(){State s;initialise(s);threadIdx.x=0;saveGasConservativeStateKernel(&s);const Real before=s.rho[0];s.gasSpecies.cellStatus[0]=3;s.gasSpecies.'+field+'=nullptr;'+action+r'''ck(s.gasSpecies.cellStatus[0]==int(ugkwp::GasTransportCode::InvalidStorage)&&s.rho[0]==before,"missing consumed species backing hidden by terminal status routing");}''')

@pytest.mark.parametrize('kernel',['convective','face','divergence','sst_update'])
def test_missing_geometry_backing_remains_error(tmp_path,kernel):
    calls={
        'convective':'computeGasConvectiveCourantByCellKernel(&s,.1);ck(std::isnan(s.gasFluxPositivityScale[0]),"missing volume Co placeholder lost");',
        'face':'s.gasGeometry.faceSweptVolume=nullptr;computeGasCourantFieldKernel(&s,.1);ck(std::isnan(s.gasPhiRhoE[0])&&s.gasPhiRho[0]==0,"missing face storage placeholder lost");',
        'divergence':'applyGasFluxDivergenceByCellKernel(&s,.1);ck(s.rho[0]==2,"missing volume published old/default inventory");',
        'sst_update':'applySstFluxAndSourceKernel(&s,.1);ck(s.rhoK[0]==6&&s.rhoOmega[0]==8,"missing volume published old/default SST inventory");',
    }
    compile_probe(tmp_path,legacy_fixture()+r'''
int main(){State s;initialise(s);s.sstConfigured=1;Real oldV[2]={1,1},newV[2]={2,2},sweep[3]={0,0,0};moving(s,oldV,newV,sweep,Real(.1));s.gasGeometry.newVolume=nullptr;threadIdx.x=0;
'''+calls[kernel]+r'''ck((s.gasSpecies.cellStatus[0]!=0||s.gasSpecies.faceStatus[0]!=0),"missing geometry failure hidden");}''')

@pytest.mark.parametrize('kernel',['convective'])
def test_post_source_strict_numeric_geometry_still_retries(tmp_path,kernel):
    calls={
        'convective':'computeGasConvectiveCourantByCellKernel(&s,.1,true);ck(std::isnan(s.gasFluxPositivityScale[0]),"strict Co geometry changed");',
    }
    compile_probe(tmp_path,legacy_fixture()+r'''
int main(){State s;initialise(s);s.sstConfigured=1;Real oldV[2]={std::numeric_limits<Real>::quiet_NaN(),1},newV[2]={2,2},sweep[3]={0,0,0};moving(s,oldV,newV,sweep,Real(.1));threadIdx.x=0;
'''+calls[kernel]+r'''ck(s.gasSpecies.cellStatus[0]!=0,"strict geometry retry certificate missing");}''')

@pytest.mark.parametrize('kernel',['divergence','sst_update','co'])
def test_plain_legacy_terminal_numeric_tail_needs_no_status_storage(tmp_path,kernel):
    ordinary=legacy_fixture().replace('ugkwp::GasSpeciesState<Real,2> gasSpecies;','').replace('s.gasSpecies.cellStatus=new int[2]{};s.gasSpecies.faceStatus=new int[3]{};','')
    calls={
        'divergence':'s.gasPhiRho[0]=std::numeric_limits<Real>::quiet_NaN();applyGasFluxDivergenceByCellKernel(&s,.1);ck(std::isnan(s.rho[0])&&s.rhoE[0]==e/2,"plain legacy numeric bulk tail trapped");',
        'sst_update':'s.sstPhiRhoK[0]=std::numeric_limits<Real>::quiet_NaN();applySstFluxAndSourceKernel(&s,.1);ck(std::isnan(s.rhoK[0])&&s.rhoOmega[0]!=8,"plain legacy numeric SST tail trapped");',
        'co':'oldV[0]=std::numeric_limits<Real>::quiet_NaN();s.gasPhiRhoE[0]=4;computeGasConvectiveCourantByCellKernel(&s,.1);ck(s.gasFluxPositivityScale[0]==Real(.5)*Real(.1)*4/2,"plain legacy Co tail trapped");',
    }
    compile_probe(tmp_path,ordinary+r'''
int main(){State s;initialise(s);s.sstConfigured=1;Real oldV[2]={1,1},newV[2]={2,2},sweep[3]={0,0,0};moving(s,oldV,newV,sweep,Real(.1));threadIdx.x=0;const Real e=s.rhoE[0];
'''+calls[kernel]+'}')

def test_rk_terminal_species_failure_preserves_actual_sst_stage_weights(tmp_path):
    compile_probe(tmp_path,fixture()+r'''
int main(){State s;initialise(s);s.sstConfigured=1;s.rhoK[0]=2;s.rhoOmega[0]=4;threadIdx.x=0;saveGasConservativeStateKernel(&s);
s.rhoK[0]=10;s.rhoOmega[0]=20;s.rhoUx[0]=8;s.gasSpecies.rho[0]=std::numeric_limits<Real>::quiet_NaN();blendGasConservativeStateKernel(&s,.25,.75);
ck(s.rhoK[0]==8&&s.rhoOmega[0]==16&&s.rhoUx[0]==6,"terminal RK physical SST and momentum stages incomplete");ck(std::isnan(s.gasSpecies.rho[0]),"terminal species arithmetic not evaluated");}
''')


def test_save_overwrites_all_conservative_and_initial_sentinels(tmp_path):
    helper=r'''void near(Real a,Real b,const char*m){ck(std::abs(a-b)<=1e-12,m);}'''
    compile_probe(tmp_path,fixture()+helper+AUDIT+r'''
int main(){State s;initialise(s);s.sstConfigured=1;s.rhoUx[0]=5;s.rhoUy[0]=7;s.rhoUz[0]=11;s.rhoK[0]=13;s.rhoOmega[0]=17;s.rhoNext[0]=s.rhoUxNext[0]=s.rhoUyNext[0]=s.rhoUzNext[0]=s.rhoENext[0]=-77;s.rhoKInitial[0]=s.rhoOmegaInitial[0]=-78;s.gasSpecies.initial[0]=s.gasSpecies.initial[2]=-79;audit(s);
threadIdx.x=0;saveGasConservativeStateKernel(&s);
ck(s.rhoNext[0]==1&&s.rhoUxNext[0]==5&&s.rhoUyNext[0]==7&&s.rhoUzNext[0]==11&&s.rhoENext[0]==s.rhoE[0],"bulk Next save incomplete");ck(s.rhoKInitial[0]==13&&s.rhoOmegaInitial[0]==17&&s.gasSpecies.initial[0]==.9&&s.gasSpecies.initial[2]==.1,"SST/species initial save incomplete");}
''')

@pytest.mark.parametrize('target',['nan','inf'])
def test_terminal_wall_projection_feeds_actual_sst_gradients_and_eddy_fields(tmp_path,target):
    from test_boundary_layer_operators import SETUP
    value='std::numeric_limits<Real>::quiet_NaN()' if target=='nan' else 'std::numeric_limits<Real>::infinity()'
    compile_probe(tmp_path,legacy_fixture()+AUDIT+SETUP+r'''
int main(){State s;initialise(s);wall(s);audit(s);s.gasBoundaryLayer.sst[0].ownerOmega=TARGET;threadIdx.x=0;
s.gradKX[0]=s.gradKY[0]=s.gradKZ[0]=s.gradOmegaX[0]=s.gradOmegaY[0]=s.gradOmegaZ[0]=s.sstF1[0]=s.sstF2[0]=s.nut[0]=-77;
applySstWallFunctionStateKernel(&s);computeSstGradientsKernel(&s);computeGasEddyViscosityKernel(&s);
ck(s.gradKX[0]==0&&s.gradKY[0]==0&&s.gradKZ[0]==0,"terminal projected k gradients not published");ck(std::isnan(s.gradOmegaX[0])&&std::isnan(s.gradOmegaY[0])&&std::isnan(s.gradOmegaZ[0]),"terminal projected omega gradients not actual arithmetic");ck(s.sstF1[0]!=-77&&s.sstF2[0]!=-77&&s.nut[0]!=-77,"terminal projected SST derived fields stale");ck(s.gasSpecies.cellStatus[0]!=0,"projection numerical certificate cleared");}
'''.replace('TARGET',value))


def test_actual_chmt_current_co_keeps_four_copies_and_restores_on_error(tmp_path):
    from pathlib import Path
    def function(source, marker):
        start=source.index(marker);brace=source.index('{',start);level=1;i=brace+1
        while level:
            level+=(source[i]=='{')-(source[i]=='}');i+=1
        return source[start:i]
    from test_current_gas_courant import CUDA, launches
    root=SOURCE_ROOT
    helper=launches((root/'common/GpuGasCourant.cuh').read_text())
    backend=(root/'applications/CHMT/gpu/Backend.cu').read_text()
    wrapper=function(backend,'static bool currentGasCourant')
    reductions_source=backend[backend.index('struct CoupledCourantMaximum'):backend.index('static bool currentGasCourant')]
    shim=CUDA+r'''
#include <stdexcept>
constexpr int cudaMemcpyDeviceToDevice=1;int copies=0,failCopy=0,failReduction=0;
int cudaMemcpy(void*to,const void*from,size_t n,int){++copies;if(copies==failCopy)return 72;std::memcpy(to,from,n);return 0;}

using SharedGasDeviceView=State;
struct WallKnot{};template<class...T>bool sampleGasWallProgramLocal(T&&...){return true;}
struct Backend{ugkwp::GasBoundaryLayerState<Real> gasBoundaryLayer;int program=0,courantReductionStorage=0;double elapsed=0;std::string error;struct Trial{int surface=0;}trial;int nCells=2,nFaces=3,nInternalFaces=1,hostTurbulenceModel=3,fixedCellBlockThreads=4,fixedFaceBlockThreads=4;double microTime=2;void*gasCaptureStream=nullptr;State*deviceState=nullptr;struct Storage{State*p=nullptr;const State&hostView(){return*p;}bool wallMassPublished(){return true;}template<class...T>bool prepareWallBoundary(T&&...){return true;}}storage;struct Model{struct Physics{Real cfl=.5;}physics;}model;};
REDUCTIONS_SOURCE
namespace thrust{
template<class T>T*device_pointer_cast(T*p){return p;}namespace cuda{struct Par{template<class A>Par& operator()(A&){return *this;}int on(void*){return 0;}}par;}
inline int make_counting_iterator(int a){return a;}
template<class M,class T,class F>T transform_reduce(int,int a,int b,M map,T value,F f){++reductions;if(reductions==failReduction)throw std::runtime_error("actual reduction failure");for(;a!=b;++a)value=f(value,map(a));return value;}
template<class T,class F>T reduce(int,const T*a,const T*b,T value,F f){++reductions;if(reductions==failReduction)throw std::runtime_error("actual reduction failure");for(;a!=b;++a)value=f(value,*a);return value;}
}
'''
    shim=shim.replace('REDUCTIONS_SOURCE',reductions_source)
    main=r'''
int main(){for(int mode:{0,1,2,3,4,5}){State s;initialise(s);s.sstConfigured=1;s.turbulenceModel=3;s.rhoK[0]=-3;s.rhoOmega[0]=-5;
Backend b;b.deviceState=b.storage.p=&s;copies=launches=reductions=failCopy=failLaunch=failReduction=0;last.clear();
if(mode==1)failLaunch=1;if(mode==2)failReduction=1;if(mode==3)failCopy=3;if(mode==4)failCopy=4;if(mode==5)failCopy=1;
GasCourantMaxima out;out.convective=out.diffusion=out.sst=out.maximum=-777;const bool okay=currentGasCourant(b,.001,out);
if(mode==0){ck(okay&&copies==4,"actual currentCo four-copy lifecycle changed");ck(s.rhoK[0]==-3&&s.rhoOmega[0]==-5,"currentCo recovery inventory was not restored");ck(out.convective!=-777&&out.diffusion!=-777&&out.sst!=-777&&out.maximum!=-777,"currentCo reduced maxima incomplete");}
if(mode==1||mode==2){ck(!okay&&copies==4&&s.rhoK[0]==-3&&s.rhoOmega[0]==-5,"currentCo actual launch/reduction error skipped restore");ck(out.maximum==-777&&!last.empty(),"currentCo error falsely published output");}
if(mode==3||mode==4)ck(!okay&&copies==4&&!last.empty(),"actual restore copy error hidden or second restore skipped");
if(mode==5)ck(!okay&&copies==1&&launches==0&&s.rhoK[0]==-3&&s.rhoOmega[0]==-5,"actual backup copy error continued computation");
}}
'''
    compile_probe(tmp_path,legacy_fixture()+shim+helper+'namespace transport{using ::setLastError;using ::GasCourantMaxima;using ::computeGasCurrentCourant;void setLastErrorText(const char*p){last=p;}}\n'+wrapper+main)

@pytest.mark.parametrize('kernel',['divergence','sst_update','convective'])
def test_supplied_incidence_slice_with_nonnegative_offset_completes(tmp_path,kernel):
    calls={
        'divergence':'s.gasPhiRho[1]=-2;applyGasFluxDivergenceByCellKernel(&s,.1);ck(s.rho[0]==2.2,"terminal supplied incidence skipped bulk formula");',
        'sst_update':'s.sstPhiRhoK[0]=-6;applySstFluxAndSourceKernel(&s,.1);ck(s.rhoK[0]!=6,"terminal supplied incidence skipped SST formula");',
        'convective':'s.gasPhiRhoE[0]=4;s.gasPhiRhoE[1]=2;computeGasConvectiveCourantByCellKernel(&s,.1);ck(s.gasFluxPositivityScale[0]==Real(.5)*Real(.1)*6,"terminal supplied incidence skipped Co formula");',
    }
    compile_probe(tmp_path,legacy_fixture()+r'''
int main(){State s;initialise(s);s.sstConfigured=1;int* backing=s.cellFaceId;s.cellFaceId=backing+1;s.cellPlaneStart[0]=0;s.cellPlaneCount[0]=2;s.cellFaceId[0]=0;s.cellFaceId[1]=1;threadIdx.x=0;
'''+calls[kernel]+r'''ck(s.gasSpecies.cellStatus[0]==0,"terminal metadata added a refusal");}''')

@pytest.mark.parametrize('case',['omega','source'])
def test_selected_wall_strict_numeric_retry_is_retained(tmp_path,case):
    from test_boundary_layer_operators import SETUP
    bodies={
        'omega':r'''s.gasBoundaryLayer.sst[0].ownerOmega=std::numeric_limits<Real>::infinity();const Real before=s.rhoOmega[0],primitive=s.omega[0];applySstWallFunctionStateKernel(&s,true);ck(s.rhoOmega[0]==before&&s.omega[0]==primitive&&s.gasSpecies.cellStatus[0]!=0,"strict selected omega numeric retry lost");''',
        'source':r'''s.k[0]=.001;s.rhoK[0]=.002;s.omega[0]=1000;const Real before=s.rhoK[0];applySstFluxAndSourceKernel(&s,.1,true);ck(s.rhoK[0]==before&&s.gasSpecies.cellStatus[0]==int(ugkwp::GasTransportCode::SourceStepLimit),"strict selected source retry lost");''',
    }
    compile_probe(tmp_path,legacy_fixture()+SETUP+'\nint main(){State s;initialise(s);wall(s);threadIdx.x=0;'+bodies[case]+'}')


def test_terminal_moving_geometry_after_budget_prefix_does_not_replay(tmp_path):
    compile_probe(tmp_path,fixture()+r'''
int main(){State s;initialise(s);Real oldV[2]={std::numeric_limits<Real>::quiet_NaN(),1},newV[2]={2,2},sweep[3]={0,0,0};s.gasGeometry.enabled=true;s.gasGeometry.oldVolume=oldV;s.gasGeometry.newVolume=newV;s.gasGeometry.faceSweptVolume=sweep;s.gasGeometry.interval=.5;
s.gasPhiRho[1]=-2;s.gasPhiRhoE[1]=-7;s.gasSpecies.flux[1]=-1.8;s.gasSpecies.flux[4]=-.2;threadIdx.x=0;applyGasFluxDivergenceByCellKernel(&s,.5);
ck(std::isnan(s.rho[0])&&std::isnan(s.rhoUx[0])&&std::isnan(s.rhoUy[0])&&std::isnan(s.rhoUz[0])&&std::isnan(s.rhoE[0])&&std::isnan(s.gasSpecies.rho[0])&&std::isnan(s.gasSpecies.rho[2]),"evaluated numeric volume did not finish actual bulk/species tails");
}
''')

@pytest.mark.parametrize('strict',[False,True])
@pytest.mark.parametrize('kind',['empty','negative_count','invalid_ids','other_owner','moving_empty'])
@pytest.mark.parametrize('absent',['flux','face_status','both'])
def test_original_skipped_face_storage_is_not_required(tmp_path,strict,kind,absent):
    setup={
        'empty':'s.cellPlaneCount[0]=0;s.cellPlaneStart[0]=-1;s.cellFaceId=nullptr;',
        'negative_count':'s.cellPlaneCount[0]=-1;',
        'invalid_ids':'s.cellPlaneCount[0]=2;s.cellFaceId[0]=-1;s.cellFaceId[1]=s.nFaces;',
        'other_owner':'s.cellPlaneCount[0]=1;s.cellFaceId[0]=2;',
        'moving_empty':'s.cellPlaneCount[0]=0;s.cellPlaneStart[0]=-1;s.cellFaceId=nullptr;Real oldV[2]={1,1},newV[2]={2,2};s.gasGeometry.enabled=true;s.gasGeometry.oldVolume=oldV;s.gasGeometry.newVolume=newV;',
    }
    missing=('s.gasSpecies.flux=nullptr;' if absent!='face_status' else '')+('s.gasSpecies.faceStatus=nullptr;' if absent!='flux' else '')
    ratio='Real(.5)' if kind=='moving_empty' else 'Real(1)'
    body='int main(){State s;initialise(s);threadIdx.x=0;'+missing+setup[kind]+r'''const Real before[]={s.rho[0],s.rhoUx[0],s.rhoUy[0],s.rhoUz[0],s.rhoE[0],s.gasSpecies.rho[0],s.gasSpecies.rho[2]};applyGasFluxDivergenceByCellKernel(&s,.1,STRICT);const Real after[]={s.rho[0],s.rhoUx[0],s.rhoUy[0],s.rhoUz[0],s.rhoE[0],s.gasSpecies.rho[0],s.gasSpecies.rho[2]};ck(s.gasSpecies.cellStatus[0]==0,"unused face backing added a gate before legitimate skips");for(int k=0;k<7;++k)ck(after[k]==before[k]*RATIO,"original skipped-face final conservative fields lost");}'''.replace('STRICT','true' if strict else 'false').replace('RATIO',ratio)
    compile_probe(tmp_path,fixture()+body)


def test_strict_face_status_returns_before_unconsumed_backing(tmp_path):
    compile_probe(tmp_path,fixture()+r'''
int main(){for(int absent:{0,1,2}){State s;initialise(s);threadIdx.x=0;s.gasSpecies.faceStatus[0]=3;
if(absent!=1)s.gasSpecies.flux=nullptr;if(absent!=0)s.gasSpecies.rho=nullptr;
const Real before[]={s.rho[0],s.rhoUx[0],s.rhoUy[0],s.rhoUz[0],s.rhoE[0]};applyGasFluxDivergenceByCellKernel(&s,.1,true);
const Real after[]={s.rho[0],s.rhoUx[0],s.rhoUy[0],s.rhoUz[0],s.rhoE[0]};ck(s.gasSpecies.cellStatus[0]==3&&std::memcmp(before,after,sizeof before)==0,"strict face-status prefix inspected unconsumed backing");}}
''')


@pytest.mark.parametrize('native', [False, True])
@pytest.mark.parametrize('kernel', ['divergence','sst_update','convective'])
def test_incidence_before_logical_array_origin_is_resource_error(tmp_path,kernel,native):
    mode='true' if native else 'false'
    calls={
        'divergence':'applyGasFluxDivergenceByCellKernel(&s,.1,MODE);',
        'sst_update':'applySstFluxAndSourceKernel(&s,.1,MODE);',
        'convective':'computeGasConvectiveCourantByCellKernel(&s,.1,MODE);',
    }
    body=r'''
int main(){State s;initialise(s);s.sstConfigured=1;
int* backing=s.cellFaceId;s.cellFaceId=backing+1;s.cellPlaneStart[0]=-1;s.cellPlaneCount[0]=2;
threadIdx.x=0;const Real before[]={s.rho[0],s.rhoUx[0],s.rhoUy[0],s.rhoUz[0],s.rhoE[0],s.rhoK[0],s.rhoOmega[0]};
CALL
const Real after[]={s.rho[0],s.rhoUx[0],s.rhoUy[0],s.rhoUz[0],s.rhoE[0],s.rhoK[0],s.rhoOmega[0]};
ck(s.gasSpecies.cellStatus[0]==int(ugkwp::GasTransportCode::InvalidStorage),"logical incidence origin error not recorded");
ck(std::memcmp(before,after,sizeof before)==0,"missing incidence changed conserved state");
}'''.replace('CALL',calls[kernel].replace('MODE',mode))
    compile_probe(tmp_path,legacy_fixture()+body,sanitizers=True)
