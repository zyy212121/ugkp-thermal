import importlib.util
from pathlib import Path
import subprocess

import pytest

from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT
from test_legacy_ale_transport import fixture as legacy_fixture
from test_mixture_transport import fixture as mixture_fixture


@pytest.fixture(scope="module", params=(32, 64))
def precision_tree(request):
    module_path = SOURCE_ROOT / "tool/precision/materialize_sources.py"
    spec = importlib.util.spec_from_file_location("diffusion_precision", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.materialize(SOURCE_ROOT, request.param)


def execute(tmp_path, precision_tree, body):
    tests = VALIDATION_ROOT / "tests/gas_transport"
    source = (tests / "gas_state_probe.cpp").read_text().split("void hashBytes", 1)[0]
    cpp = tmp_path / "diffusion.cpp"
    if "class Geometry = Real" in (precision_tree/"common/gasTransport/GasStateView.H").read_text():
        body=body.replace("GasStateView<Real,Time,ugkwp::SstCoefficients>",
                          "GasStateView<Real,Time,ugkwp::SstCoefficients,Real,double,double>")
    cpp.write_text(source + body)
    executable = tmp_path / "diffusion"
    build = subprocess.run([
        "g++", "-std=c++17", "-O1", "-I" + str(tests),
        "-I" + str(precision_tree / "common"),
        "-I" + str(precision_tree / "common/gasNumerics"),
        str(cpp), "-o", str(executable),
    ], capture_output=True, text=True)
    assert build.returncode == 0, build.stdout + build.stderr
    result = subprocess.run([str(executable)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_density_weighted_transport_and_all_turbulence_models(tmp_path, precision_tree):
    execute(tmp_path, precision_tree, legacy_fixture().split("void moving(", 1)[0] + r"""
int main(){
    State s;initialise(s);
    s.rho[0]=1;s.rho[1]=100;s.nut[0]=s.nut[1]=1;
    s.gasMu=0;s.gasPrClamped=1;s.turbulentPrandtl=100;
    s.sstConfigured=0;
    const auto left=gasCellPrimitive(s,0),right=gasCellPrimitive(s,1);
    for(int model=1;model<=3;++model){
        s.turbulenceModel=model;
        const Real signal=gasDiffusiveSignal(s,0,0,1,left,right,Real(1),Real(1));
        near(signal,Real(2)*Real(4)/Real(3)*Real(50.5),"face dynamic viscosity lost density ratio");
    }
    s.turbulenceModel=3;s.sstConfigured=1;
    s.sstCoefficients.alphaK1=s.sstCoefficients.alphaK2=Real(4);
    s.sstCoefficients.alphaOmega1=s.sstCoefficients.alphaOmega2=Real(3);
    s.sstWallTreatment=1;
    near(gasDiffusiveSignal(s,0,0,1,left,right,Real(1),Real(1)),Real(404),"wall-function SST diffusion bound omitted");
    s.sstWallTreatment=0;
    near(gasDiffusiveSignal(s,0,0,1,left,right,Real(1),Real(1)),Real(808),"bounded LowRe SST diffusion omitted");
    s.sstConfigured=0;s.turbulenceModel=0;s.gasThermalConductivity=Real(900);
    near(gasDiffusiveSignal(s,0,0,1,left,right,Real(1),Real(1)),
         Real(1800)/(s.rho[0]*s.gasCp/s.gammaGas),"single-gas volume heat capacity wrong");
    s.riemannBoundaryTFix[1]=1;s.riemannBoundaryT[1]=s.Tgas[0];threadIdx.x=1;
    computeGasCourantFieldKernel(&s,1e-3,false,false);
    ck(s.gasPhiRhoE[1]==0,"default slip Courant changed");
    computeGasCourantFieldKernel(&s,1e-3,false,true);
    near(s.gasPhiRhoE[1],Real(3600)/(s.rho[0]*s.gasCp/s.gammaGas),"fixed-temperature slip diffusion skipped");
}
""")


def test_low_re_sst_diffusion_limits_skew_face_without_changing_convection(tmp_path, precision_tree):
    execute(tmp_path, precision_tree, legacy_fixture().split("void moving(", 1)[0] + r"""
int main(){
    State s;initialise(s);
    s.sstConfigured=1;
    s.sstWallTreatment=0;
    s.Cy[1]=1;
    s.gasMu=Real(.125);
    s.gradKY[0]=s.gradKY[1]=100;
    s.gradOmegaY[0]=s.gradOmegaY[1]=-200;
    computeSstFaceFluxKernel(&s);
    ck(s.sstPhiRhoK[0]==0&&s.sstPhiRhoOmega[0]==0,
        "skew correction transported equal SST states");

    s.k[1]=2;
    s.omega[1]=5;
    computeSstFaceFluxKernel(&s);
    near(s.sstPhiRhoK[0],Real(.25),"overlarge k diffusion not bounded");
    near(s.sstPhiRhoOmega[0],Real(-.25),"overlarge omega diffusion not bounded");

    s.gasPhiRho[0]=2;
    computeSstFaceFluxKernel(&s);
    near(s.sstPhiRhoK[0],Real(6.25),"k convection changed");
    near(s.sstPhiRhoOmega[0],Real(7.75),"omega convection changed");

    s.gasPhiRho[0]=0;
    s.gradKY[0]=s.gradKY[1]=-100;
    s.gradOmegaY[0]=s.gradOmegaY[1]=200;
    computeSstFaceFluxKernel(&s);
    ck(s.sstPhiRhoK[0]==0&&s.sstPhiRhoOmega[0]==0,"skew correction reversed diffusion");

    s.gradKY[0]=s.gradKY[1]=Real(-.25);
    s.gradOmegaY[0]=s.gradOmegaY[1]=Real(.25);
    computeSstFaceFluxKernel(&s);
    near(s.sstPhiRhoK[0],Real(.09375),"admissible k gradient changed");
    near(s.sstPhiRhoOmega[0],Real(-.09375),"admissible omega gradient changed");

    s.gradOmegaY[0]=std::numeric_limits<Real>::quiet_NaN();
    computeSstFaceFluxKernel(&s);
    ck(std::isnan(s.sstPhiRhoOmega[0]),"limiter hid a nonfinite gradient");

    s.sstWallTreatment=1;
    s.gradKY[0]=s.gradKY[1]=100;
    s.gradOmegaY[0]=s.gradOmegaY[1]=-200;
    computeSstFaceFluxKernel(&s);
    near(s.sstPhiRhoK[0],Real(12.625),"wall-function k flux changed");
    near(s.sstPhiRhoOmega[0],Real(-25.125),"wall-function omega flux changed");
}
""")


def test_species_diffusion_and_variable_heat_capacity(tmp_path, precision_tree):
    execute(tmp_path, precision_tree, mixture_fixture().split("void transport(", 1)[0] + r"""
void near(Real actual,Real wanted,const char*message){
    ck(std::abs(actual-wanted)<Real(128)*std::numeric_limits<Real>::epsilon()*std::max(Real(1),std::abs(wanted)),message);
}
int main(){
    State s;initialise(s);s.gasMu=0;s.gasThermalConductivity=0;s.turbulenceModel=0;
    s.rho[1]=100;s.gasSpecies.rho[1]=10;s.gasSpecies.rho[3]=90;
    Real diffusion[2]={2,3};s.gasSpecies.diffusivity=diffusion;
    const auto left=gasCellPrimitive(s,0),right=gasCellPrimitive(s,1);
    const Real signal=gasDiffusiveSignal(s,0,0,1,left,right,Real(1),Real(1));
    near(signal,Real(303),"corrected species diffusion omitted density or fastest component");
    Real Y[2]={Real(.5),Real(.5)},grad[2]={1,-1},flux[2],energy=0;
    ck(ugkwp::correctedSpeciesDiffusion(Y,grad,Real(50.5),diffusion,Real(0),Real(1),Real(700),
        s.gasSpecies.thermo,flux,energy),"species reference failed");
    ck(signal>=Real(2)*std::abs(flux[0]),"bound smaller than actual two-cell species rate");
    s.gasSpecies.diffusivity=nullptr;s.gasThermalConductivity=900;
    s.gasSpecies.heatCapacity[0]=1000;s.gasSpecies.gasConstant[0]=800;
    s.gasSpecies.heatCapacity[1]=1000;s.gasSpecies.gasConstant[1]=100;
    near(gasDiffusiveSignal(s,0,0,1,left,right,Real(1),Real(1)),Real(9),"mixture heat capacity used face average");
    Real bad[2]={std::numeric_limits<Real>::quiet_NaN(),1};
    ck(std::isnan(ugkwp::correctedSpeciesDiffusionRate(Y,bad,Real(0),Real(1))),"invalid species diffusivity hidden");
}
""")


def test_courant_prepares_current_turbulence_from_poisoned_storage(tmp_path, precision_tree):
    from test_current_gas_courant import launches, CUDA
    helper=launches((SOURCE_ROOT/'common/GpuGasCourant.cuh').read_text())
    execute(tmp_path,precision_tree,legacy_fixture().split("void moving(",1)[0]+CUDA+helper+r"""
int main(){
    for(int model=1;model<=3;++model){
        Host s;initialise(s);s.hostTurbulenceModel=s.turbulenceModel=model;s.sstConfigured=model==3;
        s.riemannBoundaryKind[2]=s.gasBoundaryKind[2]=0;
        s.sstBoundaryKMode[2]=s.sstBoundaryOmegaMode[2]=2;
        s.sstBoundaryK[2]=Real(2);s.sstBoundaryOmega[2]=Real(3);
        s.lesDeltaCoeff=1;s.smagorinskyCs=Real(.17);s.waleCw=Real(.325);s.sstWallTreatment=0;
        for(int revision=0;revision<2;++revision){
            for(int c=0;c<2;++c){
                s.gasPhiRho[2]=std::numeric_limits<Real>::quiet_NaN();
                s.rhoUx[c]=Real(2*(c+1)*(revision==0?1:-1));s.rhoE[c]=Real(250000)+s.rhoUx[c]*s.rhoUx[c]/Real(4);
                s.rhoK[c]=Real(6*(revision+1));s.rhoOmega[c]=Real(8*(revision+1));
                s.nut[c]=s.sstF1[c]=s.sstF2[c]=s.k[c]=s.omega[c]=std::numeric_limits<Real>::quiet_NaN();
            }
            GasCourantMaxima off,on;
            ck(computeGasCurrentCourant(&s,.001,0.,4,4,nullptr,off,Maximum{})==0,"acoustic estimate failed");
            ck(std::isnan(s.nut[0]),"disabled limiter prepared unused transport");
            ck(computeGasCurrentCourant(&s,.001,0.,4,4,nullptr,on,Maximum{},true)==0,"diffusive estimate failed");
            ck(std::isfinite(on.maximum)&&on.maximum>=off.maximum,"invalid diffusive estimate");
            for(int c=0;c<2;++c){
                ck(s.gasSpecies.cellStatus[c]==0,"transport preparation reported fatal");
                ck(std::isfinite(s.nut[c])&&s.nut[c]>0,"initial turbulent viscosity absent");
                if(model==3)ck(std::isfinite(s.sstF1[c])&&s.sstF1[c]>=0&&s.sstF1[c]<=1,"initial F1 absent");
            }
            if(model==3){
                ck(std::isfinite(s.gasPhiRho[2]),"SST read unprepared inletOutlet direction");
                ck((s.gasPhiRho[2]>0)==(revision==0),"SST reused stale boundary flow direction");
            }
            const Real actualNut[2]={s.nut[0],s.nut[1]},actualF1[2]={s.sstF1[0],s.sstF1[1]};
            ck(prepareGasTransportCoefficients(&s,4,nullptr,false,false,true)==0,"physical-stage preparation failed");
            ck(prepareGasSstTransportCoefficients(&s,4,nullptr)==0,"physical F1 preparation failed");
            for(int c=0;c<2;++c){
                near(s.nut[c],actualNut[c],"CFL and physical transport differ");
                if(model==3)near(s.sstF1[c],actualF1[c],"CFL and physical F1 differ");
            }
        }
    }
}
""")
