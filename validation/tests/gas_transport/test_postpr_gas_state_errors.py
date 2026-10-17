"""Host execution of production headers; deliberately not CUDA coverage."""
from pathlib import Path
import ast
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent

def test_postpr_gas_state_errors():
    source = (HERE / "test_mixture_transport.py").read_text()
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "fixture")
    scope = {"re": re, "VALIDATION_ROOT": ROOT / "validation"}
    exec(ast.get_source_segment(source, node), scope)
    pre = (HERE / "gas_state_probe.cpp").read_text().split("void hashBytes")[0]
    body = r'''
int main(){
 int failures=0;
 auto check=[&](bool ok,const char* label){if(!ok){printf("FAIL %s\n",label);++failures;}};
 {State s;initialise(s);ugkp::diagnostics::DeviceContext d{};s.diagnostics=&d;s.gasSpecies.mode=ugkwp::GasMode::SingleLegacy;
 s.rho[0]=NAN;s.rhoE[0]=NAN;auto t=s.Tgas[0];recoverGasPrimitiveCell(s,0);
 check(ugkp::diagnostics::hasFatal(&d)&&std::isnan(s.rho[0])&&std::isnan(s.rhoE[0])&&s.Tgas[0]==t,"legacy NaN preserved and diagnosed");}
 {State s;initialise(s);ugkp::diagnostics::DeviceContext d{};s.diagnostics=&d;s.gasSpecies.mode=ugkwp::GasMode::SingleLegacy;s.gasGeometry.enabled=true;
 s.rhoE[0]=-100;auto t=s.Tgas[0];recoverGasPrimitiveCell(s,0);
 check(ugkp::diagnostics::hasFatal(&d)&&s.Tgas[0]==t,"ALE negative T not published");}
 {State s;initialise(s);ugkp::diagnostics::DeviceContext d{};s.diagnostics=&d;s.riemannBoundaryTFix[1]=1;s.riemannBoundaryT[1]=-10;s.gasThermalConductivity=.1;threadIdx.x=1;computeGasInternalFaceFluxKernel<false>(&s,1e-5);
 check(ugkp::diagnostics::hasFatal(&d),"laminar slip negative wall T");}
 {State s;initialise(s);ugkp::diagnostics::DeviceContext d{};s.diagnostics=&d;s.cellFaceId[0]=999;s.cellFaceId[1]=999;threadIdx.x=0;auto r=s.rho[0];applyGasFluxDivergenceByCellKernel(&s,1e-5);
 check(ugkp::diagnostics::hasFatal(&d)&&s.rho[0]==r,"invalid divergence face");}
 {State s;initialise(s);ugkp::diagnostics::DeviceContext d{};s.diagnostics=&d;s.faceOwner[0]=999;threadIdx.x=0;computeGasCourantFieldKernel(&s,1e-5);
 check(ugkp::diagnostics::hasFatal(&d)&&s.gasPhiRhoE[0]!=OfGreat,"Courant invalid owner");}
 {State s;initialise(s);ugkp::diagnostics::DeviceContext d{};s.diagnostics=&d;s.gasSpecies.mode=ugkwp::GasMode::SingleLegacy;s.rho[0]=s.rhoMin/2;s.rhoUx[0]=s.rhoUy[0]=s.rhoUz[0]=0;s.rhoE[0]=-1;recoverGasPrimitiveCell(s,0);
 const Real expected=fmax(s.rhoMin,ugkwp::gasLegacyThermalMinimumInternalEnergy(Real(s.rhoMin),Real(s.Rgas),Real(s.TgasMin),Real(s.gammaGas),Real(OfSmall)));
 check(!ugkp::diagnostics::hasFatal(&d)&&s.rho[0]==s.rhoMin&&s.rhoE[0]==expected,"finite legacy floors retained");}

 {State s;initialise(s);ugkp::diagnostics::DeviceContext d{};s.diagnostics=&d;s.cellFaceId[0]=-1;threadIdx.x=0;s.gradRhoX[0]=123;computeGasPrimitiveGradientsKernel(&s);
 check(ugkp::diagnostics::hasFatal(&d)&&s.gradRhoX[0]==123,"gradient invalid face does not publish");}
 {State s;initialise(s);ugkp::diagnostics::DeviceContext d{};s.diagnostics=&d;s.gasLimiter=1;s.faceNeighbour[0]=999;threadIdx.x=0;s.gasGradientLimiterRho[0]=.25;computeGasGradientLimiterKernel(&s);
 check(ugkp::diagnostics::hasFatal(&d)&&s.gasGradientLimiterRho[0]==.25,"limiter invalid neighbour before dereference");}
 {State s;initialise(s);ugkp::diagnostics::DeviceContext d{};s.diagnostics=&d;s.gasLimiter=1;s.gasSpecies.mode=ugkwp::GasMode::SingleLegacy;s.gradRhoX[0]=NAN;threadIdx.x=0;s.gasGradientLimiterRho[0]=.25;computeGasGradientLimiterKernel(&s);
 check(ugkp::diagnostics::hasFatal(&d)&&s.gasGradientLimiterRho[0]==.25,"limiter NaN slope not erased");}
 {State s;initialise(s);ugkp::diagnostics::DeviceContext d{};s.diagnostics=&d;s.riemannBoundaryTFix[1]=1;s.riemannBoundaryT[1]=-10;threadIdx.x=0;s.gradRhoX[0]=123;computeGasPrimitiveGradientsKernel(&s);
 check(ugkp::diagnostics::hasFatal(&d)&&s.gradRhoX[0]==123,"wall gradient negative T does not publish");}

 {State s;initialise(s);ugkp::diagnostics::DeviceContext d{};s.diagnostics=&d;s.riemannBoundaryTFix[1]=1;s.riemannBoundaryT[1]=-10;threadIdx.x=1;computeGasInternalFaceFluxKernel<false>(&s,1e-5);
 threadIdx.x=0;s.gasPhiRho[0]=10;auto r=s.rho[0];auto e=s.rhoE[0];applyGasFluxDivergenceByCellKernel(&s,1e-5);
 check(s.rho[0]==r&&s.rhoE[0]==e,"fatal face does not drive divergence publication");
 s.rhoNext[0]=2*r;s.rhoENext[0]=2*e;blendGasConservativeStateKernel(&s,.5,.5);
 check(s.rho[0]==r&&s.rhoE[0]==e,"fatal face does not drive RK publication");
 s.gasSpecies.mode=ugkwp::GasMode::SingleLegacy;s.rhoE[0]=-1;recoverGasPrimitiveCell(s,0);
 check(s.rhoE[0]==-1,"fatal face does not drive recovery repair");}
 return failures?1:0;
}
'''
    with tempfile.TemporaryDirectory() as temp:
        cpp=Path(temp)/"probe.cpp"; exe=Path(temp)/"probe"
        cpp.write_text(pre+scope["fixture"]()+body)
        subprocess.run(["g++","-std=c++17","-O0","-I"+str(HERE),"-I"+str(ROOT/"common"),"-I"+str(ROOT/"common/gasNumerics"),str(cpp),"-o",str(exe)],check=True,capture_output=True,text=True)
        run=subprocess.run([str(exe)],capture_output=True,text=True)
        assert run.returncode == 0, run.stdout + run.stderr
