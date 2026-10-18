"""Focused production-helper regressions for finite clipping and Cp provenance."""
import os
import shutil
import subprocess
import pytest
from test_mixture_state import compile_probe, HERE, ROOT
from test_mixture_transport import fixture
from test_mixture_wall_function import wall_fixture


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("sign", [-1, 1])
def test_nonfinite_species_quotient_is_error_before_clipping(tmp_path, native, sign):
    compile_probe(tmp_path, fixture() + r'''
#include "DiagnosticsHost.H"
int main(){using namespace ugkp::diagnostics;
 for(bool context:{false,true}){
  State s;initialise(s);DeviceContext d{};reset(d,23,9,7,.125);s.diagnostics=context?&d:nullptr;
  s.rho[0]=(std::numeric_limits<Real>::min)();
  s.gasSpecies.rho[0]=SIGN*(std::numeric_limits<Real>::max)();
  s.gasSpecies.rho[s.nCells]=0;
  const Real previous[]={s.Ux[0],s.Uy[0],s.Uz[0],s.p[0],s.Tgas[0],s.gasSpecies.soundSpeed[0],s.gasSpecies.heatCapacity[0],s.gasSpecies.gasConstant[0]};
  recoverGasPrimitiveCell(s,0,NATIVE);
  ck(s.gasSpecies.cellStatus[0]==int(ugkwp::GasTransportCode::NonFiniteState),"nonfinite quotient was clipped or lost coarse error");
  const Real actual[]={s.Ux[0],s.Uy[0],s.Uz[0],s.p[0],s.Tgas[0],s.gasSpecies.soundSpeed[0],s.gasSpecies.heatCapacity[0],s.gasSpecies.gasConstant[0]};
  ck(std::memcmp(previous,actual,sizeof(previous))==0,"invalid quotient published primitives");
  if(!context)continue;
  ck(hasFatal(&d)&&d.firstFatal.code==Code::NonFinite,"nonfinite quotient is not fatal");
  ck(d.firstFatal.sourceSiteId==2620&&d.firstFatal.operatorId==30&&d.firstFatal.entityKind==EntityKind::Cell&&d.firstFatal.entityIndex==0&&d.firstFatal.component==0,"quotient producer location lost");
  ck(d.firstFatal.quantityId==Quantity::MassFraction&&std::isinf(d.firstFatal.observed)&&std::signbit(d.firstFatal.observed)==(SIGN<0),"actual signed infinite mass fraction lost");
  ck(d.firstFatal.step==7&&d.firstFatal.simulationTime==.125,"quotient operation metadata lost");
  ck(d.warnings[2].count==0,"nonfinite quotient incorrectly became FiniteYClip");
  for(bool warnings:{false,true}){
   HostSink sink{};sink.warningsEnabled=warnings;
   sink.fatal=[](void*,const Record&,const SourceDescriptor*){throw 19;};
   bool stopped=false;try{dispatch(takeSnapshot(d),sink);}catch(int code){stopped=code==19;}
   ck(stopped,"warning output toggle suppressed physical error");
  }
 }
}
'''.replace('NATIVE', 'true' if native else 'false').replace('SIGN', str(sign)))


@pytest.mark.parametrize("native", [False, True])
def test_finite_roundoff_clip_continues_identically_with_warning_output_toggle(tmp_path, native):
    compile_probe(tmp_path, fixture() + r'''
#include "DiagnosticsHost.H"
int main(){using namespace ugkp::diagnostics;
 Real reference[8]{};
 for(bool enabled:{false,true}){
  State s;initialise(s);DeviceContext d{};s.diagnostics=&d;
  const Real eps=Real(8)*std::numeric_limits<Real>::epsilon();
  s.rho[0]=1;s.rhoUx[0]=s.rhoUy[0]=s.rhoUz[0]=0;
  s.gasSpecies.rho[0]=-eps;s.gasSpecies.rho[s.nCells]=Real(1)+eps;
  const Real clipped[2]={0,1};s.rhoE[0]=ugkwp::mixtureEnergy(clipped,Real(700),s.gasSpecies.thermo);
  recoverGasPrimitiveCell(s,0,NATIVE);
  ck(!hasFatal(&d)&&s.gasSpecies.cellStatus[0]==0,"finite roundoff clipping was made fatal");
  ck(d.warnings[2].count==2,"finite component clipping warning missing");
  ck(s.gasSpecies.rho[0]==-eps&&s.gasSpecies.rho[s.nCells]==Real(1)+eps,"recovery rewrote conserved species");
  ck(std::abs(s.Tgas[0]-700)<Real(.01)&&std::isfinite(s.p[0])&&s.p[0]>0,"finite clipped state not recovered");
  const Real outputs[]={s.Ux[0],s.Uy[0],s.Uz[0],s.p[0],s.Tgas[0],s.gasSpecies.soundSpeed[0],s.gasSpecies.heatCapacity[0],s.gasSpecies.gasConstant[0]};
  if(enabled)ck(std::memcmp(reference,outputs,sizeof(outputs))==0,"warning toggle changed computed fields");
  else std::memcpy(reference,outputs,sizeof(outputs));
  unsigned calls=0;HostSink sink{};sink.user=&calls;sink.warningsEnabled=enabled;
  sink.warning=[](void* user,const WarningSummary&,const SourceDescriptor*){++*static_cast<unsigned*>(user);};
  const auto snapshot=takeSnapshot(d);dispatch(snapshot,sink);
  ck(calls==(enabled?1u:0u)&&snapshot.warnings[0].count==2,"warning toggle must affect output only");
 }
}
'''.replace('NATIVE', 'true' if native else 'false'))


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("thermo", ["linear", "nasa7"])
def test_wall_cp_arithmetic_overflow_keeps_infinity_quantity_and_face(tmp_path, native, thermo):
    compile_probe(tmp_path, wall_fixture() + r'''
#include "DiagnosticsHost.H"
int main(){using namespace ugkp::diagnostics;
 State s;wall(s);DeviceContext d{};reset(d,31,4,12,.75);s.diagnostics=&d;
 auto* coefficients=const_cast<Real*>(s.gasSpecies.thermo.coefficients);
 coefficients[1]=(std::numeric_limits<Real>::max)();
 Real nasa[14]{};
 if(NASA){
  auto* data=const_cast<ugkwp::SpeciesThermoData<Real>*>(s.gasSpecies.thermo.species);
  data[0].model=ugkwp::SpeciesThermoModel::NASA7;data[0].midTemperature=1000;
  nasa[0]=(std::numeric_limits<Real>::max)();nasa[7]=1000;
  data[1].coefficientOffset=7;s.gasSpecies.thermo.coefficients=nasa;s.gasSpecies.thermo.coefficientCount=14;
 }
 ck(std::isfinite(coefficients[0])&&std::isfinite(coefficients[1]),"overflow fixture coefficient is not finite");
 Real mu=77,k=78,q=79;int active=1;bool evaluated=true;
 gasFaceSubgridTransportProperties(s,1,0,-1,2,Real(1),mu,k,q,active,NATIVE,&evaluated);
 ck(hasFatal(&d)&&!evaluated&&mu==0&&k==0&&q==0&&active==0,"Cp overflow did not stop physical producer");
 const auto& r=d.firstFatal;
 ck(r.code==Code::NonFinite&&r.sourceSiteId==2658&&r.operatorId==26,"Cp failure source lost");
 ck(r.entityKind==EntityKind::Face&&r.entityIndex==1&&r.component==0,"Cp failure actual face/species lost");
 ck(std::isinf(r.observed)&&r.observed>0,"actual overflowing Cp replaced by synthetic NaN");
 ck(r.quantityId==13&&std::strcmp(quantityName(r.quantityId),"heat capacity")==0,"Cp quantity missing");
 ck(r.step==12&&r.simulationTime==.75&&r.operationSequence==31,"Cp operation provenance lost");
 ck(s.gasSpecies.faceStatus[1]==int(ugkwp::GasTransportCode::InvalidThermodynamics),"Cp error lost coarse channel");
}
'''.replace('NATIVE', 'true' if native else 'false').replace('if(NASA)', 'if(true)' if thermo == 'nasa7' else 'if(false)'))


def test_actual_cuda_recovery_and_wall_transport_compile(tmp_path):
    nvcc = os.environ.get("NVCC") or shutil.which("nvcc")
    if not nvcc:
        pytest.skip("nvcc unavailable; host helper execution does not prove CUDA compilation")
    pre = (HERE / "gas_state_probe.cpp").read_text().split("void hashBytes")[0]
    for line in ["#define __host__\n", "#define __global__\n", "#define __device__\n",
                 "#define __forceinline__ inline\n", "#define asm(...) std::abort()\n",
                 "struct ThreadIndex { int x=0; } blockIdx, threadIdx, blockDim;\n"]:
        pre = pre.replace(line, "")
    source = tmp_path / "actual-producers.cu"
    source.write_text(pre + r'''
template<int Ns>struct State:ugkwp::GasStateView<Real,Time,ugkwp::SstCoefficients>{ugkwp::GasSpeciesState<Real,Ns> gasSpecies;};
template<class S>__device__ void producers(S& s){
 recoverGasPrimitiveCell(s,0,false);recoverGasPrimitiveCell(s,0,true);
 Real mu,k,q;int active;bool evaluated;
 gasFaceSubgridTransportProperties(s,1,0,-1,2,Real(1),mu,k,q,active,false,&evaluated);
 gasFaceSubgridTransportProperties(s,1,0,-1,2,Real(1),mu,k,q,active,true,&evaluated);
}
__global__ void mixture2(State<2>*s){producers(*s);}
__global__ void mixture10(State<10>*s){producers(*s);}
''')
    command = [nvcc, "-std=c++17", "-arch=sm_89", "-I"+str(HERE),
               "-I"+str(ROOT/"common"), "-I"+str(ROOT/"common/gasNumerics"),
               "-c", str(source), "-o", str(tmp_path/"actual-producers.o")]
    result = subprocess.run(command, capture_output=True, text=True)
    (tmp_path/"cuda-build.log").write_text(" ".join(command)+"\n"+result.stdout+result.stderr)
    assert result.returncode == 0, result.stdout+result.stderr
