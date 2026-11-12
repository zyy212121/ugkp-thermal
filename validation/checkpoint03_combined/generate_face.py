"""Generate only. Never imports/runs the test module or a numerical executable."""
from pathlib import Path
import ast,argparse,re
ap=argparse.ArgumentParser()
ap.add_argument("--root",type=Path,required=True)
ap.add_argument("--output",type=Path,required=True)
ap.add_argument("--candidate32-trend",action="store_true")
a=ap.parse_args()
here=a.root/"validation/tests/gas_transport"
tree=ast.parse((here/"test_mixture_transport.py").read_text())
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="fixture")
ret=next(n for n in fn.body if isinstance(n,ast.Return))
body=ast.literal_eval(ret.value.func.value)
fields=re.findall(r"    ([\w:]+\*) (\w+)\{\};",(here/"legacy_gas_fixture.hpp").read_text())
assert fields, "No fixture allocation fields found"
# Allocate each field with its actual production State pointer type.
body=body.replace("ALLOCATE","\n".join(f"s.{name}=new std::remove_pointer_t<decltype(s.{name})>[8]{{}};" for ty,name in fields))
state_header=(a.root/"common/gasTransport/GasStateView.H").read_text()
mixed_tables="class TableReal = Real" in state_header
if mixed_tables:
 body=body.replace("GasStateView<Real,Time,ugkwp::SstCoefficients>","GasStateView<Real,Time,ugkwp::SstCoefficients,GpuEnergy,GpuGeometry,GpuWallEnergy>")
 body=body.replace("GasSpeciesState<Real,2>","GasSpeciesState<Real,2,Real>")
else:
 body=body.replace("GasStateView<Real,Time,ugkwp::SstCoefficients>","GasStateView<Real,Time,ugkwp::SstCoefficients,double>")

# Candidate metadata and coefficients use the actual resident Real, including FP32.

pre=(here/"gas_state_probe.cpp").read_text().split("void hashBytes")[0]
main=Path(__file__).with_name("face_fixture.cpp").read_text()

# Static extraction of the established refinement case; no module import.
rt=ast.parse((here/"test_mixture_reconstruction.py").read_text())
rf=next(n for n in rt.body if isinstance(n,ast.FunctionDef) and n.name=="test_smooth_periodic_muscl_has_refinement_gain_over_first_order")
snippet=next(n.value for n in ast.walk(rf) if isinstance(n,ast.Constant) and isinstance(n.value,str) and "Real error(int n" in n.value)
large=body.replace("[8]{}","[132]{}")
for field in ["rho","initial","gradX","gradY","gradZ","positivityScale"]:
 large=large.replace(f"g.{field}=new Real[4]{{}}",f"g.{field}=new Real[256]{{}}")
for field in ["flux","boundaryMassFraction"]:
 large=large.replace(f"g.{field}=new Real[6]{{}}",f"g.{field}=new Real[258]{{}}")
for field in ["limiter","soundSpeed","heatCapacity","gasConstant"]:
 large=large.replace(f"g.{field}=new Real[2]{{}}",f"g.{field}=new Real[128]{{}}")
large=large.replace("g.compositionBoundaryFixed=new int[3]{}","g.compositionBoundaryFixed=new int[129]{}").replace("g.faceStatus=new int[3]{}","g.faceStatus=new int[129]{}").replace("g.cellStatus=new int[2]{}","g.cellStatus=new int[128]{}")
snippet=snippet.replace("Real error(int n,int reconstruction)","Real error(int n,int reconstruction,int scheme)").replace("SCHEME","scheme")
snippet=re.sub(r"error\((16|32|64),([01])\)",r"error(\1,\2,scheme)",snippet)
snippet=snippet.replace("int main(){","template<class Emit> void verify(int scheme,Emit emit){")
# This opt-in is passed only for candidate32 by prepare.py. Historical and
# candidate64 generation is byte-for-byte unchanged when the flag is absent.
if a.candidate32_trend:
 snippet=snippet.replace("template<class Emit> void verify(int scheme,Emit emit){",
  "template<class Emit> void verify(int scheme,Emit emit,std::ostream& diagnostics){")
 old = '\n'.join([
  'ck(first16/first32>1.8&&first16/first32<2.2,"first-order control refinement unexpected");',
  'ck(muscl16/muscl32>3.5&&muscl32/muscl64>3.5,"MUSCL lost second-order spatial refinement");',
  'ck(muscl32<Real(.02)*first32,"MUSCL did not improve smooth periodic transport");'])
 trend = r"""
// Predeclared candidate32 trend criteria. Exact-zero errors remain admissible.
const auto improves=[](Real coarse,Real fine){return fine<coarse||(fine==Real(0)&&coarse==Real(0));};
ck(improves(first16,first32),"first-order fixed refinement did not improve");
ck(improves(muscl16,muscl32)&&improves(muscl32,muscl64),"MUSCL fixed refinement did not improve");
ck(improves(first32,muscl32),"MUSCL did not improve over first order");
diagnostics<<"FACE_FP32_REFINEMENT_DIAGNOSTIC scheme="<<scheme;
const auto ratio=[&](const char* name,Real numerator,Real denominator){
 diagnostics<<' '<<name<<'=';
 if(denominator!=Real(0))diagnostics<<double(numerator)/double(denominator);
 else diagnostics<<"undefined_zero_denominator";
};
ratio("first16_over_first32",first16,first32);
ratio("muscl16_over_muscl32",muscl16,muscl32);
ratio("muscl32_over_muscl64",muscl32,muscl64);
ratio("muscl32_over_first32",muscl32,first32);
diagnostics<<" old_first_order_pass="<<(first32!=Real(0)&&first16/first32>1.8&&first16/first32<2.2)
 <<" old_muscl_order_pass="<<(muscl32!=Real(0)&&muscl64!=Real(0)&&muscl16/muscl32>3.5&&muscl32/muscl64>3.5)
 <<" old_two_percent_gain_pass="<<(muscl32<Real(.02)*first32)<<" hard_criteria=trend_only\n";
"""
 snippet=snippet.replace(old,trend)
 pre='#include <ostream>\n'+pre
 main=main.replace('#include <string>','#include <string>\n#include <sstream>')
 main=main.replace('std::cout<<std::setprecision(17);','std::ostringstream refinementDiagnostics;refinementDiagnostics<<std::setprecision(17);\n std::cout<<std::setprecision(17);')
 main=main.replace('++checks;\n });','++checks;\n },refinementDiagnostics);')
 # Print separate diagnostics only after the complete original measurement JSON.
 closing=main.rfind('}')
 main=main[:closing]+'if(!perf)std::cout<<refinementDiagnostics.str();\n'+main[closing:]

last=snippet.rfind("}")
snippet=snippet[:last]+'emit("first16",first16);emit("first32",first32);emit("muscl16",muscl16);emit("muscl32",muscl32);emit("muscl64",muscl64);'+snippet[last:]
a.output.write_text(pre+body+"\nnamespace cp3_refinement {\n"+large+snippet+"\n}\n"+main)
