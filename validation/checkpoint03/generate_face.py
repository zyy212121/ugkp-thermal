"""Generate only. Never imports/runs the test module or a numerical executable."""
from pathlib import Path
import ast,argparse,re
ap=argparse.ArgumentParser()
ap.add_argument("--root",type=Path,required=True)
ap.add_argument("--output",type=Path,required=True)
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
 body=body.replace("GasSpeciesState<Real,2>","GasSpeciesState<Real,2,double>")
else:
 body=body.replace("GasStateView<Real,Time,ugkwp::SstCoefficients>","GasStateView<Real,Time,ugkwp::SstCoefficients,double>")
body=body.replace("SpeciesThermoData<Real>","SpeciesThermoData<double>").replace("auto*coeff=new Real[6]","auto*coeff=new double[6]")
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
last=snippet.rfind("}")
snippet=snippet[:last]+'emit("first16",first16);emit("first32",first32);emit("muscl16",muscl16);emit("muscl32",muscl32);emit("muscl64",muscl64);'+snippet[last:]
a.output.write_text(pre+body+"\nnamespace cp3_refinement {\n"+large+snippet+"\n}\n"+main)
