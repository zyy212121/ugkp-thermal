"""Execute production particle copy/swap/transfer and CHT restart codecs on HOST."""
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from pathlib import Path
import importlib.util
import json
import subprocess

ROOT = SOURCE_ROOT
NAME = "pThermalEnthalpyResidual"


def contract_module():
    spec = importlib.util.spec_from_file_location("particle_fields", (SOURCE_ROOT / 'tool/particle_field_contract.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_noncold_carry_lifecycle_preserves_sub_float_bits(tmp_path):
    meta = json.loads((ROOT / "common/ParticleFieldManifest.json").read_text())
    fields = {field["name"]: field for field in meta["fields"]}
    assert NAME in fields, "CHT thermal carry must have an explicit persistent lifecycle field"
    assert fields[NAME]["storage"] == {"CHT": "double"}
    assert fields[NAME]["host"] == {"CHT": "double"}
    assert fields[NAME]["compact"] == "compactPThermalEnthalpyResidual"
    module = contract_module()
    code = module.build_probe(ROOT, meta, "CHT")
    # The real allocation, transfer, copy and swap expressions are extracted
    # by build_probe; use a value that a float roundtrip cannot preserve.
    index = [field for field in meta["fields"] if "CHT" in field["storage"]].index(fields[NAME])
    code = code.replace(f"static_cast<double>({index+1}*7+j)", "static_cast<double>(0.00030000000000000003 + j*0.00010000000000000002)")
    code = code.replace("copyCellLocalParticle(*s,1,9,0);", """
for (int state = 0; state < 4; ++state) {
    s->pStuck[1] = state;
    copyCellLocalParticle(*s,1,9,0);
    if (s->compactPThermalEnthalpyResidual[0] != s->pThermalEnthalpyResidual[1])
        throw std::runtime_error("thermal carry lost across mobile/contact/deposit copy");
}
s->pStuck[1] = 2;
copyCellLocalParticle(*s,1,9,0);
""")
    source, binary = tmp_path / f"lifecycle{64}.cpp", tmp_path / f"lifecycle{64}"
    source.write_text(code)
    subprocess.run(["g++", "-std=c++17", "-O0", "-I"+str(tmp_path), "-I" + str(ROOT / "common"), str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary), str(tmp_path / f"lifecycle{64}.wire")], check=True)


def test_native_schema_preserves_noncold_carry_and_raw_nodes(tmp_path):
    header = (ROOT / "applications/CHT/gpu/GpuChtParticleRestart.H").read_text()
    assert "writeCurrent" in header, "exact thermal carry persistence needs its own wire version"
    meta = json.loads((ROOT / "common/ParticleFieldManifest.json").read_text())
    fields = [f for f in meta["fields"] if "CHT" in f["host"] and f["schema7"] is not None]
    def wire_type(field):
        return {"unsigned long long": "std::uint64_t", "unsigned char": "std::uint8_t", "int": "std::int32_t"}.get(field["host"]["CHT"], field["host"]["CHT"])
    declarations = "\n".join(f"std::vector<{wire_type(f)}> {f['name']}(3*{f['width']}, static_cast<{wire_type(f)}>(1));" for f in fields)
    assignments = "\n".join(f"in.{f['name']}={f['name']}.data(); out.{f['name']}={f['name']}.data();" for f in fields)
    code = r'''
#include "applications/CHT/gpu/GpuChtParticleRestart.H"
#include "validation/tests/fixtures/shared_operators/legacy_restart_writer.hpp"
#include <cassert>
#include <cmath>
#include <sstream>
#include <vector>
int main() {
DECLARATIONS
  using namespace Foam::gpuFshRestart;
  ConstView in{}; MutableView out{}; in.count=out.count=3;
ASSIGNMENTS
  double residual[3]={0.00030000000000000003,-0.00020000000000000001,std::nextafter(0.0005,1.0)};
  double restored[3]={9,9,9}; in.pThermalEnthalpyResidual=residual; out.pThermalEnthalpyResidual=restored;
  for(int i=0;i<24;++i)pColdNodeSpecificEnthalpy[i]=std::nextafter(3000000.+i,4000000.);
  const auto expectedNodes=pColdNodeSpecificEnthalpy;
  std::ostringstream wire(std::ios::binary); writeCurrent(wire,in);
  std::fill(pColdNodeSpecificEnthalpy.begin(),pColdNodeSpecificEnthalpy.end(),0.);
  std::istringstream input(wire.str(),std::ios::binary); std::string marker; std::size_t count; std::uint32_t bound;
  input>>marker>>count>>bound; assert(marker==currentFormatName && count==3);
  readCurrentPayload(input,bound,out);
  for(int i=0;i<3;++i) assert(restored[i]==residual[i]);
  assert(pColdNodeSpecificEnthalpy==expectedNodes);
  for(double h:pColdNodeSpecificEnthalpy)assert(h!=double(float(h)));
  assert(marker==Foam::gpuThermalRestart::native64ChtMarker);
  assert(!nativeRestartSupported(schema9FormatName));
  for (int kind=0;kind<3;++kind) {
    std::string broken=wire.str();
    if(kind==0) broken.pop_back();
    else if(kind==1) broken.push_back('x');
    else { double nan=std::nan(""); broken.replace(broken.size()-sizeof(double),sizeof(double),reinterpret_cast<const char*>(&nan),sizeof(double)); }
    std::istringstream bad(broken,std::ios::binary); bad>>marker>>count>>bound;
    bool rejected=false;try{readCurrentPayload(bad,bound,out);}catch(const std::exception&){rejected=true;}assert(rejected==(kind==0));
    if(kind==2)assert(std::isnan(restored[2]));
  }
}
'''.replace("DECLARATIONS", declarations).replace("ASSIGNMENTS", assignments)
    source, binary = tmp_path / "restart.cpp", tmp_path / "restart"
    source.write_text(code)
    subprocess.run(["g++", "-std=c++17", "-O0", "-I" + str(ROOT), str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


def test_actual_injection_initializes_reused_slot_without_other_solver_storage(tmp_path):
    code = r'''
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#define __host__
#define __device__
#define __global__
#define __forceinline__ inline
#define GPU_OPERATOR_REAL double
#define GPU_OPERATOR_TIME double
#define GPU_OPERATOR_R(x) static_cast<double>(x)
#define GPU_OPERATOR_THERMAL 0
struct Dim { int x=0; } blockIdx, threadIdx, blockDim, gridDim;
#include "GpuFrozenParticleRelease.cuh"
#include "BoundaryInjectionApi.H"
struct DeviceState {
 ugkwp::BoundaryInjectionStorage boundaryInjection;
 bool particleInjectionVelocitySpecified=false; double particleInjectionVelocity[3]{};
 int nBoundarySources=1,nFaces=1,nCells=1,particleCapacity=1,n=0;
 double injectionParcelMass=1,rhoSolid=1,TpMin=1,TpMax=4000,particleDiameterFallback=.001;
 unsigned long long rngSeed=7;
 int sourceFace[1]={0},sourceCell[1]={0};
 double sourceUx[1]={0},sourceUy[1]={0},sourceUz[1]={0},sourceTheta[1]={0};
 double gasBoundaryUx[1]={0},gasBoundaryUy[1]={0},gasBoundaryUz[1]={0},theta[1]={0};
 double Sfx[1]={0},Sfy[1]={0},Sfz[1]={0},sourceMassRate[1]={1},sourceResidualMass[1]={0};
 double sourcePx[1]={0},sourcePy[1]={0},sourcePz[1]={0},Cx[1]={0},Cy[1]={0},Cz[1]={0};
 double px[1]={0},py[1]={0},pz[1]={0},pux[1]={0},puy[1]={0},puz[1]={0};
 double pT[1]={0},pTheta[1]={0},pd[1]={0},pm[1]={0},sourceT[1]={300},sourceD[1]={.001};
 int pCellId[1]={-1},pStatus[1]={0}; unsigned long long pRng[1]={0},pOrigId[1]={0};
 int* particleCountDevice=&n;
 double pThermalEnthalpyResidual[1]={-0.00030000000000000003};
};
template<class A,class B> auto finiteOr(A a,B b)->decltype(a+b){return std::isfinite(a)?a:b;}
template<class A,class B> auto clampMin(A a,B b)->decltype(a+b){return a<b?b:a;}
template<class A,class B,class C> auto clampRange(A a,B b,C c)->decltype(a+b+c){return a<b?b:(a>c?c:a);}
int atomicAdd(int* p,int x){int old=*p;*p+=x;return old;}
int atomicSub(int* p,int x){int old=*p;*p-=x;return old;}
bool scheduledInletFaceDevice(const DeviceState&,int){return false;}
double scheduledSolidVolumeFractionDevice(const DeviceState&,double){return 0;}
unsigned long long mixSeed(unsigned long long x){return x;}
double sampleDiameterAroundDevice(const DeviceState&,double d,unsigned long long&){return d;}
#include "GpuContactAgeStorage.cuh"
#include "operators/injectBoundaryParticlesKernel.cuh"
int main(){DeviceState s;blockDim.x=1;gridDim.x=1;injectBoundaryParticlesKernel(&s,1.0,0.0);
assert(s.n==1&&s.pStatus[0]==1&&s.pT[0]==300&&s.pThermalEnthalpyResidual[0]==0.0);
struct OtherState{} other;
resetParticleThermalEnthalpyResidual(other,0,0);
copyParticleThermalEnthalpyResidual(other,0,0,0);
swapParticleThermalEnthalpyResidual(other,0);
}
'''
    (tmp_path/"cuda_runtime.h").write_text("")
    source, binary = tmp_path / "injection.cpp", tmp_path / "injection"
    source.write_text(code)
    subprocess.run(["g++", "-std=c++17", "-O0", "-I"+str(tmp_path), "-I" + str(ROOT / "common"), str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


def test_cht_direct_backend_matches_public_restart_api(tmp_path):
    source = tmp_path / "api.cpp"
    source.write_text('#include "applications/CHT/gpu/GpuBackendApi.H"\n')
    subprocess.run(["g++", "-std=c++17", "-fsyntax-only", "-I" + str(ROOT), "-I" + str(ROOT / "common"), str(source)], check=True)
    module = contract_module()
    for direction, expected in (("Upload", 30), ("Download", 31)):
        name = f"ugkwpGpuResidentStrict{direction}ParticleRestartMirror"
        for filename in ("common/thermal/GpuBackendApi.H", "applications/CHT/gpu/GpuResidentStrict.H", "common/thermal/GpuResidentStrict.cu"):
            source = (ROOT / filename).read_text()
            matches=list(module.calls(source,name))
            assert matches, (filename, name, "missing production declaration/definition/call")
            if filename.endswith(".cu"):
                assert module.function_body(source,name), (filename,name,"missing definition body")
            for call in matches:
                args = call[call.index("(") + 1:-1]
                depth, count = 0, 1
                for char in args:
                    if char == "(":
                        depth += 1
                    elif char == ")":
                        depth -= 1
                    elif char == "," and depth == 0:
                        count += 1
                assert count == expected, (filename, name, count)


def test_raw_cold_profile_authority_and_release_without_compensation(tmp_path):
    # Execute actual native64 initialization, caloric lookup and release helpers.
    # Mobile residual remains a separate noncold model contract.
    code = r"""
#include <cassert>
#include <cmath>
#include <initializer_list>
#include "GpuParticleBoundaryModel.H"
#include "wall/GpuFiniteWallContact.H"
#include "GpuPrecisionTypes.H"
#include "wall/GpuColdWallSolidification.H"
#include "GpuParticleCaloric.H"
#define __host__
#define __device__
#define __global__
#define __forceinline__ inline
struct Dim {int x=0;}; Dim blockIdx,threadIdx,blockDim{1},gridDim{1};
#include "GpuFrozenParticleRelease.cuh"
#define GPU_OPERATOR_REAL GpuReal
#define GPU_OPERATOR_R GPU_R
struct DeviceState {
 int coldWallSolidificationEnabled=1,nFaces=1;
 double pT[1]={300},compactPT[1]={300};
 double pThermalEnthalpyResidual[1]={0},compactPThermalEnthalpyResidual[1]={0};
 unsigned char pStuck[1]={0},compactPStuck[1]={0},particleStuckCandidateMask[1]={3};
 int pStuckFaceId[1]={0},compactPStuckFaceId[1]={0};
 double pColdNodeSpecificEnthalpy[8]={},compactPColdNodeSpecificEnthalpy[8]={};
 double pColdRingSolidMass[8]={},pColdFrozenArea[1]={},pColdContactAge[1]={};
 float pCold2DNodeSpecificEnthalpy[64]={},compactPCold2DNodeSpecificEnthalpy[64]={};
 Foam::gpuThermal::ColdWallSolidificationParameters coldWallSolidificationParameters
 {2327,20,1e6,3900,1000,5,.1,1e-6,0,3};
};
#define GPU_PAIRED_PARTICLE_HEAT
#include <cstdlib>
#define asm(...) std::abort()
#include "operators/clearColdWallParticleState.cuh"
#include "GpuParticleCaloricDevice.cuh"
int main(){
 for(double temperature:{2200.,2327.,3000.}){
  DeviceState s;s.pT[0]=temperature;s.pThermalEnthalpyResidual[0]=.0003;
  const double mobile=particleCaloricDevice(s,0).enthalpy(temperature);
  assert(particleEffectiveSpecificEnthalpyDevice(s,0)==mobile+.0003);
  s.pStuck[0]=1;
  initialiseColdWallParticleState(s,0,temperature);
  assert(s.pThermalEnthalpyResidual[0]==0.);
  double mean=0;for(double h:s.pColdNodeSpecificEnthalpy)mean+=h/8.;
  s.pThermalEnthalpyResidual[0]=123.; // stale shared noncold state cannot alter cold authority
  assert(particleEffectiveSpecificEnthalpyDevice(s,0)==mean);
  const ugkwp::ParticleCaloric freeLaw;
  publishFrozenReleaseEnergy(s,0,mean,true);
  assert(s.pThermalEnthalpyResidual[0]==0.);
  assert(s.pT[0]==freeLaw.temperature(mean));
  clearColdWallParticleState(s,0);
  for(double h:s.pColdNodeSpecificEnthalpy)assert(h==0.);
  publishFrozenReleaseEnergy(s,0,mean,false);
  assert(s.pThermalEnthalpyResidual[0]==mean-freeLaw.enthalpy(s.pT[0]));
 }
}
"""
    (tmp_path/"cuda_runtime.h").write_text("")
    source=tmp_path/"raw.cpp";source.write_text(code)
    binary=tmp_path/"raw"
    subprocess.run(["g++","-std=c++17","-O0","-I"+str(tmp_path),"-I"+str(ROOT/"common"),str(source),"-o",str(binary)],check=True)
    subprocess.run([str(binary)],check=True)


def test_native_schema_preserves_noncold_temperature_and_carry(tmp_path):
    header = (ROOT / "applications/CHT/gpu/GpuChtParticleRestart.H").read_text()
    assert "prepareSchema8ThermalAuthorityForPrecision" not in header
    meta = json.loads((ROOT / "common/ParticleFieldManifest.json").read_text())
    fields = [f for f in meta["fields"] if "CHT" in f["host"] and f["schema7"] is not None]
    def wire_type(field):
        return {"unsigned long long": "std::uint64_t", "unsigned char": "std::uint8_t", "int": "std::int32_t"}.get(field["host"]["CHT"], field["host"]["CHT"])
    declarations = "\n".join(f"std::vector<{wire_type(f)}> {f['name']}({f['width']}, static_cast<{wire_type(f)}>(1));" for f in fields)
    assignments = "\n".join(f"in.{f['name']}={f['name']}.data(); out.{f['name']}={f['name']}.data();" for f in fields)
    code = r'''
#include "applications/CHT/gpu/GpuChtParticleRestart.H"
#include "validation/tests/fixtures/shared_operators/legacy_restart_writer.hpp"
#include "GpuParticleCaloric.H"
#include <cassert>
#include <cmath>
#include <limits>
#include <sstream>
#include <vector>
using Foam::gpuFshRestart::MutableView;
void wireRoundtrip(double& temperature,double& residual){
 DECLARATIONS
 Foam::gpuFshRestart::ConstView in{};MutableView out{};in.count=out.count=1;
 ASSIGNMENTS
 pT[0]=temperature;double restored=0;in.pThermalEnthalpyResidual=&residual;out.pThermalEnthalpyResidual=&restored;
 std::ostringstream wire(std::ios::binary);Foam::gpuFshRestart::writeCurrent(wire,in);
 pT[0]=-1;
 std::istringstream stream(wire.str(),std::ios::binary);std::string marker;std::size_t count;std::uint32_t bound;
 stream>>marker>>count>>bound;assert(marker==Foam::gpuFshRestart::currentFormatName&&count==1);
 Foam::gpuFshRestart::readCurrentPayload(stream,bound,out);
 assert(pT[0]==temperature&&restored==residual);temperature=pT[0];residual=restored;
}
int main(){
 const ugkwp::ParticleCaloric caloric;
 const unsigned char masks[3]={1,3,4};
 // Both sides of each ordinary piecewise endpoint, plus the reported7.6J case.
 for(double wireTemperature:{300.00001,2326.99995,2327.00005,3000.00005,3999.99995,4000.00005,4500.00005}) {
  for(double residual:{.00030000000000000003,-.00020000000000000001}) {
   for(int kind:{0,1,2,3}) {
    double temperature=wireTemperature,carry=residual;
    unsigned char stuck=kind==0?0:2;std::int32_t face=kind==0?-1:kind-1;
    MutableView view{};view.count=1;view.pT=&temperature;view.pThermalEnthalpyResidual=&carry;view.pStuck=&stuck;view.pStuckFaceId=&face;
    wireRoundtrip(temperature,carry);
    assert(temperature==wireTemperature && carry==residual);
    // The direct double backend needs no host temperature or carry rebase.
    wireRoundtrip(temperature,carry);
    assert(temperature==wireTemperature && carry==residual);
   }
  }
 }
}
'''.replace("DECLARATIONS", declarations).replace("ASSIGNMENTS", assignments)
    source, binary = tmp_path / "precision_restart.cpp", tmp_path / "precision_restart"
    source.write_text(code)
    subprocess.run(["g++", "-std=c++17", "-O0", "-I" + str(ROOT / "common"), "-I" + str(ROOT), str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
    host = (ROOT / "applications/CHT/gpu/GpuResidentStrict.H").read_text()
    assert "readNativePayload" in host
    assert "prepareSchema8ThermalAuthorityForPrecision" not in host


def test_lifecycle_probe_detects_missing_shared_allocation(monkeypatch):
    import pytest
    module=contract_module()
    owner=ROOT/'common/thermal/GpuResidentStrict.cu'
    original=Path.read_text
    text=owner.read_text()
    allocation=next(call for call in module.calls(text,'allocate') if 's->pThermalEnthalpyResidual,' in call)
    def mutated(path,*args,**kwargs):
        content=original(path,*args,**kwargs)
        return content.replace(allocation,'0') if path==owner else content
    monkeypatch.setattr(Path,'read_text',mutated)
    meta=json.loads((ROOT/'common/ParticleFieldManifest.json').read_text())
    with pytest.raises(ValueError,match='allocation closure for pThermalEnthalpyResidual'):
        module.build_probe(ROOT,meta,'CHT')


def test_function_extractor_rejects_call_without_definition():
    import pytest
    module=contract_module()
    with pytest.raises(ValueError,match='missing function'):
        module.function_body('if (owner(a) != 0) { return 1; }','owner')
    assert 'return a;' in module.function_body('if (owner(a) != 0) { return 1; } int owner(int a) { return a; }','owner')
