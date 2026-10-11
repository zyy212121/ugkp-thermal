"""Execute production particle copy/swap/transfer and CHT restart codecs on HOST."""
from pathlib import Path
import importlib.util
import json
import subprocess

ROOT = Path(__file__).resolve().parents[2]
NAME = "pThermalEnthalpyResidual"


def contract_module():
    spec = importlib.util.spec_from_file_location("particle_fields", ROOT / "tools/particle_field_contract.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_carry_is_cht_only_and_real_lifecycle_preserves_sub_float_bits(tmp_path):
    meta = json.loads((ROOT / "common/ParticleFieldManifest.json").read_text())
    fields = {field["name"]: field for field in meta["fields"]}
    assert NAME in fields, "CHT thermal carry must have an explicit persistent lifecycle field"
    assert fields[NAME]["storage"] == {"CHT": "double"}
    assert fields[NAME]["host"] == {"CHT": "double"}
    assert fields[NAME]["compact"] == "compactPThermalEnthalpyResidual"
    module = contract_module()
    for bits in (32, 64):
        code = module.build_probe(ROOT, meta, "CHT", bits)
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
        source, binary = tmp_path / f"lifecycle{bits}.cpp", tmp_path / f"lifecycle{bits}"
        source.write_text(code)
        subprocess.run(["g++", "-std=c++17", "-O0", "-I" + str(ROOT / "common"), str(source), "-o", str(binary)], check=True)
        subprocess.run([str(binary), str(tmp_path / f"lifecycle{bits}.wire")], check=True)


def test_schema8_preserves_carry_and_schema7_initializes_zero(tmp_path):
    header = (ROOT / "applications/CHT/gpu/GpuChtParticleRestart.H").read_text()
    assert "writeV8" in header, "exact thermal carry persistence needs its own wire version"
    meta = json.loads((ROOT / "common/ParticleFieldManifest.json").read_text())
    fields = [f for f in meta["fields"] if "CHT" in f["host"] and f["schema7"] is not None]
    def wire_type(field):
        return {"unsigned long long": "std::uint64_t", "unsigned char": "std::uint8_t", "int": "std::int32_t"}.get(field["host"]["CHT"], field["host"]["CHT"])
    declarations = "\n".join(f"std::vector<{wire_type(f)}> {f['name']}(3*{f['width']}, static_cast<{wire_type(f)}>(1));" for f in fields)
    assignments = "\n".join(f"in.{f['name']}={f['name']}.data(); out.{f['name']}={f['name']}.data();" for f in fields)
    code = r'''
#include "applications/CHT/gpu/GpuChtParticleRestart.H"
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
  std::ostringstream wire(std::ios::binary); writeV8(wire,in);
  std::istringstream input(wire.str(),std::ios::binary); std::string marker; std::size_t count; std::uint32_t bound;
  input>>marker>>count>>bound; assert(marker==schema8FormatName && count==3);
  readV8Payload(input,bound,out);
  for(int i=0;i<3;++i) assert(restored[i]==residual[i]);
  std::ostringstream old(std::ios::binary); writeV7(old,in);
  std::istringstream legacy(old.str(),std::ios::binary); legacy>>marker>>count>>bound;
  readV7Payload(legacy,bound,out);
  for(double value:restored) assert(value==0.0);
  for (int kind=0;kind<3;++kind) {
    std::string broken=wire.str();
    if(kind==0) broken.pop_back();
    else if(kind==1) broken.push_back('x');
    else { double nan=std::nan(""); broken.replace(broken.size()-sizeof(double),sizeof(double),reinterpret_cast<const char*>(&nan),sizeof(double)); }
    std::istringstream bad(broken,std::ios::binary); bad>>marker>>count>>bound;
    bool rejected=false;try{readV8Payload(bad,bound,out);}catch(const std::exception&){rejected=true;}assert(rejected);
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
#define GPU_OPERATOR_REAL float
#define GPU_OPERATOR_TIME double
#define GPU_OPERATOR_R(x) static_cast<float>(x)
#define GPU_OPERATOR_THERMAL 0
struct Dim { int x=0; } blockIdx, threadIdx, blockDim;
struct DeviceState {
 int nBoundarySources=1,nFaces=1,nCells=1,particleCapacity=1,n=0;
 float injectionParcelMass=1,rhoSolid=1,TpMin=1,TpMax=4000,particleDiameterFallback=.001;
 unsigned long long rngSeed=7;
 int sourceFace[1]={0},sourceCell[1]={0};
 float sourceUx[1]={0},sourceUy[1]={0},sourceUz[1]={0},sourceTheta[1]={0};
 float gasBoundaryUx[1]={0},gasBoundaryUy[1]={0},gasBoundaryUz[1]={0},theta[1]={0};
 float Sfx[1]={0},Sfy[1]={0},Sfz[1]={0},sourceMassRate[1]={1},sourceResidualMass[1]={0};
 float sourcePx[1]={0},sourcePy[1]={0},sourcePz[1]={0},Cx[1]={0},Cy[1]={0},Cz[1]={0};
 float px[1]={0},py[1]={0},pz[1]={0},pux[1]={0},puy[1]={0},puz[1]={0};
 float pT[1]={0},pTheta[1]={0},pd[1]={0},pm[1]={0},sourceT[1]={300},sourceD[1]={.001};
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
float scheduledSolidVolumeFractionDevice(const DeviceState&,double){return 0;}
unsigned long long mixSeed(unsigned long long x){return x;}
float sampleDiameterAroundDevice(const DeviceState&,float d,unsigned long long&){return d;}
#include "GpuContactAgeStorage.cuh"
#include "operators/injectBoundaryParticlesKernel.cuh"
int main(){DeviceState s;blockDim.x=1;injectBoundaryParticlesKernel(&s,1.0,0.0);
assert(s.n==1&&s.pStatus[0]==1&&s.pT[0]==300&&s.pThermalEnthalpyResidual[0]==0.0);
struct OtherState{} other;
resetParticleThermalEnthalpyResidual(other,0,0);
copyParticleThermalEnthalpyResidual(other,0,0,0);
swapParticleThermalEnthalpyResidual(other,0);
}
'''
    source, binary = tmp_path / "injection.cpp", tmp_path / "injection"
    source.write_text(code)
    subprocess.run(["g++", "-std=c++17", "-O0", "-I" + str(ROOT / "common"), str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


def test_cht_dispatch_matches_public_restart_api():
    subprocess.run([
        "g++", "-std=c++17", "-fsyntax-only", "-I" + str(ROOT / "common"),
        "-I" + str(ROOT / "applications/CHT/gpu"),
        "-include", str(ROOT / "applications/CHT/gpu/GpuBackendApi.H"),
        str(ROOT / "applications/CHT/gpu/GpuPrecisionDispatch.C"),
    ], check=True)
    module = contract_module()
    for direction, expected in (("Upload", 30), ("Download", 31)):
        name = f"ugkwpGpuResidentStrict{direction}ParticleRestartMirror"
        for filename in ("GpuBackendApi.H", "GpuResidentStrict.H", "GpuResidentStrict.cu"):
            source = (ROOT / "applications/CHT/gpu" / filename).read_text()
            for call in module.calls(source, name):
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


def test_actual_cold_profile_transitions_preserve_increment_and_legacy_reference(tmp_path):
    """No transition-law repair: compare actual ±carry routes with zero-carry twins.

    Schema7 supplies the initial nonzero temperature and zero carry. The test then
    seeds a small represented increment, runs real initialization/clear/detach
    bodies, and separates that increment from the pre-existing caloric-law jump.
    """
    def function(source, name):
        start = source.index(name)
        brace = source.index("{", start)
        depth, end = 1, brace + 1
        while depth:
            depth += (source[end] == "{") - (source[end] == "}")
            end += 1
        return source[start:end]

    source2d = (ROOT / "common/wall/GpuColdWall2DDevice.cuh").read_text()
    bodies2d = "\n".join(function(source2d, "__device__ void " + name) for name in (
        "clearColdWall2DParticleState", "initialiseColdWall2DParticleState"))
    contact = (ROOT / "common/operators/FiniteContactRelaxation.inl").read_text()
    detach = function(contact, "if (detach)")
    meta = json.loads((ROOT / "common/ParticleFieldManifest.json").read_text())
    fields = [f for f in meta["fields"] if "CHT" in f["host"] and f["schema7"] is not None]
    def wire_type(field):
        return {"unsigned long long": "std::uint64_t", "unsigned char": "std::uint8_t", "int": "std::int32_t"}.get(field["host"]["CHT"], field["host"]["CHT"])
    declarations = "\n".join(f"std::vector<{wire_type(f)}> {f['name']}({f['width']}, static_cast<{wire_type(f)}>(1));" for f in fields)
    assignments = "\n".join(f"in.{f['name']}={f['name']}.data(); out.{f['name']}={f['name']}.data();" for f in fields)
    code = r'''
#include "GpuPrecisionTypes.H"
#include "wall/GpuColdWallSolidification.H"
#include "wall/GpuColdWall2DSolidification.H"
#include "wall/GpuFiniteWallContact.H"
#include "GpuParticleCaloric.H"
#include "applications/CHT/gpu/GpuChtParticleRestart.H"
#include <cassert>
#include <cmath>
#include <sstream>
#include <vector>
#define __device__
#define __forceinline__ inline
#define GPU_OPERATOR_REAL GpuReal
#define GPU_OPERATOR_R(x) GPU_R(x)
#define GPU_RESET_CONTACT_AGE(s,i) s.pContactAge[i]=0;
struct InvalidTransition {};
#define asm(...) throw InvalidTransition{}
struct DeviceState {
 int coldWallSolidificationEnabled=1,coldWall2DEnabled=1,nFaces=1;
 GpuReal pT[1]={GPU_R(300)},compactPT[1]={GPU_R(300)};
 double pThermalEnthalpyResidual[1]={0},compactPThermalEnthalpyResidual[1]={0};
 unsigned char pStuck[1]={0},compactPStuck[1]={0},particleStuckCandidateMask[1]={0};
 int pStuckFaceId[1]={-1},compactPStuckFaceId[1]={-1};
 float pColdNodeSpecificEnthalpy[8]={},compactPColdNodeSpecificEnthalpy[8]={};
 float pColdRingSolidMass[8]={},pColdFrozenArea[1]={};double pColdContactAge[1]={};
 float pCold2DNodeSpecificEnthalpy[64]={},compactPCold2DNodeSpecificEnthalpy[64]={};
 double pCold2DRingContactAge[8]={};float pCold2DFrozenArea[1]={};
 GpuReal pux[1]={},puy[1]={},puz[1]={},puxOld[1]={GPU_R(1)},puyOld[1]={GPU_R(2)},puzOld[1]={GPU_R(3)},pTheta[1]={};
 double pContactAge[1]={};float pDepositionArea[1]={},pContactDuration[1]={},pContactMaximumArea[1]={},pContactPeakFraction[1]={};
 Foam::gpuThermal::ColdWallSolidificationParameters coldWallSolidificationParameters
  {GPU_R(2327),GPU_R(20),GPU_R(1e6),GPU_R(3900),GPU_R(1000),GPU_R(5),GPU_R(.1),GPU_R(1e-6),0,3};
};
#include "operators/clearColdWallParticleState.cuh"
BODIES2D
#include "GpuParticleCaloricDevice.cuh"
void detachParticle(DeviceState& s,int i){const bool detach=true;DETACH}

double schema7Temperature(const double temperature) {
 DECLARATIONS
 Foam::gpuFshRestart::ConstView in{};Foam::gpuFshRestart::MutableView out{};in.count=out.count=1;
 ASSIGNMENTS
 pT[0]=temperature;double carry=9;out.pThermalEnthalpyResidual=&carry;
 std::ostringstream old(std::ios::binary);Foam::gpuFshRestart::writeV7(old,in);
 pT[0]=-1;
 std::istringstream stream(old.str(),std::ios::binary);std::string marker;std::size_t count;std::uint32_t bound;
 stream>>marker>>count>>bound;assert(marker==Foam::gpuFshRestart::schema7FormatName&&count==1);
 Foam::gpuFshRestart::readV7Payload(stream,bound,out);
 assert(pT[0]==temperature&&carry==0.0);return pT[0];
}
void sameMaterialRepresentation(const DeviceState& baseline,const DeviceState& carried,double increment) {
 assert(baseline.pT[0]==carried.pT[0]);
 assert(baseline.pThermalEnthalpyResidual[0]==0&&carried.pThermalEnthalpyResidual[0]==increment);
 assert(particleMaterialEnthalpyBaseDevice(baseline,0)==particleMaterialEnthalpyBaseDevice(carried,0));
 assert(particleCaloricDevice(baseline,0).specificHeat(double(baseline.pT[0]))==particleCaloricDevice(carried,0).specificHeat(double(carried.pT[0])));
 assert(particleCaloricDevice(baseline,0).phaseChange==particleCaloricDevice(carried,0).phaseChange);
 for(int n=0;n<8;++n){
  assert(baseline.pColdNodeSpecificEnthalpy[n]==carried.pColdNodeSpecificEnthalpy[n]);
  assert(baseline.pColdRingSolidMass[n]==carried.pColdRingSolidMass[n]);
  assert(baseline.pCold2DRingContactAge[n]==carried.pCold2DRingContactAge[n]);
 }
 assert(baseline.pColdFrozenArea[0]==carried.pColdFrozenArea[0]&&baseline.pColdContactAge[0]==carried.pColdContactAge[0]&&baseline.pCold2DFrozenArea[0]==carried.pCold2DFrozenArea[0]);
 for(int n=0;n<64;++n)assert(baseline.pCold2DNodeSpecificEnthalpy[n]==carried.pCold2DNodeSpecificEnthalpy[n]);
 const double effectiveIncrement=particleEffectiveSpecificEnthalpyDevice(carried,0)-particleEffectiveSpecificEnthalpyDevice(baseline,0);
 assert(std::abs(effectiveIncrement-increment)<2e-9);
}
int main(){
 using namespace Foam::gpuThermal;
 for(double originalTemperature:{2200.0,2327.0,3000.0}) {
  const GpuReal temperature=GpuReal(schema7Temperature(originalTemperature));
  for(double increment:{0.00030000000000000003,-0.00020000000000000001}) {
   for(int dimensions:{1,2}) {
    DeviceState baseline,carried;baseline.pT[0]=carried.pT[0]=temperature;
    carried.pThermalEnthalpyResidual[0]=increment;
    sameMaterialRepresentation(baseline,carried,increment);
    const double mobileBase=particleMaterialEnthalpyBaseDevice(baseline,0);
    const double mobileEffective=particleEffectiveSpecificEnthalpyDevice(carried,0);
    for(DeviceState* state:{&baseline,&carried}) {
     state->pStuck[0]=particleWallTransientRebound;state->pStuckFaceId[0]=0;
     state->particleStuckCandidateMask[0]=dimensions==1?particleWallSolidifyingDeposition:particleWallColdWall2D;
     if(dimensions==1){initialiseColdWallParticleState(*state,0,temperature);clearColdWall2DParticleState(*state,0);}
     else {clearColdWallParticleState(*state,0);initialiseColdWall2DParticleState(*state,0,temperature);}
    }
    // A mutation that erases carry on either transition must fail this probe.
    MUTATION
    sameMaterialRepresentation(baseline,carried,increment);
    assert(particleCaloricDevice(carried,0).phaseChange);
    const double coldBase=particleMaterialEnthalpyBaseDevice(baseline,0);
    const double coldEffective=particleEffectiveSpecificEnthalpyDevice(carried,0);
    const double legacyReferenceJump=coldBase-mobileBase;
    assert(std::abs(legacyReferenceJump)>1.0); // Explicit existing reference/model difference.
    assert(std::abs((coldEffective-mobileEffective)-legacyReferenceJump)<2e-9);
    // Run the real stuck-to-mobile block, including both real clear helpers.
    detachParticle(baseline,0);detachParticle(carried,0);
    sameMaterialRepresentation(baseline,carried,increment);
    assert(carried.pStuck[0]==particleWallMobile&&carried.pStuckFaceId[0]==-1);
    assert(!particleCaloricDevice(carried,0).phaseChange);
    assert(particleMaterialEnthalpyBaseDevice(carried,0)==mobileBase);
    assert(std::abs((particleEffectiveSpecificEnthalpyDevice(carried,0)-coldEffective)+legacyReferenceJump)<2e-9);
    for(float value:carried.pColdNodeSpecificEnthalpy)assert(value==0);
    for(float value:carried.pCold2DNodeSpecificEnthalpy)assert(value==0);
   }
  }
 }
}
'''.replace("BODIES2D", bodies2d).replace("DETACH", detach).replace("DECLARATIONS", declarations).replace("ASSIGNMENTS", assignments)
    for bits in (32, 64):
        for mutant in (False, True) if bits == 32 else (False,):
            source = tmp_path / f"transition{bits}_{mutant}.cpp"
            binary = source.with_suffix("")
            mutation = "carried.pThermalEnthalpyResidual[0]=0.0;" if mutant else ""
            source.write_text(code.replace("MUTATION", mutation))
            subprocess.run(["g++", "-std=c++17", "-O0", f"-DUGKWP_GPU_REAL_BITS={bits}", "-I" + str(ROOT / "common"), "-I" + str(ROOT), str(source), "-o", str(binary)], check=True)
            result = subprocess.run([str(binary)], capture_output=True)
            if mutant:
                assert result.returncode != 0, "transition regression did not detect erased carry"
            else:
                assert result.returncode == 0, result.stderr.decode()


def test_schema8_cross_precision_rebases_ordinary_authority_only(tmp_path):
    header = (ROOT / "applications/CHT/gpu/GpuChtParticleRestart.H").read_text()
    assert "prepareSchema8ThermalAuthorityForPrecision" in header, "schema8 FP32 upload drops H(Twire)-H(Tstored)"
    meta = json.loads((ROOT / "common/ParticleFieldManifest.json").read_text())
    fields = [f for f in meta["fields"] if "CHT" in f["host"] and f["schema7"] is not None]
    def wire_type(field):
        return {"unsigned long long": "std::uint64_t", "unsigned char": "std::uint8_t", "int": "std::int32_t"}.get(field["host"]["CHT"], field["host"]["CHT"])
    declarations = "\n".join(f"std::vector<{wire_type(f)}> {f['name']}({f['width']}, static_cast<{wire_type(f)}>(1));" for f in fields)
    assignments = "\n".join(f"in.{f['name']}={f['name']}.data(); out.{f['name']}={f['name']}.data();" for f in fields)
    code = r'''
#include "applications/CHT/gpu/GpuChtParticleRestart.H"
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
 std::ostringstream wire(std::ios::binary);Foam::gpuFshRestart::writeV8(wire,in);
 pT[0]=-1;
 std::istringstream stream(wire.str(),std::ios::binary);std::string marker;std::size_t count;std::uint32_t bound;
 stream>>marker>>count>>bound;assert(marker==Foam::gpuFshRestart::schema8FormatName&&count==1);
 Foam::gpuFshRestart::readV8Payload(stream,bound,out);
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
    const double oldBase=caloric.enthalpy(temperature);
    const double narrowed=double(float(temperature));
    if(wireTemperature==3000.00005&&kind==0)assert(std::abs(100*(caloric.enthalpy(narrowed)-oldBase))>7.0);
    wireRoundtrip(temperature,carry);
    Foam::gpuFshRestart::prepareSchema8ThermalAuthorityForPrecision(view,32,masks,3);
    assert(temperature==narrowed);
    if(kind<2){
     const double expected=residual+(oldBase-caloric.enthalpy(narrowed));
     assert(carry==expected);
     const double actualIncrement=(caloric.enthalpy(temperature)-oldBase)+(carry-residual);
     assert(std::abs(100*actualIncrement)<1e-10);
    } else {
     // Active8/64-node profiles are already wire-float authority; pT is a cache.
     assert(carry==residual);
    }
    const double floatTemperature=temperature,floatCarry=carry;
    //32→64 must preserve both components bit-for-bit, then32→64→32 is stable.
    wireRoundtrip(temperature,carry);
    Foam::gpuFshRestart::prepareSchema8ThermalAuthorityForPrecision(view,64,masks,3);
    assert(temperature==floatTemperature&&carry==floatCarry);
    Foam::gpuFshRestart::prepareSchema8ThermalAuthorityForPrecision(view,32,masks,3);
    assert(temperature==floatTemperature&&carry==floatCarry);
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
    v8 = host.index("gpuFshRestart::readV8Payload")
    v7 = host.index("else if (weightedV7)", v8)
    assert "prepareSchema8ThermalAuthorityForPrecision" in host[v8:v7]
    assert host.count("prepareSchema8ThermalAuthorityForPrecision") == 1, "legacy schemas must remain unrebased"
