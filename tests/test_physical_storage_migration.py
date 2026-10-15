"""Changed owner storage and restart byte boundaries; no alternate solver mode."""
import importlib.util
import json
from pathlib import Path
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[1]
PROFILES = {"pColdNodeSpecificEnthalpy", "pCold2DNodeSpecificEnthalpy"}


def test_physical_manifest_owner_widths():
    meta = json.loads((ROOT / "common/ParticleFieldManifest.json").read_text())
    codec=(ROOT/"common/GpuThermalParticleRestartCodec.H").read_text()
    for owner,size in [("CHT",613),("FSH",893)]:
        layout=meta["current_disk_layouts"][owner]
        assert layout["bytes_per_particle"]==size
        assert layout["marker"] in codec
    for f in meta["fields"]:
        for owner, typ in f["storage"].items():
            if typ in ("float", "double", "GpuReal", "GpuTime"):
                expected = "float" if owner == "CHT" and f["name"] in PROFILES else "double"
                assert {"GpuReal": "double", "GpuTime": "double"}.get(typ, typ) == expected, (owner, f["name"], typ)
                assert f["host"][owner] == expected, (owner, f["name"], "host")


def compile_run(tmp_path, code, name="probe"):
    source, exe = tmp_path / (name + ".cpp"), tmp_path / name
    source.write_text(code)
    subprocess.run(["g++", "-std=c++17", "-O0", "-I"+str(ROOT), "-I"+str(ROOT/"common"), str(source), "-o", str(exe)], check=True)
    subprocess.run([str(exe)], check=True)


def test_legacy_schema7_widths_independent_of_memory(tmp_path):
    meta = json.loads((ROOT / "common/ParticleFieldManifest.json").read_text())
    fields = sorted((f for f in meta["fields"] if f["schema7"] is not None), key=lambda f:f["schema7"])
    types = {"int":"std::int32_t", "unsigned long long":"std::uint64_t", "unsigned char":"std::uint8_t"}


    declarations=[]; writes=[]; checks=[]
    for f in fields:
        old="float" if f["schema7"] in range(16,27) else f["host"]["FSH"]; new="double" if old=="float" else types.get(old,old)
        disk=types.get(old,old); n=f["width"]
        declarations.append(f'{new}* {f["name"]}=new {new}[{n}]{{}};')
        writes.append(f'{{ {disk} x=static_cast<{disk}>({f["schema7"]+1}.25); for(int j=0;j<{n};++j) Foam::gpuThermalRestart::writeArray(w,&x,0,1); }}')
        checks.append(f'for(int j=0;j<{n};++j) assert(v.{f["name"]}[j]==static_cast<{new}>({f["schema7"]+1}.25));')
    compile_run(tmp_path, '''#include "common/GpuThermalParticleRestartCodec.H"
#include <cassert>
#include <sstream>
struct View {std::size_t count=1; DECL};
int main(){View v;std::ostringstream w(std::ios::binary); w<<'\\n';std::uint32_t n=1;
Foam::gpuThermalRestart::writeArray(w,&n,0,1);WRITES
std::istringstream r(w.str(),std::ios::binary);
Foam::gpuThermalRestart::readPayload<float>(r,262144,v);CHECKS
assert(r.peek()==std::char_traits<char>::eof());}
'''.replace("DECL", "\n".join(declarations)).replace("WRITES","\n".join(writes)).replace("CHECKS","\n".join(checks)))


def test_cht_binary_reader_does_not_fall_through_to_text():
    import re
    text = (ROOT / "applications/CHT/gpu/GpuResidentStrict.H").read_text()
    match = re.search(r'if\s*\((\s*(?:!currentLayout\s*&&\s*)?!weightedFshV1[^)]+)\)', text)
    assert match
    expression = match.group(1).replace('&&', ' and ').replace('!', ' not ')
    names = set(re.findall(r'\b(?:weighted\w+|currentLayout)\b', expression))
    for active in ("weightedV7", "weightedV8", "currentLayout"):
        flags = {name: name == active for name in names}
        assert not eval(" ".join(expression.split()), {"__builtins__": {}}, flags), active


@pytest.mark.parametrize('owner,header,versions,per_particle', [
    ('CHT','GpuChtParticleRestart.H',(1,2,3,4,5,6,7,8),613),
    ('FSH','GpuFshParticleRestart.H',(1,2,3,4,5,7),893),
])
def test_native_current_and_frozen_legacy_layouts(tmp_path,owner,header,versions,per_particle):
    import base64
    meta=json.loads((ROOT/'common/ParticleFieldManifest.json').read_text())
    fields=[f for f in meta['fields'] if owner in f['host'] and (f['schema7'] is not None or f['name']=='pThermalEnthalpyResidual')]
    types={'int':'std::int32_t','unsigned long long':'std::uint64_t','unsigned char':'std::uint8_t'}
    declarations=[];assignments=[];equal=[];legacy=[]
    fixture=json.loads((ROOT/'tests/fixtures/particle_restart_legacy_widths.json').read_text())['files']
    for i,f in enumerate(fields):
        name=f['name'];typ=types.get(f['host'][owner],f['host'][owner]);n=2*f['width']
        declarations.append(f'std::vector<{typ}> a_{name}({n}),b_{name}({n});for(int j=0;j<{n};++j)a_{name}[j]=static_cast<{typ}>({i+1}.25+j*0.125+0x1p-30);')
        assignments.append(f'in.{name}=a_{name}.data();out.{name}=b_{name}.data();')
        equal.append(f'assert(a_{name}==b_{name});')
    for v in versions:
        file=tmp_path/f'{owner}-{v}.bin';file.write_bytes(base64.b64decode(fixture[f'{owner}-{v}']))
        checks=[]
        for i,f in enumerate(fields):
            ordinal=f['schema7'];name=f['name'];typ=types.get(f['host'][owner],f['host'][owner]);n=2*f['width']
            if ordinal is None:
                if v==8:checks.append(f'for(int j=0;j<{n};++j)assert(b_{name}[j]=={i+1}.25+j*0.125);')
                elif v==7:checks.append(f'for(auto x:b_{name})assert(x==0.0);')
                continue
            if v<=3 and ordinal>{1:16,2:18,3:19}[v]:continue
            if v==4 and ordinal>23 or v in (5,6) and ordinal>26:continue
            expected='0' if name=='pStuck' else f'static_cast<{typ}>({i+1}.25+j*0.125)'
            checks.append(f'for(int j=0;j<{n};++j)assert(b_{name}[j]=={expected});')
        legacy.append(f'{{std::ifstream file("{file}",std::ios::binary);file>>marker>>count>>bound;readV{v}Payload(file,bound,out);'+''.join(checks)+'assert(file.peek()==std::char_traits<char>::eof());}')
    legacy_extra=""
    carry_include=""
    if owner=='CHT':
        import struct
        carry_include='#include "common/wall/GpuColdWallThermalCarry.cuh"'
        raw=bytearray(base64.b64decode(fixture['CHT-8']))
        raw[-16:]=struct.pack('<dd',-.00030000000000000003,.00010000000000000002)
        signed_file=tmp_path/'signed-legacy-carry.bin';signed_file.write_bytes(raw)
        legacy_extra=f'''{{std::ifstream file("{signed_file}",std::ios::binary);file>>marker>>count>>bound;readV8Payload(file,bound,out);
assert(b_pThermalEnthalpyResidual[0]==-.00030000000000000003&&b_pThermalEnthalpyResidual[1]==.00010000000000000002);
const auto oldT=b_pT;double before=b_pThermalEnthalpyResidual[0];for(int j=0;j<8;++j)before+=double(b_pColdNodeSpecificEnthalpy[j])/8.;
const auto next=ugkwp::publishColdWallUniformSpecificHeat(b_pColdNodeSpecificEnthalpy.data(),8,b_pThermalEnthalpyResidual[0],.0001);
assert(next.valid);double after=b_pThermalEnthalpyResidual[0];for(int j=0;j<8;++j)after+=double(b_pColdNodeSpecificEnthalpy[j])/8.;
assert(std::abs((after-before)-.0001)<1.e-13);assert(oldT==b_pT);assert(b_pThermalEnthalpyResidual[1]==.00010000000000000002);}}'''
    code=f'''#include "applications/{owner}/gpu/{header}"
{carry_include}
#include <cassert>
#include <fstream>
#include <sstream>
#include <vector>
int main(){{using namespace Foam::gpuFshRestart;
{''.join(declarations)}
ConstView in{{}};MutableView out{{}};in.count=out.count=2;
{''.join(assignments)}
std::ostringstream w(std::ios::binary);writeCurrent(w,in);
{{std::ofstream file("{tmp_path}/current.bin",std::ios::binary);file<<w.str();}}
std::istringstream rd(w.str(),std::ios::binary);std::string marker;size_t count;std::uint32_t bound;
rd>>marker>>count>>bound;assert(marker==currentFormatName&&count==2);
assert(w.str().size()==w.str().find('\\n')+1+4+2*{per_particle});
readCurrentPayload(rd,bound,out);{''.join(equal)}assert(rd.peek()==std::char_traits<char>::eof());
std::string broken=w.str();broken.pop_back();std::istringstream bad(broken,std::ios::binary);bad>>marker>>count>>bound;
bool truncated=false;try{{readCurrentPayload(bad,bound,out);}}catch(const std::runtime_error&){{truncated=true;}}assert(truncated);
{''.join(legacy)}
{legacy_extra}
}}'''
    compile_run(tmp_path,code,owner)
    import sys
    sys.path.insert(0,str(ROOT/'common'))
    from thermal_particle_restart_current import iter_chunks
    chunks=list(iter_chunks(tmp_path/'current.bin'))
    assert len(chunks)==1
    data=chunks[0][1]
    assert data['pColdNodeSpecificEnthalpy'].dtype.itemsize == (4 if owner=='CHT' else 8)
    assert data['pColdRingSolidMass'].dtype.itemsize==8
    assert data['pT'].dtype.itemsize==8
    assert ('pThermalEnthalpyResidual' in data)==(owner=='CHT')

    assert float(data['pColdRingSolidMass'][0,0]) % 1 == .25+2**-30



@pytest.mark.parametrize('owner',['CHT','FSH'])
def test_real_lifecycle_retains_widened_bits_at_odd_destination(tmp_path,owner):
    spec=importlib.util.spec_from_file_location('contract',ROOT/'tools/particle_field_contract.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    meta=json.loads((ROOT/'common/ParticleFieldManifest.json').read_text())
    code=module.build_probe(ROOT,meta,owner)
    fs=[f for f in meta['fields'] if owner in f['storage']]
    for i,f in enumerate(fs):
        if f['host'][owner]=='double':
            code=code.replace(f'static_cast<double>({i+1}*7+j)',f'static_cast<double>({i+1}*7+j+0x1p-30)')
    extra=''
    for f in fs:
        compact=f['compact'];width=f['width']
        extra+=f'const std::vector<std::remove_pointer_t<decltype(s->{compact})>> before_{compact}(s->{compact},s->{compact}+3*{width});\n'
    extra+='copyCellLocalParticle(*s,2,9,1);\n'
    for f in fs:
        name=f['name'];compact=f['compact'];width=f['width']
        expected='9' if name=='pCellId' else '1' if name=='pStatus' else f's->{name}[2*{width}+j]'
        extra+=f'for(size_t j=0;j<{width};++j){{if(s->{compact}[{width}+j]!={expected})throw std::runtime_error("odd destination {name}");if(s->{compact}[j]!=before_{compact}[j]||s->{compact}[2*{width}+j]!=before_{compact}[2*{width}+j])throw std::runtime_error("neighbor overwritten {name}");}}\n'
    code=code.replace('copyCellLocalParticle(*s,1,9,0);',extra+'copyCellLocalParticle(*s,1,9,0);')
    code=code.replace('std::ofstream out(argv[1],','std::ofstream out("'+str(tmp_path/'lifecycle.wire')+'",')
    compile_run(tmp_path,code,'lifecycle')


def test_contact_table_keeps_double_knots_and_interpolation(tmp_path):
    compile_run(tmp_path,r'''
#include "common/wall/GpuFiniteWallContact.H"
#include <vector>
#include <cassert>
#include <cmath>
int main(){using namespace Foam::gpuThermal;
const int count=finiteContactPeakTableSize*finiteContactDamageTableSize*finiteContactTimeTableSize;
std::vector<double> table(count);buildFiniteContactRateTable(table.data());
bool beyondFloat=false;
for(int p=0;p<finiteContactPeakTableSize;++p)for(int d=0;d<finiteContactDamageTableSize;++d)for(int t=0;t<finiteContactTimeTableSize;++t){
const double peak=double(p)/(finiteContactPeakTableSize-1),damage=double(d)/(finiteContactDamageTableSize-1),theta=double(t)/(finiteContactTimeTableSize-1);
const int i=(p*finiteContactDamageTableSize+d)*finiteContactTimeTableSize+t;
assert(table[i]==dimensionlessFiniteContactRate(theta,peak,damage));
if(table[i]!=double(float(table[i])))beyondFloat=true;
}
assert(beyondFloat);
for(double peak:{std::nextafter(1.0,0.0),std::nextafter(0.0,1.0),.4}){
const double result=interpolateFiniteContactRate(table.data(),.351,peak,.21);
assert(std::isfinite(result));
}
}
''','table')


def test_fsh_actual_cold_initialization_and_2d_microheat_keep_double(tmp_path):
    fixture=(ROOT/'tests/fixtures/cold_wall_contact_routes.cpp').read_text()
    prefix=fixture[:fixture.index('#include "GpuThermalParticleRelaxation.cuh"')]
    prefix=prefix.replace('#include <thread>','#include <thread>\n#include <cfloat>\n#include "GpuColdWall2DSolidification.H"')
    prefix=prefix.replace('struct DeviceState {','struct DeviceState {\nint coldWall2DEnabled=1;\ndouble cold2DH[64]{},cold2DAge[8]{};\ndouble* pCold2DNodeSpecificEnthalpy=cold2DH;double* pCold2DRingContactAge=cold2DAge;')
    prefix=prefix.replace('void clearColdWall2DParticleState(DeviceState&,int) {}','void clearColdWall2DParticleState(DeviceState&,int);')
    prefix+='\n#define pContactAge pTheta\n#include "GpuColdWall2DDevice.cuh"\n#undef pContactAge\n'
    helper=fixture[fixture.index('void require('):fixture.index('GpuTime& mechanicalAge(')]
    conductivity=(ROOT/'common/operators/updateLegacyGasBoundaryMirrorKernel.cuh').read_text()
    a=conductivity.index('template<class GasState>\n__device__ GPU_OPERATOR_REAL molecularGasConductivity');b=conductivity.index('\n}',a)+2
    (tmp_path/'route_conductivity.cuh').write_text('#include "gasTransport/GasStateView.H"\n'+conductivity[a:b])
    code=prefix+helper+r'''
int main(){DeviceState s;
for(double temperature:{1000.00000000001,1.e36}){
const double expected=coldWallSpecificEnthalpyJkg(temperature,s.coldWallSolidificationParameters);
require(std::isfinite(expected),"finite initial enthalpy");
initialiseColdWallParticleState(s,0,temperature);initialiseColdWall2DParticleState(s,0,temperature);
for(auto h:s.coldH)require(h==expected,"double1D initial publication",h,expected);
for(auto h:s.cold2DH)require(h==expected,"double2D initial publication",h,expected);
if(temperature>1.e30)require(expected>FLT_MAX,"FSH profile storage exceeds former float range");
}
const double temperature=1000.00000000001;
initialiseColdWallParticleState(s,0,temperature);
const double old1=s.coldH[0];double increment=0;
require(coldWallGasSpecificEnthalpyIncrement(old1,3.e-9,1100.,1.e-5,1.e-10,s.coldWallSolidificationParameters,increment),"independent1D increment");
require(increment>0&&increment<.01,"sub-float-ULP1D heat");
int errors1=launch8([&]{double t=0;require(advanceColdWall1DGasOnlyGroup(s,0,threadIdx.x,255,3.e-9,1100.,1.e-5,1.e-10,0.,t),"actual1D gas-only accepted");});
require(errors1==0,"gas-only eight lanes");for(auto h:s.coldH)require(h==old1+increment,"gas-only double publication",h,old1+increment);
initialiseColdWallParticleState(s,0,temperature);
errors1=launch8([&]{double t=0,frozen=0,wall=0;require(advanceColdWall1DThermalGroup(s,0,threadIdx.x,255,1.e-12,3.e-9,2.2e-8,0.,.01,.4,1.e-10,temperature,10000.,0.,1100.,1.e-5,t,frozen,wall),"actual1D group accepted");});
require(errors1==0,"1D group eight lanes");for(auto h:s.coldH)near(h,old1+increment,2.e-9,"1D double publication");
initialiseColdWall2DParticleState(s,0,temperature);
const double old=s.cold2DH[0],mass=3.e-9,gasG=1.e-5,dt=1.e-10;
double mean=0;
const int errors=launch8([&]{double t=0,surface=0,frozen=0,wall=0;
bool ok=advanceColdWall2DThermalGroup(s,0,threadIdx.x,255,1.e-12,mass,2.2e-8,0.,dt,temperature,10000.,0.,1100.,gasG,t,surface,frozen,wall);
require(ok,"actual2D gas-only microheat accepted");require(wall==0.,"no wall transfer");if(threadIdx.x==0)mean=t;});
require(errors==0,"eight lanes complete");
double sum=0;for(auto h:s.cold2DH)sum+=(h-old)/64.;
const double expected=gasG*(1100.-temperature)*dt/mass;
near(sum,expected,3.e-9,"mean2D gas energy retained");
require(sum>0&&sum<double(std::nextafter(float(old),INFINITY))-double(float(old)),"heat is below float ULP");
require(std::isfinite(mean),"derived double temperature");
}
'''
    source=tmp_path/'fsh.cpp';exe=tmp_path/'fsh';source.write_text(code)
    subprocess.run(['g++','-std=c++20','-O2','-pthread','-DROUTE_FSH=1','-I'+str(tmp_path),'-I'+str(ROOT/'common'),'-I'+str(ROOT/'common/wall'),str(source),'-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True,timeout=30)


@pytest.mark.parametrize('owner',['CHT','FSH'])
def test_actual_text_restart_area_assignment_keeps_small_and_large_double(tmp_path,owner):
    source=(ROOT/f'applications/{owner}/gpu/GpuResidentStrict.H').read_text()
    end=source.index('            else if (version2 && stuck != 0)')
    start=source.rfind('                pDepositionArea[i] = ',0,end)
    body=source[start:end].rstrip()
    body=body[:body.rfind('}')]


    body=body.replace('FatalErrorInFunction','diagnostic').replace('exit(FatalError)','reject()')
    compile_run(tmp_path,'''#include <sstream>
#include <limits>
#include <cmath>
#include <cassert>
using scalar=double;bool rejected=false;const char* reject(){rejected=true;return "";}
int main(){for(double depositionArea:{std::numeric_limits<double>::denorm_min(),1.e100,1.0+0x1p-30}){
int i=0;double pDepositionArea[1]={};std::ostringstream diagnostic;
BODY
assert(!rejected);assert(pDepositionArea[0]==depositionArea);}}
'''.replace('BODY',body),'text')


@pytest.mark.parametrize('owner',['CHT','FSH'])
def test_lifecycle_probe_uses_actual_native_copy_adapter(owner):
    import re
    spec=importlib.util.spec_from_file_location('contract',ROOT/'tools/particle_field_contract.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    native=(ROOT/f'applications/{owner}/{"gpu" if owner=="CHT" else "private_backend"}/GpuResidentStrict.cu').read_text()
    adapter=re.search(r'#define GPU_PARTICLE_EXTRA_FIELDS (CellLocalThermalExtraFields<[^>]+>)',native).group(1)
    meta=json.loads((ROOT/'common/ParticleFieldManifest.json').read_text())
    code=module.build_probe(ROOT,meta,owner)
    assert ('using ProbeExtra='+adapter+';').replace(' ','') in code.replace(' ','')


def test_lifecycle_rejects_stale_float_memset_for_double_ring_mass(tmp_path):
    spec=importlib.util.spec_from_file_location('contract',ROOT/'tools/particle_field_contract.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    meta=json.loads((ROOT/'common/ParticleFieldManifest.json').read_text())
    code=module.build_probe(ROOT,meta,'CHT')
    assert 'coldRingStorage*sizeof(double)' in code
    code=code.replace('coldRingStorage*sizeof(double)','coldRingStorage*sizeof(float)',1)
    code=code.replace('std::ofstream out(argv[1],','std::ofstream out("'+str(tmp_path/'unused.wire')+'",')
    source=tmp_path/'bad.cpp';exe=tmp_path/'bad';source.write_text(code)
    subprocess.run(['g++','-std=c++17','-O0','-I'+str(ROOT/'common'),str(source),'-o',str(exe)],check=True)
    result=subprocess.run([str(exe)],capture_output=True,text=True)
    assert result.returncode!=0
    assert 'memset byte closure' in result.stderr
