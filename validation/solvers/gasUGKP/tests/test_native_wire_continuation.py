"""T08 exercises actual framing and restart codecs, without GPU execution."""
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from pathlib import Path
import importlib.util
import json
import subprocess
import pytest
ROOT=SOURCE_ROOT
def run_cpp(tmp_path,code,app='gasUGKP',flags=(),timeout=30):
    source=tmp_path/'probe.cpp';source.write_text(code);binary=tmp_path/'probe'
    build=subprocess.run(['g++','-std=c++17','-O0','-pthread','-Wall','-Wextra','-Werror','-ffunction-sections','-fdata-sections','-Wl,--gc-sections','-I',str(ROOT),'-I',str(ROOT/f'applications/{app}/gpu'),*flags,str(source),'-o',str(binary)],capture_output=True,text=True)
    assert build.returncode==0,build.stderr
    run=subprocess.run([str(binary)],capture_output=True,text=True,timeout=timeout)
    assert run.returncode==0,run.stdout+run.stderr
    return run.stdout
@pytest.mark.parametrize('app',['gasUGKP','FSH'])
def test_representable_payload_has_no_policy_ceiling(tmp_path,app):
    if app=='FSH':
        run_cpp(tmp_path,r'''
#include "applications/FSH/gpu/GpuBackendProtocol.H"
#include <cassert>
#include <limits>
int main(){using namespace ugkwpGpuIpc;std::uint64_t bytes=0;
addBytes(bytes,33ULL*1024*1024*1024,1);assert(bytes==33ULL*1024*1024*1024);
addBytes(bytes,1,8);assert(bytes==33ULL*1024*1024*1024+8);
bytes=std::numeric_limits<std::uint64_t>::max()-3;addBytes(bytes,1,4);assert(bytes==0);
bytes=0;addBytes(bytes,std::numeric_limits<std::uint64_t>::max(),2);assert(bytes==std::numeric_limits<std::uint64_t>::max()-1);}
''',app)
        return
    run_cpp(tmp_path,f'#include "applications/{app}/gpu/GpuBackendProtocol.H"\n'+r'''
#include <cassert>
#include <limits>
int main(){using namespace ugkwpGpuIpc;std::uint64_t bytes=0;
assert(addBytes(bytes,33ULL*1024*1024*1024,1));assert(bytes==33ULL*1024*1024*1024);
assert(addBytes(bytes,1,8));
bytes=std::numeric_limits<std::uint64_t>::max()-3;assert(!addBytes(bytes,1,4));
bytes=0;assert(!addBytes(bytes,std::numeric_limits<std::uint64_t>::max(),2));
}
''',app)
@pytest.mark.parametrize('app',['gasUGKP','FSH'])
def test_actual_client_consumes_surplus_and_metadata_with_next_frame(tmp_path,app):
    run_cpp(tmp_path,f'#include "applications/{app}/gpu/GpuBackendClient.C"\n'+r'''
#include <cassert>
#include <thread>
int main(){using namespace ugkwpGpuIpc;
int sockets[2];assert(socketpair(AF_UNIX,SOCK_STREAM,0,sockets)==0);
ClientState s;s.fd=sockets[0];s.nCells=2;
std::thread peer([&]{for(int frame=0;frame<2;++frame){RequestHeader h{};
assert(readAll(sockets[1],&h,sizeof(h)));
ResponseHeader r{magic+1,99,77,0,0,2*sizeof(double)+3};
assert(sendObject(sockets[1],r));double values[]={-1.25-frame,2.5+frame};
assert(sendArray(sockets[1],values,2));assert(writeAll(sockets[1],"abc",3));}close(sockets[1]);});
for(int frame=0;frame<2;++frame){double out[]={-999,-999};
assert(ugkwpGpuResidentStrictDownloadNut(&s,out)==0);
assert(out[0]==-1.25-frame&&out[1]==2.5+frame);}
peer.join();close(sockets[0]);
}
''',app)
def test_adapter_reconfiguration_reupload_signed_state_and_no_certificate_copy(tmp_path):
    spec=importlib.util.spec_from_file_location('adapter',(VALIDATION_ROOT / 'solvers/gasUGKP/tests/test_shared_gas_adapter.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    harness=m.ADAPTER_PROBE.split('int main(){',1)[0].replace('std::string last;','int hostCopies=0;\nstd::string last;').replace('int copyToHost(T*d,const T*s,std::size_t n,const char*){','int copyToHost(T*d,const T*s,std::size_t n,const char*){++hostCopies;').replace('std::string last;','int allocatedCalls=0,tableCopies=0;std::string last;').replace('if(!n)return 0;if(failAfter', 'if(!n)return 0;++allocatedCalls;if(failAfter').replace('int copyToDevice(T*d,const T*s,std::size_t n,const char*){','int copyToDevice(T*d,const T*s,std::size_t n,const char*){if(n)++tableCopies;')
    run_cpp(tmp_path,harness+r'''
int main(){DeviceState state,device;state.deviceState=&device;
const std::string text=R"(gasMode mixtureFrozen;species(A B);speciesThermo{
A{model linearCp;molarMass .028;minTemperature 100;maxTemperature 4000;coefficients(1040 0 0);}
B{model linearCp;molarMass .032;minTemperature 100;maxTemperature 4000;coefficients(1200 0 0);}}
diffusion{model constant;coefficients(.02 .03);turbulentSchmidt .7;})";
ugkwpGpuIpc::GasModelConfigureArgsV1 args{99,77,42,33,0,0,0,text.size(),0};
state.gasInitialFieldsUploaded=true;
assert(ugkwpGpuResidentStrictConfigureGasModelV1(&state,&args,text.data(),nullptr)==0);
const int count=allocations;
ugkwpGpuIpc::GasSpeciesIdentityV1 identity{99,42,0,0,0};
double values[]={-.1,.4,1.1,.6};
assert(ugkwpGpuResidentStrictUploadSpeciesV1(&state,&identity,values)==0);
assert(hostCopies==0);
values[1]=-.75;assert(ugkwpGpuResidentStrictUploadSpeciesV1(&state,&identity,values)==0);
assert(allocations==count&&device.gasSpecies.rho==state.gasSpecies.rho);
double output[]={-999,-999,-999,-999};
assert(ugkwpGpuResidentStrictDownloadSpeciesV1(&state,&identity,output)==0);
for(int i=0;i<4;++i)assert(output[i]==values[i]);
int fixed[]={2,-1};double boundary[]={-.3,.4,1.3,.6};
assert(ugkwpGpuResidentStrictUploadSpeciesBoundaryV1(&state,&identity,fixed,boundary)==0);
for(int i=0;i<2;++i)assert(device.gasSpecies.compositionBoundaryFixed[i]==fixed[i]);
for(int i=0;i<4;++i)assert(device.gasSpecies.boundaryMassFraction[i]==boundary[i]);
// Configuration replacement keeps pending state; failed allocation and sync keep old views live.
auto* accepted=state.gasSpecies.rho;failAfter=1;
assert(ugkwpGpuResidentStrictConfigureGasModelV1(&state,&args,text.data(),nullptr)!=0);
assert(allocations==count&&state.gasSpecies.rho==accepted&&device.gasSpecies.rho==accepted);failAfter=-1;
syncFailAfter=0;
assert(ugkwpGpuResidentStrictConfigureGasModelV1(&state,&args,text.data(),nullptr)!=0);
assert(allocations==count&&state.gasSpecies.rho==accepted&&device.gasSpecies.rho==accepted);
const int callsBefore=allocatedCalls,copiesBefore=tableCopies;
assert(ugkwpGpuResidentStrictConfigureGasModelV1(&state,&args,text.data(),nullptr)==0);
assert(allocatedCalls-callsBefore==3&&tableCopies-copiesBefore==3);
assert(allocations==count&&device.gasSpecies.rho==state.gasSpecies.rho&&state.gasSpeciesUploaded);
for(int i=0;i<4;++i){assert(device.gasSpecies.rho[i]==values[i]);assert(device.gasSpecies.boundaryMassFraction[i]==boundary[i]);}
releaseSharedGasSpecies(state.gasSpecies);assert(allocations==0);
}
''')
@pytest.mark.parametrize('app',['CHT','FSH'])
def test_restart_full_fields_surplus_bound_and_nonfinite_are_data(tmp_path,app):
    meta=json.loads((ROOT/'common/ParticleFieldManifest.json').read_text())
    fields=[f for f in meta['fields'] if app in f['host'] and f['schema7'] is not None]
    typ=lambda f:{'unsigned long long':'std::uint64_t','unsigned char':'std::uint8_t','int':'std::int32_t'}.get(f['host'][app],f['host'][app])
    declarations='\n'.join(f'std::vector<{typ(f)}> in_{f["name"]}(3*{f["width"]}),out_{f["name"]}(3*{f["width"]},static_cast<{typ(f)}>(9));for(std::size_t i=0;i<in_{f["name"]}.size();++i)in_{f["name"]}[i]=static_cast<{typ(f)}>({j+1}+i*.125);' for j,f in enumerate(fields))
    assignments='\n'.join(f'in.{f["name"]}=in_{f["name"]}.data();out.{f["name"]}=out_{f["name"]}.data();' for f in fields)
    checks='\n'.join(f'assert(in_{f["name"]}==out_{f["name"]});' for f in fields if f['name']!='pT')
    header='GpuChtParticleRestart.H' if app=='CHT' else 'GpuFshParticleRestart.H'
    extra='double carry[]={.00030000000000000003,-.0002,.0005},restored[]={9,9,9};in.pThermalEnthalpyResidual=carry;out.pThermalEnthalpyResidual=restored;' if app=='CHT' else ''
    version='8' if app=='CHT' else '7'
    run_cpp(tmp_path,f'#include "applications/{app}/gpu/{header}"\n#include "validation/tests/fixtures/shared_operators/legacy_restart_writer.hpp"\n'+r'''
#include <cassert>
#include <sstream>
#include <vector>
int main(){using namespace Foam::gpuFshRestart;
DECLARATIONS
ConstView in{};MutableView out{};in.count=out.count=3;
ASSIGNMENTS
EXTRA
in_pT[0]=std::numeric_limits<double>::infinity();
std::ostringstream wire(std::ios::binary);writeVX(wire,in);
std::string bytes=wire.str()+"metadata";
std::istringstream is(bytes,std::ios::binary);std::string marker;std::size_t n;std::uint32_t bound;is>>marker>>n>>bound;
readVXPayload(is,0,out);CHECKS
assert(std::isinf(out_pT[0]));assert(in_pT[1]==out_pT[1]&&in_pT[2]==out_pT[2]);
CARRYCHECK
std::string truncated=wire.str();truncated.pop_back();std::istringstream bad(truncated,std::ios::binary);bad>>marker>>n>>bound;
bool failed=false;try{readVXPayload(bad,99999999,out);}catch(const std::exception&){failed=true;}assert(failed);
}
'''.replace('DECLARATIONS',declarations).replace('ASSIGNMENTS',assignments).replace('EXTRA',extra).replace('VX','Current').replace('CHECKS',checks).replace('CARRYCHECK','for(int i=0;i<3;++i)assert(restored[i]==carry[i]);' if app=='CHT' else ''),app)

@pytest.mark.parametrize('app',['gasUGKP','FSH'])
def test_actual_server_decodes_extra_bytes_then_next_frame_and_short_field(tmp_path,app):


    run_cpp(tmp_path,f'#define main backend_main\n#include "applications/{app}/private_backend/GpuBackendServer.C"\n#undef main\n'+r'''
#include <cassert>
#include <sys/socket.h>
static double gotDt=0,gotTime=0;
extern "C" int ugkwpGpuResidentStrictAdvanceGasOnly(void*,double dt,double time){gotDt=dt;gotTime=time;return 0;}
extern "C" const char* ugkwpGpuResidentStrictLastError(){return "spy";}
int main(){using namespace ugkwpGpuIpc;int sockets[2];assert(socketpair(AF_UNIX,SOCK_STREAM,0,sockets)==0);
ServerState s;s.backend=&s;s.nCells=2;s.nFaces=2;
for(int frame=0;frame<2;++frame){RequestHeader req{magic+1,88,99,unsigned(Op::advanceGasOnly),42,wireBytes(AdvanceArgs{})+3};
const auto head=encodeWire(req),dummy=head;(void)dummy;const auto body=encodeWire(AdvanceArgs{-.25-frame,4.5+frame});
assert(writeAll(sockets[1],head.data(),head.size()));assert(writeAll(sockets[1],body.data(),body.size()));assert(writeAll(sockets[1],"abc",3));
RequestHeader incoming{};assert(readObject(sockets[0],incoming));readingFrame=true;frameRemaining=incoming.payloadBytes;
assert(handleGasOnly(sockets[0],incoming,s));assert(gotDt==-.25-frame&&gotTime==4.5+frame);
assert(discardRequest(sockets[0]));assert(frameRemaining==0);
ResponseHeader reply{};assert(readObject(sockets[1],reply));assert(reply.status==0);}
// Actual scalar missing from body must not read bytes from a future frame.
RequestHeader shortRequest{magic,protocolMajor,protocolMinor,unsigned(Op::advanceGasOnly),0,7};
assert(writeAll(sockets[1],"short!!",7));readingFrame=true;frameRemaining=7;assert(!handleGasOnly(sockets[0],shortRequest,s));readingFrame=false;
close(sockets[0]);close(sockets[1]);
}
''',app)

@pytest.mark.parametrize('app',['CHT','FSH'])
def test_schema7_actual_time_width_conversion_keeps_all_other_fields(tmp_path,app):
    meta=json.loads((ROOT/'common/ParticleFieldManifest.json').read_text())
    fields=[f for f in meta['fields'] if app in f['host'] and f['schema7'] is not None]
    typ=lambda f:{'unsigned long long':'std::uint64_t','unsigned char':'std::uint8_t','int':'std::int32_t'}.get(f['host'][app],f['host'][app])
    disk='float' if app=='CHT' else 'double'
    declarations='\n'.join(f'std::vector<{typ(f)}> src_{f["name"]}(3*{f["width"]}),dst_{f["name"]}(3*{f["width"]},static_cast<{typ(f)}>(99));for(std::size_t i=0;i<src_{f["name"]}.size();++i)src_{f["name"]}[i]=static_cast<{typ(f)}>({j+1}+.00001+i*.010001);const auto saved_{f["name"]}=src_{f["name"]};in.{f["name"]}=src_{f["name"]}.data();out.{f["name"]}=dst_{f["name"]}.data();' for j,f in enumerate(fields))
    checks='\n'.join(f'assert(src_{f["name"]}==saved_{f["name"]});for(std::size_t i=0;i<dst_{f["name"]}.size();++i)assert(dst_{f["name"]}[i]=='+(f'static_cast<{typ(f)}>(static_cast<{disk}>(src_{f["name"]}[i]))' if f['name'] in ['pContactDuration','pColdContactAge','pCold2DRingContactAge'] else f'static_cast<{typ(f)}>(static_cast<float>(src_{f["name"]}[i]))' if f['schema7'] in range(16,27) else f'src_{f["name"]}[i]')+');' for f in fields)
    header='GpuChtParticleRestart.H' if app=='CHT' else 'GpuFshParticleRestart.H'
    run_cpp(tmp_path,f'#include "applications/{app}/gpu/{header}"\n#include "validation/tests/fixtures/shared_operators/legacy_restart_writer.hpp"\n'+r'''
#include <cassert>
#include <sstream>
#include <vector>
int main(){using namespace Foam::gpuFshRestart;
ConstView in{};MutableView out{};in.count=out.count=3;DECLARATIONS
std::ostringstream wire(std::ios::binary);Foam::gpuThermalRestart::write<DISK>(wire,in);
std::istringstream input(wire.str(),std::ios::binary);std::string marker;std::size_t n;std::uint32_t bound;input>>marker>>n>>bound;
Foam::gpuThermalRestart::readPayload<DISK>(input,0,out);CHECKS
// The actual narrowing consumer still reports finite-to-infinite overflow.
double huge=std::numeric_limits<double>::max();std::ostringstream bad(std::ios::binary);bool failed=false;
try{Foam::gpuThermalRestart::writeConverted<float>(bad,&huge,0,1);}catch(const std::runtime_error&){failed=true;}assert(failed);
float sentinel=99;std::ostringstream raw(std::ios::binary);Foam::gpuThermalRestart::writeArray(raw,&huge,0,1);
std::istringstream overflow(raw.str(),std::ios::binary);failed=false;
try{Foam::gpuThermalRestart::readConverted<double>(overflow,&sentinel,0,1);}catch(const std::runtime_error&){failed=true;}assert(failed&&sentinel==99);
}
'''.replace('DECLARATIONS',declarations).replace('CHECKS',checks).replace('DISK',disk),app)

def test_fsh_actual_stuck_configuration_forwards_signed_values_and_supplied_extent(tmp_path):
    run_cpp(tmp_path,r'''
#include "applications/FSH/gpu/GpuBackendClient.C"
#include <cassert>
#include <thread>
int main(){using namespace ugkwpGpuIpc;int sockets[2];assert(socketpair(AF_UNIX,SOCK_STREAM,0,sockets)==0);ClientState s;s.fd=sockets[0];s.nFaces=2;
std::thread peer([&]{RequestHeader header{};assert(receiveObject(sockets[1],header));ParticleStuckConfigArgs a{};assert(receiveObject(sockets[1],a));
assert(a.nFaces==3&&a.heatTransferEnabled==2&&a.sommerfeldThreshold==-.2&&a.nonlinearIterations==15&&a.maximumCoverage==2);
unsigned char mask[3];assert(receiveArray(sockets[1],mask,3));assert(mask[0]==0&&mask[1]==1&&mask[2]==2);
assert(sendObject(sockets[1],ResponseHeader{magic,protocolMajor,protocolMinor,0,0,0}));close(sockets[1]);});
unsigned char mask[]={0,1,2};assert(ugkwpGpuResidentStrictConfigureParticleStuckModel(&s,3,mask,-.2,2,2,-.3,1.5,-.4,2,3,4,181,2,15,-500,-1,-2,3,4,5,1.1,-.01)==0);
peer.join();close(sockets[0]);}
''','FSH')

@pytest.mark.parametrize('app',['gasUGKP','FSH'])
@pytest.mark.parametrize('packed',[False,True])
def test_fixed_field_codec_matches_ordinary_bytes_and_packed_host_layout(tmp_path,app,packed):
    prefix=r'''
#include <array>
#include <algorithm>
#include <cstring>
#include <type_traits>
#include <limits>
#include <cstdint>
#include <cstddef>
'''
    prefix+=('#pragma pack(push,1)\n' if packed else '')+f'#include "applications/{app}/gpu/GpuBackendProtocol.H"\n'+('#pragma pack(pop)\n' if packed else '')
    run_cpp(tmp_path,prefix+r'''
#include <cassert>
int main(){using namespace ugkwpGpuIpc;CreateArgs a{};a.nCells=17;a.nFaces=29;a.rngSeed=0xfedcba9876543210ULL;a.gammaGas=-1.25;a.Rgas=287.25;a.gravityZ=-9.5;
a.particleBlockThreads=37;
const auto bytes=encodeWire(a);CreateArgs restored{};WireCodec<CreateArgs>::decode(bytes.data(),restored);
assert(restored.nCells==17&&restored.nFaces==29&&restored.rngSeed==a.rngSeed&&restored.gammaGas==-1.25&&restored.Rgas==287.25&&restored.gravityZ==-9.5&&restored.particleBlockThreads==37);
assert(bytes[0]==17&&bytes[4]==29&&bytes[32]==0x10&&bytes[39]==0xfe);
if constexpr (WireCodec<CreateArgs>::nativeLayout) assert(std::memcmp(bytes.data(),&a,bytes.size())==0);
else assert(sizeof(a)!=bytes.size());
}
''',app)

@pytest.mark.parametrize('app',['gasUGKP','FSH'])
def test_actual_client_full_field_output_and_error_bytes_without_cap(tmp_path,app):
    run_cpp(tmp_path,f'#include "applications/{app}/gpu/GpuBackendClient.C"\n'+r'''
#include <cassert>
#include <thread>
int main(){using namespace ugkwpGpuIpc;int sockets[2];assert(socketpair(AF_UNIX,SOCK_STREAM,0,sockets)==0);ClientState s;s.fd=sockets[0];s.nCells=3;
std::thread peer([&]{RequestHeader h{};assert(receiveObject(sockets[1],h));assert(h.operation==unsigned(Op::downloadFields));
assert(sendObject(sockets[1],ResponseHeader{magic+1,88,99,0,0,23*3*sizeof(double)+2}));
for(int f=0;f<23;++f){double v[]={f+.125,f+.25,f+.5};assert(sendArray(sockets[1],v,3));}assert(writeAll(sockets[1],"xx",2));
assert(receiveObject(sockets[1],h));AdvanceArgs request{};assert(receiveObject(sockets[1],request));assert(request.dt==.1&&request.simulationTime==1);std::string error(1024*1024+7,'e');
assert(sendObject(sockets[1],ResponseHeader{magic+1,88,99,7,unsigned(error.size()),3}));assert(writeAll(sockets[1],error.data(),error.size()));assert(writeAll(sockets[1],"abc",3));
assert(receiveObject(sockets[1],h));assert(receiveObject(sockets[1],request));assert(request.dt==.1&&request.simulationTime==1.1);assert(sendObject(sockets[1],ResponseHeader{magic,protocolMajor,protocolMinor,0,0,0}));close(sockets[1]);});
double out[23][3];for(auto&f:out)for(double&x:f)x=-999;
assert(ugkwpGpuResidentStrictDownloadFields(&s,out[0],out[1],out[2],out[3],out[4],out[5],out[6],out[7],out[8],out[9],out[10],out[11],out[12],out[13],out[14],out[15],out[16],out[17],out[18],out[19],out[20],out[21],out[22])==0);
for(int f=0;f<23;++f){assert(out[f][0]==f+.125&&out[f][1]==f+.25&&out[f][2]==f+.5);}assert(!s.responseActive&&s.responseRemaining==0);
assert(ugkwpGpuResidentStrictAdvanceGasOnly(&s,.1,1)==7);assert(lastError.size()==1024*1024+7);assert(!s.responseActive&&s.responseRemaining==0);
assert(ugkwpGpuResidentStrictAdvanceGasOnly(&s,.1,1.1)==0);peer.join();close(sockets[0]);
}
''',app)

@pytest.mark.parametrize('app',['gasUGKP','FSH'])
def test_actual_client_required_bytes_and_disconnect_are_errors(tmp_path,app):
    run_cpp(tmp_path,f'#include "applications/{app}/gpu/GpuBackendClient.C"\n'+r'''
#include <cassert>
#include <thread>
int main(){using namespace ugkwpGpuIpc;for(int kind=0;kind<2;++kind){int sockets[2];assert(socketpair(AF_UNIX,SOCK_STREAM,0,sockets)==0);ClientState s;s.fd=sockets[0];s.nCells=2;
std::thread peer([&]{RequestHeader h{};assert(receiveObject(sockets[1],h));assert(sendObject(sockets[1],ResponseHeader{magic,protocolMajor,protocolMinor,0,0,kind?16u:7u}));
if(kind){double value=.25;assert(sendArray(sockets[1],&value,1));}close(sockets[1]);});
double out[]={-999,-999};assert(ugkwpGpuResidentStrictDownloadNut(&s,out)!=0);peer.join();assert(lastError.find(kind?"needed 16 received 8":"needed 16 available 7")!=std::string::npos);s.responseActive=false;close(sockets[0]);}
}
''',app)

def test_actual_boundary_zero_wall_clear_needs_no_unconsumed_arrays(tmp_path):
    run_cpp(tmp_path,r'''
#include "applications/gasUGKP/gpu/GpuBackendClient.C"
#include <cassert>
#include <thread>
int main(){using namespace ugkwpGpuIpc;int sockets[2];assert(socketpair(AF_UNIX,SOCK_STREAM,0,sockets)==0);ClientState s;s.fd=sockets[0];
std::thread peer([&]{RequestHeader h{};assert(receiveObject(sockets[1],h));assert(h.payloadBytes==WireCodec<BoundaryLayerConfigV1>::size);
BoundaryLayerConfigV1 a{};assert(receiveObject(sockets[1],a));assert(a.wallCount==0&&a.version==99);
assert(sendObject(sockets[1],ResponseHeader{magic,protocolMajor,protocolMinor,0,0,0}));close(sockets[1]);});
BoundaryLayerConfigV1 a{};a.version=99;a.wallCount=0;a.speciesCount=42;
assert(ugkwpGpuResidentStrictConfigureBoundaryLayerV1(&s,&a,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr)==0);
peer.join();close(sockets[0]);}
''')

@pytest.mark.parametrize('app',['CHT','FSH'])
def test_zero_restart_and_actual_zero_chunk_are_distinct(tmp_path,app):
    header='GpuChtParticleRestart.H' if app=='CHT' else 'GpuFshParticleRestart.H'
    run_cpp(tmp_path,f'#include "applications/{app}/gpu/{header}"\n#include "validation/tests/fixtures/shared_operators/legacy_restart_writer.hpp"\n'+r'''
#include <cassert>
#include <sstream>
int main(){using namespace Foam::gpuFshRestart;ConstView in{};MutableView out{};
std::ostringstream wire(std::ios::binary);Foam::gpuThermalRestart::write<double>(wire,in);std::istringstream input(wire.str()+"metadata",std::ios::binary);
std::string marker;std::size_t count;std::uint32_t bound;input>>marker>>count>>bound;readV7Payload(input,0,out);assert(count==0);
out.count=1;std::string broken="\n"+std::string(4,'\0');std::istringstream bad(broken,std::ios::binary);
bool failed=false;try{readV7Payload(bad,0,out);}catch(const std::exception&){failed=true;}assert(failed);
}
''',app)

@pytest.mark.parametrize('app',['gasUGKP','FSH'])
def test_server_large_surplus_progress_full_output_and_second_frame(tmp_path,app):
    api=(ROOT/f'applications/{app}/gpu/GpuBackendApi.H').read_text()
    import re
    args=re.search(r'extern "C" int ugkwpGpuResidentStrictDownloadFields\s*\((.*?)\);',api,re.S).group(1)
    pointers=re.findall(r'double\*\s*(\w+)',args)
    spy='extern "C" int ugkwpGpuResidentStrictDownloadFields('+args+'){(void)handle;'+''.join(f'for(int i=0;i<4096;++i){v}[i]={j}+.125+i*.01;' for j,v in enumerate(pointers))+'return 0;}\n'
    run_cpp(tmp_path,f'#define main backend_main\n#include "applications/{app}/private_backend/GpuBackendServer.C"\n#undef main\n'+spy+r'''
#include <cassert>
#include <sys/socket.h>
#include <thread>
extern "C" const char* ugkwpGpuResidentStrictLastError(){return "spy";}
int main(){using namespace ugkwpGpuIpc;int sockets[2];assert(socketpair(AF_UNIX,SOCK_STREAM,0,sockets)==0);int size=4096;
setsockopt(sockets[0],SOL_SOCKET,SO_SNDBUF,&size,sizeof size);setsockopt(sockets[1],SOL_SOCKET,SO_SNDBUF,&size,sizeof size);
std::thread server([&]{ServerState state;state.backend=&state;state.nCells=4096;for(int frame=0;frame<2;++frame){RequestHeader request{};assert(readObject(sockets[0],request));readingFrame=true;frameBytes=frameRemaining=request.payloadBytes;
assert(handleDownloadFields(sockets[0],request,state));assert(discardRequest(sockets[0]));assert(frameRemaining==0);}});
for(int frame=0;frame<2;++frame){const auto head=encodeWire(RequestHeader{magic+1,99,77,unsigned(Op::downloadFields),0,65536});assert(writeAll(sockets[1],head.data(),head.size()));
std::string surplus(65536,'x');assert(writeAll(sockets[1],surplus.data(),surplus.size()));
ResponseHeader response{};assert(readObject(sockets[1],response));assert(response.status==0&&response.payloadBytes==23*4096*sizeof(double));
std::vector<double> payload(23*4096,99);assert(readAll(sockets[1],payload.data(),payload.size()*sizeof(double)));
for(int j=0;j<23;++j)for(int i=0;i<4096;++i)assert(payload[j*4096+i]==j+.125+i*.01);}
server.join();close(sockets[0]);close(sockets[1]);}
''',app,timeout=5)

def test_configuration_has_no_dead_capability_query(tmp_path):
    run_cpp(tmp_path,r'''
#define main backend_main
#include "applications/gasUGKP/private_backend/GpuBackendServer.C"
#undef main
#include <cassert>
#include <sys/socket.h>
static int configured=0,queried=0;
extern "C" const char* ugkwpGpuResidentStrictLastError(){return "query unavailable";}
extern "C" int ugkwpGpuResidentStrictQueryGasModelCapabilitiesV1(void*,ugkwpGpuIpc::GasModelCapabilitiesV1*){++queried;return 7;}
extern "C" int ugkwpGpuResidentStrictConfigureGasModelV1(void*,const ugkwpGpuIpc::GasModelConfigureArgsV1* a,const char* text,const char*){assert(a->modelBytes==5&&std::memcmp(text,"model",5)==0);++configured;return 0;}
int main(){using namespace ugkwpGpuIpc;int sockets[2];assert(socketpair(AF_UNIX,SOCK_STREAM,0,sockets)==0);ServerState s;s.backend=&s;
GasModelConfigureArgsV1 a{99,1,42,33,0,0,0,5,0};auto bytes=encodeWire(a);assert(writeAll(sockets[1],bytes.data(),bytes.size()));assert(writeAll(sockets[1],"model",5));
RequestHeader request{magic,protocolMajor,protocolMinor,unsigned(Op::configureGasModelV1),0,bytes.size()+5};readingFrame=true;frameBytes=frameRemaining=request.payloadBytes;
assert(handleGasModelConfiguration(sockets[0],request,s));assert(configured==1&&queried==0&&frameRemaining==0);
ResponseHeader response{};assert(readObject(sockets[1],response));assert(response.status==0);close(sockets[0]);close(sockets[1]);}
''')

def test_fsh_profile_resizing_executes_wide_actual_products(tmp_path):
    source=(ROOT/'applications/FSH/private_backend/GpuBackendServer.C').read_text()
    start=source.index('    coldNodeSpecificEnthalpy.resize(');end=source.index('    return writeVector',start)


    run_cpp(tmp_path,r'''
#include <cassert>
#include <climits>
#include <cstddef>
struct Extent {std::size_t count=0;void resize(std::size_t n){count=n;}};
int main(){const int count=INT_MAX/64+1;Extent coldNodeSpecificEnthalpy,coldRingSolidMass,coldFrozenArea,coldContactAge,cold2DNodeSpecificEnthalpy,cold2DRingContactAge,cold2DFrozenArea,rng,origId;
'''+source[start:end]+r'''
assert(coldNodeSpecificEnthalpy.count==std::size_t(count)*8);
assert(coldRingSolidMass.count==std::size_t(count)*8);
assert(cold2DNodeSpecificEnthalpy.count==std::size_t(count)*64);
assert(cold2DRingContactAge.count==std::size_t(count)*8);}
''','FSH',('-fsanitize=undefined','-fno-sanitize-recover=undefined'))

def test_fsh_client_profile_arguments_execute_wide_actual_products(tmp_path):
    import re
    source=(ROOT/'applications/FSH/gpu/GpuBackendClient.C').read_text()
    expressions=re.findall(r'(?:sendArray|receiveArray)\(s->fd, pCold(?:NodeSpecificEnthalpy|RingSolidMass|2DNodeSpecificEnthalpy|2DRingContactAge), ([^\n]+)\)',source)
    expressions=[e[:-2] if e.endswith(',s') else e for e in expressions]
    assert len(expressions)==8
    checks=''.join(f'assert(({e})==std::uint64_t(count)*{64 if e.endswith("*64") else 8});' for e in expressions)
    run_cpp(tmp_path,'#include <cassert>\n#include <climits>\n#include <cstdint>\nint main(){const int count=INT_MAX/64+1;'+checks+'}\n','FSH',('-fsanitize=undefined','-fno-sanitize-recover=undefined'))

def test_boundary_diagnostic_required_extent_arithmetic(tmp_path):
    run_cpp(tmp_path,r'''
#include "applications/gasUGKP/gpu/GpuBackendProtocol.H"
#include <cassert>
int main(){using namespace ugkwpGpuIpc;std::uint64_t bytes=99;
assert(boundaryLayerDiagnosticBytes(0,0xffffffffu,bytes)&&bytes==0);
assert(boundaryLayerDiagnosticBytes(3,2,bytes)&&bytes==3*(25+4)*sizeof(double));
assert(boundaryLayerDiagnosticBytes(1,0xffffffffu,bytes)&&bytes==(25+2ULL*0xffffffffu)*sizeof(double));
assert(!boundaryLayerDiagnosticBytes(0xffffffffu,0xffffffffu,bytes));}
''')

@pytest.mark.parametrize('app',['gasUGKP','FSH'])
def test_ordinary_empty_response_has_no_bookkeeping_allocation(tmp_path,app):
    run_cpp(tmp_path,r'''
#include <cstdlib>
#include <new>
static bool countNew=false;static int allocations=0;
void* operator new(std::size_t n){if(countNew)++allocations;if(void*p=std::malloc(n))return p;throw std::bad_alloc();}
void operator delete(void*p)noexcept{std::free(p);}void operator delete(void*p,std::size_t)noexcept{std::free(p);}
'''+f'#include "applications/{app}/gpu/GpuBackendClient.C"\n'+r'''
#include <cassert>
int main(){using namespace ugkwpGpuIpc;int sockets[2];assert(socketpair(AF_UNIX,SOCK_STREAM,0,sockets)==0);ClientState state;state.fd=sockets[0];
lastError="no GPU backend client error";
for(int frame=0;frame<2;++frame){assert(sendObject(sockets[1],ResponseHeader{magic,protocolMajor,protocolMinor,0,0,0}));
countNew=true;int result=ugkwpGpuResidentStrictAdvanceGasOnly(&state,.1,2);countNew=false;assert(result==0&&allocations==0);}
close(sockets[0]);close(sockets[1]);}
''',app)

def test_generic_restart_raw_scalar_has_no_ieee_certificate_gate(tmp_path):
    run_cpp(tmp_path,r'''
#include "common/GpuThermalParticleRestartCodec.H"
#include <cassert>
#include <cstring>
#include <sstream>
int main(){long double source[]={1.25L,-2.5L},output[2];std::memset(output,0,sizeof(output));
std::ostringstream wire(std::ios::binary);Foam::gpuThermalRestart::writeArray(wire,source,0,2);
assert(wire.str().size()==sizeof(source));std::istringstream input(wire.str(),std::ios::binary);Foam::gpuThermalRestart::readArray(input,output,0,2);
assert(std::memcmp(source,output,sizeof(source))==0);}
''')


def test_boundary_diagnostic_actual_server_required_extent_and_full_output(tmp_path):
    run_cpp(tmp_path,r"""
#define main backend_main
#include "applications/gasUGKP/private_backend/GpuBackendServer.C"
#undef main
#include <cassert>
#include <sys/socket.h>
static int called=0;
extern "C" const char* ugkwpGpuResidentStrictLastError(){return "spy";}
extern "C" int ugkwpGpuResidentStrictDownloadBoundaryLayerV1(void*,std::uint32_t walls,std::uint32_t species,double*p){++called;assert(walls==3&&species==2);for(int i=0;i<87;++i)p[i]=i+.125;return 0;}
int main(){using namespace ugkwpGpuIpc;int sockets[2];assert(socketpair(AF_UNIX,SOCK_STREAM,0,sockets)==0);ServerState s;s.backend=&s;
for(int frame=0;frame<2;++frame){std::uint32_t dimensions[]={frame?0xffffffffu:3u,frame?0xffffffffu:2u};assert(writeVector(sockets[1],std::vector<std::uint32_t>(dimensions,dimensions+2)));
RequestHeader request{magic,protocolMajor,protocolMinor,unsigned(Op::downloadBoundaryLayerV1),0,8};readingFrame=true;frameBytes=frameRemaining=8;
bool result=handleDownloadBoundaryLayer(sockets[0],request,s);assert(result==!frame&&frameRemaining==0);
ResponseHeader reply{};assert(readObject(sockets[1],reply));if(!frame){assert(reply.status==0&&reply.payloadBytes==87*sizeof(double));std::vector<double> values;assert(readVector(sockets[1],values,87));for(int i=0;i<87;++i)assert(values[i]==i+.125);}else{assert(reply.status!=0&&reply.payloadBytes==0);std::vector<char> text;assert(readVector(sockets[1],text,reply.errorBytes));assert(std::string(text.begin(),text.end()).find("byte extent overflow")!=std::string::npos);}}
assert(called==1);close(sockets[0]);close(sockets[1]);}
""")

def test_boundary_diagnostic_actual_client_required_extent_and_full_output(tmp_path):
    run_cpp(tmp_path,r"""
#include "applications/gasUGKP/gpu/GpuBackendClient.C"
#include <cassert>
#include <thread>
int main(){using namespace ugkwpGpuIpc;int sockets[2];assert(socketpair(AF_UNIX,SOCK_STREAM,0,sockets)==0);ClientState s;s.fd=sockets[0];
std::thread peer([&]{RequestHeader request{};assert(receiveObject(sockets[1],request));std::uint32_t dimensions[2];assert(receiveArray(sockets[1],dimensions,2));assert(dimensions[0]==3&&dimensions[1]==2);
assert(sendObject(sockets[1],ResponseHeader{magic,protocolMajor,protocolMinor,0,0,87*sizeof(double)}));double values[87];for(int i=0;i<87;++i)values[i]=i+.125;assert(sendArray(sockets[1],values,87));});
double output[87];std::fill(output,output+87,99);assert(ugkwpGpuResidentStrictDownloadBoundaryLayerV1(&s,3,2,output)==0);peer.join();for(int i=0;i<87;++i)assert(output[i]==i+.125);
assert(ugkwpGpuResidentStrictDownloadBoundaryLayerV1(&s,0xffffffffu,0xffffffffu,output)!=0);assert(lastError.find("byte extent overflow")!=std::string::npos);
char byte;assert(recv(sockets[1],&byte,1,MSG_DONTWAIT)<0&&(errno==EAGAIN||errno==EWOULDBLOCK));close(sockets[0]);close(sockets[1]);}
""")


def test_boundary_zero_diagnostics_consumes_no_output_buffer(tmp_path):
    run_cpp(tmp_path,r"""
#include "applications/gasUGKP/gpu/GpuBackendClient.C"
#include <cassert>
int main(){using namespace ugkwpGpuIpc;int sockets[2];assert(socketpair(AF_UNIX,SOCK_STREAM,0,sockets)==0);ClientState s;s.fd=sockets[0];
assert(sendObject(sockets[1],ResponseHeader{magic,protocolMajor,protocolMinor,0,0,0}));assert(ugkwpGpuResidentStrictDownloadBoundaryLayerV1(&s,0,0xffffffffu,nullptr)==0);
RequestHeader request{};assert(receiveObject(sockets[1],request));std::uint32_t dims[2];assert(receiveArray(sockets[1],dims,2));assert(dims[0]==0&&dims[1]==0xffffffffu&&!s.responseActive);
close(sockets[0]);close(sockets[1]);}
""")
