"""New optional model messages must never enlarge the legacy v7 payload."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]

def test_versioned_gas_model_wire_layout_and_checked_lengths(tmp_path):
    code = r'''
#include "applications/gasUGKP/gpu/GpuBackendProtocol.H"
#include "common/gasTransport/GasBuildConfig.H"
#include <cassert>
#include <limits>
#include <type_traits>
int main(){
 using namespace ugkwpGpuIpc;
 static_assert(protocolMajor==7 && protocolMinor==0,"legacy peers remain compatible");
 static_assert(sizeof(CreateArgs)==392,"do not append model pointers to create");
 static_assert(sizeof(GasModelCapabilitiesV1)==16,"no wire padding");
 static_assert(sizeof(GasModelConfigureArgsV1)==56,"no wire padding");
 static_assert(sizeof(GasSpeciesIdentityV1)==32,"no wire padding");
 static_assert(std::is_trivially_copyable<GasModelConfigureArgsV1>::value,"wire scalar POD only");
 static_assert(ugkwp::compiledGasSpecies==2,"default storage shape only");
 GasModelConfigureArgsV1 args{};
 args.version=1;args.mode=1;args.speciesCount=2;
 args.speciesOrderHash=123;args.thermoHash=456;
 args.modelBytes=100;
 std::uint64_t bytes=0;
 assert(gasModelPayloadBytes(args,2,bytes)&&bytes==156);
 args.mode=2;args.mechanismBytes=100;args.mechanismHash=789;
 assert(gasModelPayloadBytes(args,2,bytes)&&bytes==256);
 args.modelBytes=std::numeric_limits<std::uint64_t>::max();
 assert(!gasModelPayloadBytes(args,2,bytes));
 args.modelBytes=100;args.speciesCount=3;
 assert(!gasModelPayloadBytes(args,2,bytes));
 args.speciesCount=2;args.mode=1;
 assert(!gasModelPayloadBytes(args,2,bytes));
 args.mechanismBytes=0;args.mechanismHash=0;
 assert(gasModelPayloadBytes(args,2,bytes));
 GasSpeciesIdentityV1 identity{1,2,123,456,0};
 assert(sameGasSpeciesIdentity(identity,identity));
 auto wrong=identity;wrong.speciesOrderHash=124;
 assert(!sameGasSpeciesIdentity(identity,wrong));
 assert(gasSpeciesPayloadBytes(identity,20,bytes)&&bytes==32+40*sizeof(double));
 assert(!gasSpeciesPayloadBytes(identity,std::numeric_limits<std::uint64_t>::max(),bytes));
}
'''
    source=tmp_path/"probe.cpp";source.write_text(code);binary=tmp_path/"probe"
    build=subprocess.run(["g++","-std=c++14","-Wall","-Wextra","-Werror","-I",str(ROOT),str(source),"-o",str(binary)],capture_output=True,text=True)
    assert build.returncode==0,build.stderr
    run=subprocess.run([str(binary)],capture_output=True,text=True)
    assert run.returncode==0,run.stderr

def test_new_operations_are_declared_and_client_server_compile(tmp_path):
    code = r'''
#include "applications/gasUGKP/gpu/GpuBackendApi.H"
int main(){
 auto* query=&ugkwpGpuResidentStrictQueryGasModelCapabilitiesV1;
 auto* configure=&ugkwpGpuResidentStrictConfigureGasModelV1;
 auto* upload=&ugkwpGpuResidentStrictUploadSpeciesV1;
 auto* download=&ugkwpGpuResidentStrictDownloadSpeciesV1;
 auto* boundary=&ugkwpGpuResidentStrictUploadSpeciesBoundaryV1;
 (void)query;(void)configure;(void)upload;(void)download;(void)boundary;
}
'''
    probe=tmp_path/"api.cpp";probe.write_text(code)
    for source in [probe,ROOT/"applications/gasUGKP/gpu/GpuBackendClient.C",ROOT/"applications/gasUGKP/private_backend/GpuBackendServer.C"]:
        run=subprocess.run(["g++","-std=c++14","-Wall","-Wextra","-I",str(ROOT),"-I",str(ROOT/"applications/gasUGKP/gpu"),"-c",str(source),"-o",str(tmp_path/(source.stem+".o"))],capture_output=True,text=True)
        assert run.returncode==0,run.stderr

def test_build_scripts_select_the_same_explicit_species_shape():
    root=ROOT/"applications/gasUGKP"
    for relative in ("tools/build_public_frontend.sh","private_backend/build_private_backend.sh"):
        script=(root/relative).read_text()
        assert 'gas_species="${UGKWP_GAS_SPECIES-2}"' in script
        assert '-DUGKWP_GAS_SPECIES=${gas_species}' in script

def test_explicit_empty_species_build_option_is_not_defaulted():
    for relative in ("tools/build_public_frontend.sh","private_backend/build_private_backend.sh"):
        text=(ROOT/"applications/gasUGKP"/relative).read_text()
        line=next(line for line in text.splitlines() if line.startswith('gas_species='))
        empty=subprocess.run(["bash","-c",'UGKWP_GAS_SPECIES=""; '+line+'; printf "%s" "$gas_species"'],capture_output=True,text=True)
        absent=subprocess.run(["bash","-c",'unset UGKWP_GAS_SPECIES; '+line+'; printf "%s" "$gas_species"'],capture_output=True,text=True)
        assert empty.stdout==""
        assert absent.stdout=="2"

def test_species_download_rejects_truncated_socket_payload_without_partial_output(tmp_path):
    """Actual client over a local socket; the peer sends an interrupted v1 reply."""
    source=tmp_path/"download.cpp"
    source.write_text(r'''
#include "applications/gasUGKP/gpu/GpuBackendClient.C"
#include <cassert>
#include <thread>
int main(){
 using namespace ugkwpGpuIpc;
 for(int truncated=0;truncated<2;++truncated){
  int sockets[2];assert(socketpair(AF_UNIX,SOCK_STREAM,0,sockets)==0);
  ClientState state;state.fd=sockets[0];state.nCells=2;state.gasModelConfigured=true;
  state.gasSpeciesIdentity={1,2,123,456,0};
  const double expected[]={.3,.4,.7,.6};
  std::thread peer([&]{
   RequestHeader request{};GasSpeciesIdentityV1 sent{};
   assert(readAll(sockets[1],&request,sizeof(request)));
   assert(request.operation==static_cast<unsigned>(Op::downloadSpeciesV1));
   assert(readAll(sockets[1],&sent,sizeof(sent)));
   assert(sameGasSpeciesIdentity(sent,state.gasSpeciesIdentity));
   ResponseHeader reply{magic,protocolMajor,protocolMinor,0,0,sizeof(sent)+sizeof(expected)};
   assert(sendObject(sockets[1],reply));assert(sendObject(sockets[1],sent));
   assert(sendArray(sockets[1],expected,truncated?1:4));
   close(sockets[1]);
  });
  double output[]={-9,-9,-9,-9};
  const int result=ugkwpGpuResidentStrictDownloadSpeciesV1(&state,&state.gasSpeciesIdentity,output);
  peer.join();close(sockets[0]);
  assert((result!=0)==bool(truncated));
  for(int i=0;i<4;++i)assert(output[i]==(truncated?-9:expected[i]));
 }
}
''')
    binary=tmp_path/"download"
    build=subprocess.run(["g++","-std=c++14","-pthread","-Wall","-Wextra","-Werror","-I",str(ROOT),str(source),"-o",str(binary)],capture_output=True,text=True)
    assert build.returncode==0,build.stderr
    run=subprocess.run([str(binary)],capture_output=True,text=True)
    assert run.returncode==0,run.stderr
