from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[3]

def test_boundary_layer_ipc_payload_preserves_layout_without_policy_bounds(tmp_path):
    source=tmp_path/'probe.cpp';exe=tmp_path/'probe'
    source.write_text(r'''
#include "applications/gasUGKP/gpu/GpuBackendProtocol.H"
int main(){using namespace ugkwpGpuIpc;BoundaryLayerConfigV1 a{};a.version=1;a.speciesCount=2;a.model=1;a.nodes=24;a.maxIterations=60;a.wallCount=2;a.quadratureCount=6;a.matchingCount=3;a.workspaceSlots=2;
std::uint64_t bytes=0;if(!boundaryLayerPayloadBytes(a,4,2,bytes))return 1;
const auto expected=sizeof(a)+(2+2*3+3)*sizeof(int)+(2*7+2*6+3+2*2)*sizeof(double);
if(bytes!=expected||sizeof(SstConfigArgs)!=152||sizeof(CreateArgs)!=392)return 2;
a.wallCount=5;if(!boundaryLayerPayloadBytes(a,4,2,bytes))return 3;a.wallCount=2;
a.nodes=128;if(!boundaryLayerPayloadBytes(a,4,2,bytes))return 4;a.nodes=129;if(!boundaryLayerPayloadBytes(a,4,2,bytes))return 7;a.nodes=24;
a.speciesCount=3;if(!boundaryLayerPayloadBytes(a,4,2,bytes))return 5;a.speciesCount=2;
a.workspaceSlots=0;if(!boundaryLayerPayloadBytes(a,4,2,bytes))return 6;
a.workspaceSlots=4097;if(!boundaryLayerPayloadBytes(a,4,2,bytes))return 8;a.workspaceSlots=0;
a.budgetAudit=1;if(!boundaryLayerPayloadBytes(a,4,2,bytes))return 9;
a.budgetAudit=2;if(!boundaryLayerPayloadBytes(a,4,2,bytes))return 10;
if(sizeof(a)!=80||boundaryLayerDiagnosticScalars!=25)return 11;
a.budgetAudit=0;a.quadratureCount=0;if(!boundaryLayerPayloadBytes(a,4,2,bytes)||bytes!=expected-12*sizeof(double))return 12;
a.quadratureCount=1;if(!boundaryLayerPayloadBytes(a,4,2,bytes))return 13;}
''')
    result=subprocess.run(['g++','-std=c++17','-I'+str(ROOT),str(source),'-o',str(exe)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    result=subprocess.run([str(exe)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_zero_length_quadrature_reaches_actual_client_socket(tmp_path):
    source=tmp_path/'socket.cpp';binary=tmp_path/'socket'
    source.write_text(r'''
#include "applications/gasUGKP/gpu/GpuBackendClient.C"
#include <cassert>
#include <thread>
int main(){using namespace ugkwpGpuIpc;int sockets[2];assert(socketpair(AF_UNIX,SOCK_STREAM,0,sockets)==0);
 ClientState state;state.fd=sockets[0];state.nFaces=2;state.gasModelConfigured=true;state.gasSpeciesIdentity={1,2,123,456,0};
 BoundaryLayerConfigV1 a;a.speciesCount=2;a.wallCount=1;a.matchingCount=1;
 std::uint64_t bytes;assert(boundaryLayerPayloadBytes(a,2,2,bytes));
 std::thread peer([&]{RequestHeader request{};assert(readAll(sockets[1],&request,sizeof(request)));
 assert(request.operation==static_cast<unsigned>(Op::configureBoundaryLayerV1)&&request.payloadBytes==bytes);
 std::vector<unsigned char> payload(bytes);assert(readAll(sockets[1],payload.data(),payload.size()));
 const auto* c=reinterpret_cast<const BoundaryLayerConfigV1*>(payload.data());assert(c->quadratureCount==0);
 ResponseHeader reply{magic,protocolMajor,protocolMinor,0,0,0};assert(sendObject(sockets[1],reply));close(sockets[1]);});
 int faces[]={0},qo[]={0,0},mo[]={0,1},mc[]={1};double geometry[]={1,0,0,.5,1.5,1,.5},mw[]={1},flux[]={0,0};
 assert(ugkwpGpuResidentStrictConfigureBoundaryLayerV1(&state,&a,faces,qo,mo,mc,geometry,nullptr,nullptr,mw,flux)==0);
 peer.join();close(sockets[0]);}
''')
    build=subprocess.run(['g++','-std=c++17','-pthread','-I'+str(ROOT),str(source),'-o',str(binary)],capture_output=True,text=True)
    assert build.returncode==0,build.stderr
    run=subprocess.run([str(binary)],capture_output=True,text=True)
    assert run.returncode==0,run.stderr
