"""FSH actual protocol/client/server boundaries, no GPU backend execution."""
from pathlib import Path
import importlib.util
import subprocess
import json
ROOT=Path(__file__).resolve().parents[1]
BASE='0136c834d76719d3232ae57211c7fe94bdbff4e6'
spec=importlib.util.spec_from_file_location('wire',ROOT/'applications/gasUGKP/tests/test_native_wire_continuation.py');wire=importlib.util.module_from_spec(spec);spec.loader.exec_module(wire)

def test_actual_unsigned_payload_arithmetic_continues(tmp_path):
    wire.run_cpp(tmp_path,r'''
#include "applications/FSH/gpu/GpuBackendProtocol.H"
#include <cassert>
int main(){using namespace ugkwpGpuIpc;std::uint64_t n=~std::uint64_t(0)-3;addBytes(n,1,4);assert(n==0);n=0;addArrayBytes<double>(n,~std::uint64_t(0));assert(n==~std::uint64_t(0)-7);}
''','FSH')

def test_negative_client_count_reaches_actual_io_failure(tmp_path):
    wire.run_cpp(tmp_path,r'''
#include "applications/FSH/gpu/GpuBackendClient.C"
#include <cassert>
int main(){int fds[2];assert(pipe(fds)==0);close(fds[0]);close(fds[1]);ClientState s;s.fd=fds[0];double x=1;assert(ugkwpGpuResidentStrictUploadParticleSavedVelocity(&s,-1,&x,&x,&x)!=0);assert(lastError.find("IPC write failed")!=std::string::npos);}
''','FSH')

def variant(tmp_path,kind,rev=None):
    folder=tmp_path/'sources';folder.mkdir()
    for name,p in [('GpuBackendProtocol.H','applications/FSH/gpu/GpuBackendProtocol.H'),('source.C',f'applications/FSH/'+('gpu/GpuBackendClient.C' if kind=='client' else 'private_backend/GpuBackendServer.C'))]:
        value=subprocess.check_output(['git','show',rev+':'+p],cwd=ROOT,text=True) if rev else (ROOT/p).read_text()
        if name=='GpuBackendProtocol.H':
            start=value.index('inline '+('bool' if rev else 'void')+' addBytes(')
            brace=value.index('{',start)
            value=value[:brace+1]+'++arithmeticCalls;'+value[brace+1:]
            value=value[:start]+'inline std::uint64_t arithmeticCalls=0;\n'+value[start:]
        (folder/name).write_text(value)
    return '#include "'+str(folder/'source.C')+'"\n'

def test_old_normal_all_upload_frames_and_byte_order_exact(tmp_path):
    import re
    src=(ROOT/'applications/FSH/gpu/GpuBackendClient.C').read_text()
    names=['UploadMesh','UploadBoundarySources','ConfigureParticleStuckModel','ConfigureScheduledInlet','UploadSourceResidualMass','UploadGasBoundaryFields','UploadFields','ConfigureSst','UploadParticleSavedVelocity','UploadParticleRestartMirror']
    calls=[]
    for n in names:
        signature=src.split('extern "C" int ugkwpGpuResidentStrict'+n,1)[1].split('{',1)[0]
        args=[]
        for p in signature.strip()[1:-1].split(','):
            p=p.strip()
            if 'void*' in p:args.append('&s')
            elif '*' in p:args.append('u' if 'unsigned long long' in p else 'b' if 'unsigned char' in p else 'ints' if 'int' in p else 'values')
            else:args.append('2' if 'int' in p else '1.25')
        calls.append('assert(ugkwpGpuResidentStrict'+n+'('+','.join(args)+')==0);')
    main=r'''
#include <cassert>
#include <thread>
#include <iostream>
#include <iomanip>
#include <vector>
int main(){int sockets[2];assert(socketpair(AF_UNIX,SOCK_STREAM,0,sockets)==0);ClientState s;s.fd=sockets[0];s.nCells=2;s.nFaces=3;s.nCellPlanes=1;
std::vector<unsigned char>all;std::thread peer([&]{for(int n=0;n<10;++n){RequestHeader request{};assert(receiveObject(sockets[1],request));auto bytes=encodeWire(request);all.insert(all.end(),bytes.begin(),bytes.end());std::vector<unsigned char>body(request.payloadBytes);assert(readAll(sockets[1],body.data(),body.size()));all.insert(all.end(),body.begin(),body.end());assert(sendObject(sockets[1],ResponseHeader{magic,protocolMajor,protocolMinor,0,0,0}));}close(sockets[1]);});
double values[128];int ints[128];unsigned char b[128];unsigned long long u[128];for(int i=0;i<128;++i){values[i]=i+.25;ints[i]=i;b[i]=i;u[i]=1000+i;}CALLS
peer.join();close(sockets[0]);std::cout<<"BYTES "<<all.size()<<" arithmetic "<<arithmeticCalls<<"\n";for(unsigned char byte:all)std::cout<<std::hex<<std::setw(2)<<std::setfill('0')<<int(byte);std::cout<<"\n";}
'''.replace('CALLS','\n'.join(calls))
    results={}
    for label,rev in [('old',BASE),('current',None)]:
        d=tmp_path/label;d.mkdir();results[label]=wire.run_cpp(d,variant(d,'client',rev)+main,'FSH',('-ftrivial-auto-var-init=zero',))
    assert results['old']==results['current'];(tmp_path/'frame-parity.json').write_text(json.dumps(results,indent=2))

def test_complete_residual_response_ignores_capacity_but_short_bytes_fail(tmp_path):
    wire.run_cpp(tmp_path,r'''
#include "applications/FSH/gpu/GpuBackendClient.C"
#include <cassert>
#include <thread>
int main(){int sockets[2];assert(socketpair(AF_UNIX,SOCK_STREAM,0,sockets)==0);ClientState s;s.fd=sockets[0];
std::thread peer([&]{for(int turn=0;turn<3;++turn){RequestHeader req{};CountArgs a{};assert(receiveObject(sockets[1],req)&&receiveObject(sockets[1],a));int n=2;assert(sendObject(sockets[1],ResponseHeader{magic,protocolMajor,protocolMinor,0,0,turn==2?4U:28U}));assert(sendObject(sockets[1],n));if(turn!=2){int faces[]={3,1};double mass[]={-.5,2.75};assert(sendArray(sockets[1],faces,2)&&sendArray(sockets[1],mass,2));}}close(sockets[1]);});
int count=0,faces[2];double mass[2];assert(ugkwpGpuResidentStrictDownloadSourceResidualMass(&s,&count,1,faces,mass)==0&&count==2&&faces[1]==1&&mass[0]==-.5);assert(ugkwpGpuResidentStrictDownloadSourceResidualMass(&s,&count,-7,faces,mass)==0&&count==2);assert(ugkwpGpuResidentStrictDownloadSourceResidualMass(&s,&count,1,faces,mass)!=0);peer.join();close(sockets[0]);}
''','FSH')

def test_server_negative_request_needs_actual_missing_bytes_not_count_policy(tmp_path):
    wire.run_cpp(tmp_path,r'''
#define main backend_main
#include "applications/FSH/private_backend/GpuBackendServer.C"
#undef main
#include <cassert>
#include <sys/socket.h>
extern "C" int ugkwpGpuResidentStrictUploadParticleSavedVelocity(void*,int,const double*,const double*,const double*){std::abort();}
extern "C" int ugkwpGpuResidentStrictDownloadParticleSavedVelocity(void*,int,double*,double*,double*){std::abort();}
extern "C" const char*ugkwpGpuResidentStrictLastError(){return "backend";}
int main(){int sockets[2];assert(socketpair(AF_UNIX,SOCK_STREAM,0,sockets)==0);ServerState s;s.backend=&s;CountArgs a{-1};assert(writeObject(sockets[1],a));RequestHeader req{magic,protocolMajor,protocolMinor,unsigned(Op::uploadParticleSavedVelocity),0,wireBytes(a)};readingFrame=true;frameBytes=frameRemaining=req.payloadBytes;assert(!handleSavedVelocity(sockets[0],req,s,true));readingFrame=false;ResponseHeader response{};assert(readObject(sockets[1],response));std::string error(response.errorBytes,' ');assert(readAll(sockets[1],error.data(),error.size()));assert(error.find("short GPU backend request")!=std::string::npos);close(sockets[0]);close(sockets[1]);}
''','FSH')

def test_server_old_residual_bytes_and_new_negative_query_count(tmp_path):
    main=r'''
#include <cassert>
#include <sys/socket.h>
#include <iostream>
#include <iomanip>
static int returnedCount=2,backendStatus=0;
extern "C" const char*ugkwpGpuResidentStrictLastError(){return "real backend error";}
extern "C" int ugkwpGpuResidentStrictDownloadSourceResidualMass(void*,int*n,int capacity,int*face,double*mass){*n=returnedCount;if(capacity){face[0]=3;face[1]=1;mass[0]=-.5;mass[1]=2.75;}return backendStatus;}
void run(int capacity){int sockets[2];assert(socketpair(AF_UNIX,SOCK_STREAM,0,sockets)==0);ServerState s;s.backend=&s;CountArgs a{capacity};assert(writeObject(sockets[1],a));RequestHeader req{magic,protocolMajor,protocolMinor,unsigned(Op::downloadSourceResidualMass),0,wireBytes(a)};readingFrame=true;frameBytes=frameRemaining=req.payloadBytes;assert(handleDownloadResidual(sockets[0],req,s));readingFrame=false;ResponseHeader response{};assert(readObject(sockets[1],response));std::vector<unsigned char>payload(response.errorBytes+response.payloadBytes);assert(readAll(sockets[1],payload.data(),payload.size()));std::cout<<response.status<<" "<<payload.size()<<" ";for(auto b:payload)std::cout<<std::hex<<std::setw(2)<<std::setfill('0')<<int(b);std::cout<<"\n";if(returnedCount<0&&backendStatus==0){assert(response.status==0&&response.payloadBytes==4);int n=0;WireCodec<int>::decode(payload.data(),n);assert(n==returnedCount);}close(sockets[0]);close(sockets[1]);}
int main(){run(2);run(0);backendStatus=7;run(0);EXTRA}
'''
    results={}
    for label,rev in [('old',BASE),('current',None)]:
        d=tmp_path/label;d.mkdir();include=variant(d,'server',rev)
        results[label]=wire.run_cpp(d,'#define main backend_main\n'+include+'#undef main\n'+main.replace('EXTRA',''),'FSH')
    assert results['old']==results['current'];(tmp_path/'server-parity.json').write_text(json.dumps(results,indent=2))
    d=tmp_path/'negative';d.mkdir();wire.run_cpp(d,'#define main backend_main\n'+variant(d,'server')+'#undef main\n'+main.replace('EXTRA','backendStatus=0;returnedCount=-3;run(0);'),'FSH')
