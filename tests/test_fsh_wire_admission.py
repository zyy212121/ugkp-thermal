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

def test_saved_velocity_client_capacity_before_io_and_valid_prefix(tmp_path):
    wire.run_cpp(tmp_path,r'''
#include "applications/FSH/gpu/GpuBackendClient.C"
#include <cassert>
#include <thread>
#include <limits>
int main(){int sockets[2];assert(socketpair(AF_UNIX,SOCK_STREAM,0,sockets)==0);ClientState s;s.fd=sockets[0];s.particleCapacity=5;
 double x[6]={1.+0x1p-30,-0.,std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN(),5.,-99.},y[6]={2,3,4,5,6,-99.},z[6]={7,8,9,10,11,-99.};
 for(int count:{-1,6,std::numeric_limits<int>::max()}){
  assert(ugkwpGpuResidentStrictUploadParticleSavedVelocity(&s,count,x,y,z)!=0);
  assert(lastError=="invalid saved velocity upload");
  assert(ugkwpGpuResidentStrictDownloadParticleSavedVelocity(&s,count,x,y,z)!=0);
  assert(lastError=="invalid saved velocity download");}
 assert(ugkwpGpuResidentStrictUploadParticleSavedVelocity(&s,1,nullptr,y,z)!=0);
 assert(ugkwpGpuResidentStrictDownloadParticleSavedVelocity(&s,1,x,nullptr,z)!=0);
 char pending;assert(recv(sockets[1],&pending,1,MSG_DONTWAIT)==-1&&(errno==EAGAIN||errno==EWOULDBLOCK));
 std::thread peer([&]{for(int count:{0,1,3,5})for(bool upload:{true,false}){
  RequestHeader req{};CountArgs a{};assert(receiveObject(sockets[1],req)&&receiveObject(sockets[1],a));
  assert(req.operation==unsigned(upload?Op::uploadParticleSavedVelocity:Op::downloadParticleSavedVelocity)&&a.count==count);
  assert(req.payloadBytes==wireBytes(a)+(upload?std::uint64_t(count)*24:0));
  if(upload){std::vector<double> values(count);for(auto* source:{x,y,z}){assert(receiveArray(sockets[1],values.data(),count));assert(std::memcmp(values.data(),source,count*8)==0);}}
  assert(sendObject(sockets[1],ResponseHeader{magic,protocolMajor,protocolMinor,0,0,upload?0U:std::uint64_t(count)*24}));
  if(!upload)for(auto* source:{x,y,z})assert(sendArray(sockets[1],source,count));}
 close(sockets[1]);});
 for(int count:{0,1,3,5}){assert(ugkwpGpuResidentStrictUploadParticleSavedVelocity(&s,count,count?x:nullptr,count?y:nullptr,count?z:nullptr)==0);
  double outX[6],outY[6],outZ[6];for(auto* values:{outX,outY,outZ})std::fill(values,values+6,-99.);
  assert(ugkwpGpuResidentStrictDownloadParticleSavedVelocity(&s,count,count?outX:nullptr,count?outY:nullptr,count?outZ:nullptr)==0);
  const double* inputs[]={x,y,z};double* outputs[]={outX,outY,outZ};for(int component=0;component<3;++component){assert(std::memcmp(inputs[component],outputs[component],count*8)==0);for(int i=count;i<6;++i)assert(outputs[component][i]==-99.);}}
 peer.join();close(sockets[0]);}
''','FSH')


def variant(tmp_path,kind,rev=None,clear_padding=False):
    folder=tmp_path/'sources';folder.mkdir()
    for name,p in [('GpuBackendProtocol.H','applications/FSH/gpu/GpuBackendProtocol.H'),('source.C',f'applications/FSH/'+('gpu/GpuBackendClient.C' if kind=='client' else 'private_backend/GpuBackendServer.C'))]:
        value=subprocess.check_output(['git','show',rev+':'+p],cwd=ROOT,text=True) if rev else (ROOT/p).read_text()
        if name=='GpuBackendProtocol.H':
            start=value.index('inline '+('bool' if rev else 'void')+' addBytes(')
            brace=value.index('{',start)
            value=value[:brace+1]+'++arithmeticCalls;'+value[brace+1:]
            value=value[:start]+'inline std::uint64_t arithmeticCalls=0;\n'+value[start:]
        if name=='source.C' and kind=='client' and clear_padding:
            native='''    if (wireLittleEndian() && WireCodec<T>::nativeLayout)
        return writeAll(fd,&value,WireCodec<T>::size);'''
            canonical='''    if (wireLittleEndian() && WireCodec<T>::nativeLayout)
    {
        T initialized=value;
        __builtin_clear_padding(&initialized);
        return writeAll(fd,&initialized,WireCodec<T>::size);
    }'''
            assert value.count(native)==1
            value=value.replace(native,canonical)
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
int main(){int sockets[2];assert(socketpair(AF_UNIX,SOCK_STREAM,0,sockets)==0);ClientState s;s.fd=sockets[0];s.nCells=2;s.nFaces=3;s.nCellPlanes=1;s.particleCapacity=2;
std::vector<unsigned char>all;std::thread peer([&]{for(int n=0;n<10;++n){RequestHeader request{};assert(receiveObject(sockets[1],request));auto bytes=encodeWire(request);all.insert(all.end(),bytes.begin(),bytes.end());std::vector<unsigned char>body(request.payloadBytes);assert(readAll(sockets[1],body.data(),body.size()));all.insert(all.end(),body.begin(),body.end());assert(sendObject(sockets[1],ResponseHeader{magic,protocolMajor,protocolMinor,0,0,0}));}close(sockets[1]);});
double values[128];int ints[128];unsigned char b[128];unsigned long long u[128];for(int i=0;i<128;++i){values[i]=i+.25;ints[i]=i;b[i]=i;u[i]=1000+i;}CALLS
peer.join();close(sockets[0]);std::cout<<"BYTES "<<all.size()<<" arithmetic "<<arithmeticCalls<<"\n";for(unsigned char byte:all)std::cout<<std::hex<<std::setw(2)<<std::setfill('0')<<int(byte);std::cout<<"\n";}
'''.replace('CALLS','\n'.join(calls))
    results={}
    for label,rev in [('old',BASE),('current',None)]:
        d=tmp_path/label;d.mkdir();results[label]=wire.run_cpp(d,variant(d,'client',rev,clear_padding=True)+main,'FSH')
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

def test_saved_velocity_server_capacity_refusal_drains_frame_and_retries(tmp_path):
    folder=tmp_path/'server';folder.mkdir()
    text=(ROOT/'applications/FSH/private_backend/GpuBackendServer.C').read_text()
    allocation='    std::vector<double> x(a.count),y(a.count),z(a.count);'
    assert text.count(allocation)==1
    (folder/'source.C').write_text('int savedVelocityArrayAllocations=0;\n'+text.replace(allocation,'    ++savedVelocityArrayAllocations;\n'+allocation))
    wire.run_cpp(tmp_path,'#define main backend_main\n#include "'+str(folder/'source.C')+'"\n#undef main\n'+r'''
#include <cassert>
#include <sys/socket.h>
#include <limits>
int nativeCalls=0,nativeStatus=0;double expectedX[]={1.+0x1p-30,-0.,std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN(),5.};double expectedY[]={2,3,4,5,6},expectedZ[]={7,8,9,10,11};
extern "C" int ugkwpGpuResidentStrictUploadParticleSavedVelocity(void*,int count,const double*x,const double*y,const double*z){++nativeCalls;const double* actual[]={x,y,z};const double* expected[]={expectedX,expectedY,expectedZ};for(int k=0;k<3;++k){if(count)assert(std::memcmp(actual[k],expected[k],count*8)==0);else assert(!actual[k]);}return nativeStatus;}
extern "C" int ugkwpGpuResidentStrictDownloadParticleSavedVelocity(void*,int count,double*x,double*y,double*z){++nativeCalls;double* actual[]={x,y,z};const double* expected[]={expectedX,expectedY,expectedZ};for(int k=0;k<3;++k){if(count)std::memcpy(actual[k],expected[k],count*8);else assert(!actual[k]);}return nativeStatus;}
extern "C" const char*ugkwpGpuResidentStrictLastError(){return "injected native copy failure";}
int main(){int sockets[2];assert(socketpair(AF_UNIX,SOCK_STREAM,0,sockets)==0);ServerState s;s.backend=&s;s.particleCapacity=5;
 const auto request=[&](int count,bool upload,bool invalid){CountArgs a{count};RequestHeader req{magic,protocolMajor,protocolMinor,unsigned(upload?Op::uploadParticleSavedVelocity:Op::downloadParticleSavedVelocity),0,wireBytes(a)+(invalid?17U:upload?std::uint64_t(count)*24:0U)};
  assert(writeObject(sockets[1],req)&&writeObject(sockets[1],a));
  if(invalid){unsigned char tail[17]{};assert(writeAll(sockets[1],tail,sizeof(tail)));}
  else if(upload)for(auto* data:{expectedX,expectedY,expectedZ})assert(writeAll(sockets[1],data,count*8));
  RequestHeader received{};assert(readObject(sockets[0],received));readingFrame=true;frameBytes=frameRemaining=received.payloadBytes;frameReadError.clear();const int calls=nativeCalls,allocations=savedVelocityArrayAllocations;
  assert(handleSavedVelocity(sockets[0],received,s,upload));assert(discardRequest(sockets[0])&&!readingFrame&&frameRemaining==0);
  ResponseHeader response{};assert(readObject(sockets[1],response));std::string error(response.errorBytes,' ');assert(readAll(sockets[1],error.data(),error.size()));
  if(invalid){assert(response.status!=0&&response.payloadBytes==0&&error.find("invalid saved velocity count")!=std::string::npos);assert(nativeCalls==calls&&savedVelocityArrayAllocations==allocations);}
  else {assert(nativeCalls==calls+1&&savedVelocityArrayAllocations==allocations+1&&response.status==nativeStatus);if(nativeStatus){assert(response.payloadBytes==0&&error=="injected native copy failure");}else {assert(error.empty()&&response.payloadBytes==(upload?0U:std::uint64_t(count)*24));if(!upload)for(auto* expected:{expectedX,expectedY,expectedZ}){double values[5]{};assert(readAll(sockets[1],values,count*8));assert(std::memcmp(values,expected,count*8)==0);}}}
 };
 for(bool upload:{true,false})for(int count:{-1,6,std::numeric_limits<int>::max()}){request(count,upload,true);request(1,upload,false);}
 for(bool upload:{true,false})for(int count:{0,1,3,5})request(count,upload,false);
 nativeStatus=7;for(bool upload:{true,false})request(1,upload,false);
 nativeStatus=0;request(5,true,false);request(5,false,false);close(sockets[0]);close(sockets[1]);}
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
