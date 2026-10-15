"""Execute the real FSH mirror client and server framing with double payloads."""
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from pathlib import Path
import json
import re
import subprocess
import pytest
ROOT=SOURCE_ROOT
UPLOAD='ugkwpGpuResidentStrictUploadParticleRestartMirror'
DOWNLOAD='ugkwpGpuResidentStrictDownloadParticleRestartMirror'


def signature(name):
    source=(ROOT/'applications/FSH/gpu/GpuBackendApi.H').read_text()
    return re.search(r'int\s+'+name+r'\s*\((.*?)\);',source,re.S).group(1)


def fields():
    meta={f['name']:f for f in json.loads((ROOT/'common/ParticleFieldManifest.json').read_text())['fields']}
    result=[]
    for arg in signature(UPLOAD).split(',')[2:]:
        name=arg.split()[-1].lstrip('*');f=meta[name]
        result.append((name,f['host']['FSH'],f['width']))
    return result


def value(k,typ,index='j'):
    return f'static_cast<{typ}>({k+1}+.25+{index}*.125+0x1p-30)' if typ=='double' else f'static_cast<{typ}>({k+1}+{index})'


def run(tmp_path,code):
    source=tmp_path/'ipc.cpp';exe=tmp_path/'ipc';source.write_text(code)
    cmd=['g++','-std=c++17','-O0','-pthread','-ffunction-sections','-fdata-sections','-Wl,--gc-sections','-I'+str(ROOT),'-I'+str(ROOT/'applications/FSH/gpu'),str(source),'-o',str(exe)]
    subprocess.run(cmd,check=True)
    subprocess.run([str(exe)],check=True,timeout=30)


def test_fsh_actual_client_upload_download_and_count_only(tmp_path):
    fs=fields();decl=[];recv=[];send=[];checks=[]
    for k,(name,typ,width) in enumerate(fs):
        decl.append(f'std::vector<{typ}> {name}(2*{width});for(int j=0;j<2*{width};++j){name}[j]={value(k,typ)};')
        recv.append(f'{{std::vector<{typ}> got(2*{width});assert(receiveArray(peer,got.data(),got.size()));assert(got=={name});}}')
        send.append(f'assert(sendArray(peer,{name}.data(),{name}.size()));')
        checks.append(f'for(int j=0;j<2*{width};++j)assert({name}[j]=={value(k,typ)});')
    args=','.join(name+'.data()' for name,_,_ in fs)
    run(tmp_path,'''#include "applications/FSH/gpu/GpuBackendClient.C"
#include <cassert>
#include <thread>
#include <vector>
int main(){using namespace ugkwpGpuIpc;
DECL
int sockets[2];assert(socketpair(AF_UNIX,SOCK_STREAM,0,sockets)==0);ClientState s;s.fd=sockets[0];s.particleCapacity=2;const int peer=sockets[1];
std::thread worker([&]{RequestHeader h{};CountArgs a{};
assert(receiveObject(peer,h)&&receiveObject(peer,a));assert(a.count==2&&h.payloadBytes==wireBytes(a)+2*869);RECV
assert(sendObject(peer,ResponseHeader{magic,protocolMajor,protocolMinor,0,0,0}));
assert(receiveObject(peer,h)&&receiveObject(peer,a));assert(a.count==2);
assert(sendObject(peer,ResponseHeader{magic,protocolMajor,protocolMinor,0,0,wireBytes(int{})+2*869}));assert(sendObject(peer,2));SEND
assert(receiveObject(peer,h)&&receiveObject(peer,a));assert(a.count==0);
assert(sendObject(peer,ResponseHeader{magic,protocolMajor,protocolMinor,0,0,wireBytes(int{})}));assert(sendObject(peer,2));
});
assert(UPLOAD(&s,2,ARGS)==0);int count=-1;
assert(DOWNLOAD(&s,&count,2,ARGS)==0);assert(count==2);CHECKS
assert(DOWNLOAD(&s,&count,0,ARGS)==0);assert(count==2);
worker.join();close(sockets[0]);close(sockets[1]);}
'''.replace('DECL',''.join(decl)).replace('RECV',''.join(recv)).replace('SEND',''.join(send)).replace('CHECKS',''.join(checks)).replace('ARGS',args).replace('UPLOAD',UPLOAD).replace('DOWNLOAD',DOWNLOAD))


def test_fsh_actual_server_upload_download_and_count_only(tmp_path):
    fs=fields();upload=[];download=[];decl=[];send=[];recv=[]
    for k,(name,typ,width) in enumerate(fs):
        upload.append(f'for(int j=0;j<nParticles*{width};++j)assert({name}[j]=={value(k,typ)});')
        download.append(f'if(maxParticles)for(int j=0;j<2*{width};++j){name}[j]={value(k,typ)};')
        decl.append(f'std::vector<{typ}> {name}(2*{width});for(int j=0;j<2*{width};++j){name}[j]={value(k,typ)};')
        send.append(f'assert(writeVector(peer,{name}));')
        recv.append(f'{{std::vector<{typ}> got;assert(readVector(peer,got,2*{width}));assert(got=={name});}}')
    run(tmp_path,'''#define main backend_main
#include "applications/FSH/private_backend/GpuBackendServer.C"
#undef main
#include <cassert>
#include <sys/socket.h>
extern "C" const char* ugkwpGpuResidentStrictLastError(){return "test backend";}
extern "C" int UPLOAD(UPLOADSIG){assert(nParticles==2);UPCHECK;return 0;}
extern "C" int DOWNLOAD(DOWNSIG){*nParticles=2;DOWNFILL;return 0;}
int main(){using namespace ugkwpGpuIpc;DECL
int sockets[2];assert(socketpair(AF_UNIX,SOCK_STREAM,0,sockets)==0);const int peer=sockets[1];ServerState s;s.backend=&s;s.particleCapacity=2;
CountArgs a{2};RequestHeader request{magic,protocolMajor,protocolMinor,unsigned(Op::uploadParticleRestartMirror),0,wireBytes(a)+2*869};
assert(writeObject(peer,a));SEND
readingFrame=true;frameRemaining=request.payloadBytes;assert(handleUploadParticles(sockets[0],request,s));assert(frameRemaining==0);readingFrame=false;
ResponseHeader response{};assert(readObject(peer,response));assert(response.status==0&&response.payloadBytes==0);
for(int capacity:{2,0}){a.count=capacity;request.payloadBytes=wireBytes(a);request.operation=unsigned(Op::downloadParticleRestartMirror);
assert(writeObject(peer,a));readingFrame=true;frameRemaining=request.payloadBytes;assert(handleDownloadParticles(sockets[0],request,s));assert(frameRemaining==0);readingFrame=false;
assert(readObject(peer,response));assert(response.status==0&&response.payloadBytes==wireBytes(int{})+(capacity?2*869:0));int count;assert(readObject(peer,count));assert(count==2);if(capacity){RECV}}
close(sockets[0]);close(sockets[1]);}
'''.replace('UPLOADSIG',signature(UPLOAD)).replace('DOWNSIG',signature(DOWNLOAD)).replace('UPLOAD',UPLOAD).replace('DOWNLOAD',DOWNLOAD).replace('UPCHECK',''.join(upload)).replace('DOWNFILL',''.join(download)).replace('DECL',''.join(decl)).replace('SEND',''.join(send)).replace('RECV',''.join(recv)))
