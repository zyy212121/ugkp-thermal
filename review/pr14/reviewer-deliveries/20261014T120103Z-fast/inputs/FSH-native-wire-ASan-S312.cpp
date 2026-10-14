#include "applications/FSH/gpu/GpuBackendClient.C"
#include <cassert>
#include <thread>
#include <cstdio>
int main(){int sockets[2];assert(socketpair(AF_UNIX,SOCK_STREAM,0,sockets)==0);ClientState s;s.fd=sockets[0];s.nCells=2;s.nFaces=3;s.nCellPlanes=1;std::size_t total=0;
 std::thread peer([&]{for(int n=0;n<10;++n){RequestHeader request{};assert(receiveObject(sockets[1],request));std::vector<unsigned char>body(request.payloadBytes);assert(readAll(sockets[1],body.data(),body.size()));total+=body.size();assert(sendObject(sockets[1],ResponseHeader{magic,protocolMajor,protocolMinor,0,0,0}));}close(sockets[1]);});
 double values[128];int ints[128];unsigned char b[128];unsigned long long u[128];for(int i=0;i<128;++i){values[i]=i+.25;ints[i]=i;b[i]=i;u[i]=1000+i;}
assert(ugkwpGpuResidentStrictUploadMesh(&s,ints,ints,ints,values,values,values,values,values,values,values,values,values,values,values,values,values,values,values,values,values,ints,ints,ints,ints,ints,values,values,values,values,values,values)==0);
assert(ugkwpGpuResidentStrictUploadBoundarySources(&s,2,ints,ints,values,values,values,values,values,values,values,values,values,values)==0);
assert(ugkwpGpuResidentStrictConfigureParticleStuckModel(&s,2,b,1.25,2,1.25,1.25,1.25,1.25,1.25,1.25,1.25,1.25,2,2,1.25,1.25,1.25,1.25,1.25,1.25,1.25,2)==0);
assert(ugkwpGpuResidentStrictConfigureScheduledInlet(&s,2,ints,1.25,2,values,values,2,values,values)==0);
assert(ugkwpGpuResidentStrictUploadSourceResidualMass(&s,2,ints,values)==0);
assert(ugkwpGpuResidentStrictUploadGasBoundaryFields(&s,ints,ints,ints,ints,ints,ints,values,values,values,values,values,values,values,values,values)==0);
assert(ugkwpGpuResidentStrictUploadFields(&s,values,values,values,values,values,values,values,values,values,values)==0);
assert(ugkwpGpuResidentStrictConfigureSst(&s,1.25,1.25,1.25,1.25,1.25,1.25,1.25,1.25,1.25,1.25,1.25,1.25,1.25,1.25,1.25,2,1.25,1.25,1.25,values,values,values,ints,ints,values,values)==0);
assert(ugkwpGpuResidentStrictUploadParticleSavedVelocity(&s,2,values,values,values)==0);
assert(ugkwpGpuResidentStrictUploadParticleRestartMirror(&s,2,values,values,values,values,values,values,values,values,values,values,ints,ints,b,ints,values,values,values,values,values,values,values,values,values,values,values,u,u)==0);
 peer.join();close(sockets[0]);std::printf("PASS realPOSIX FSH nativeclient 10uploadframes payloadbytes=%zu normalcleanup\n",total);}
