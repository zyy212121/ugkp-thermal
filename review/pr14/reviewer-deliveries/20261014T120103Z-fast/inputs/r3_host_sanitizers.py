from r3_runner import *
source=D/'inputs/native-mapper-ASan-S312.C';source.write_text(r'''#include "fvCFD.H"
#include "thermal/GpuSolidPatchMapper.H"
#include <cassert>
#include <cstdio>
int main(int argc,char** argv){Foam::argList args(argc,argv);Foam::Time time(Foam::Time::controlDictName,args);Foam::fvMesh mesh(Foam::IOobject(Foam::fvMesh::defaultRegion,time.timeName(),time,Foam::IOobject::MUST_READ));
 using namespace Foam;using namespace Foam::gpuThermal;
 for(int mode=0;mode<3;++mode){List<SolidPatchPair> pairs(mode);if(mode>0)pairs[0]={"left","left"};if(mode>1)pairs[1]={"right","right"};
 OneToOneConformalSolidPatchMapper owner(mesh,mesh,pairs);assert(owner.size()==mode);assert(owner.fluidPatchIds().size()==mode&&owner.solidPatchIds().size()==mode);
 for(int pair=0;pair<mode;++pair){const int n=mesh.boundary()[owner.fluidPatchIds()[pair]].size();scalarField heat(n),temperature(n);for(int i=0;i<n;++i){heat[i]=(pair+1)*.03125;temperature[i]=300.+pair+2.*i;}
  auto actualHeat=owner.mapIntegratedFaceEnergyToSolid(pair,heat);auto actualT=owner.mapSolidScalarToFluid(pair,temperature);assert(actualHeat==heat&&actualT==temperature);std::printf("MAPPER pair=%d faces=%d heat=%a temperature=%a\n",pair,n,n?actualHeat[0]:0,n?actualT[0]:0);}
 std::printf("PASS actualnative hostmapper selectedPairs=%d\n",mode);}
}
''')
with (D/'TEST_PLAN.md').open('a') as h:h.write('\nS312 codec之外host ASan/UBSan：真实OpenFOAM fvMesh和未改GpuSolidPatchMapper.C，空pairlist、single、multiple(left/right)选择，实际scalar/energy两向mapper逐bit identity与无工作empty。这是native host owner/mapper/allocator析构范围，coupler入口的empty dictionary原语义不由这里关闭，真实GPU multi-selectedconsumer另列缺口。不用hostmock代GPU。FSH真实client/协议socket normal10upload frame的ASan/UBSan也单列，不链接mockbackend或宣称GPU验收。compiler900/run300。\n')
app=ROOT/'applications/CHT';binary=ROOT/'work/fixtures/native-mapper-ASan-S312';flags=['-std=c++17','-O1','-g','-fsanitize=address,undefined','-fno-omit-frame-pointer','-DNoRepository','-DWM_DP','-DWM_LABEL_SIZE=32','-I'+str(app),'-I'+str(ROOT/'common'),'-I/opt/openfoam10/src/OpenFOAM/lnInclude','-I/opt/openfoam10/src/finiteVolume/lnInclude','-I/opt/openfoam10/src/meshTools/lnInclude','-I/opt/openfoam10/src/OSspecific/POSIX/lnInclude'];libs=['-L/opt/openfoam10/platforms/linux64GccDPInt32Opt/lib','-lfiniteVolume','-lmeshTools','-lOpenFOAM','-L/opt/openfoam10/platforms/linux64GccDPInt32Opt/lib/dummy','-lPstream']
t=run('BUILD-NATIVE-MAPPER-ASAN-S312',cmd('g++',*flags,source,app/'thermal/GpuSolidPatchMapper.C',*libs,'-o',binary),[18,20,23,26,28,33],900,'ActualOpenFoam/native mapper hostASanUBSan, no substitutehostkernel.',inputs=[source,app/'thermal/GpuSolidPatchMapper.C'],kind='BUILD',variant='SUPPORT_S312');assert t['exit_code']==0
case=ROOT/'work/cases/phase/onecell-melting-S308';t=run('HOST-NATIVE-MAPPER-ASAN-S312',cmd(binary,'-case',case),[18,20,23,26,28,33],300,'Actualhostmapper empty/single/multiple andnormaldestruction; GPUconsumernotclaimed.',inputs=[source,case/'constant/polyMesh/points'],binary=binary,kind='HOST_SANITIZER',variant='SUPPORT_S312');raw=(D/t['log']).read_text();assertion(t,t['exit_code']==0 and all('selectedPairs='+str(i) in raw for i in range(3)),'Actualhostempty/single/multiple mapping/normaldestruction ASanUBSan, noGPUconsumerclaim.')
# Real IPC client and protocol, ordinary two-record uploads over actual POSIX socket.
client=ROOT/'applications/FSH/gpu/GpuBackendClient.C';text=client.read_text();names=['UploadMesh','UploadBoundarySources','ConfigureParticleStuckModel','ConfigureScheduledInlet','UploadSourceResidualMass','UploadGasBoundaryFields','UploadFields','ConfigureSst','UploadParticleSavedVelocity','UploadParticleRestartMirror'];calls=[]
for n in names:
 sig=text.split('extern "C" int ugkwpGpuResidentStrict'+n,1)[1].split('{',1)[0];args=[]
 for p in sig.strip()[1:-1].split(','):
  p=p.strip();args.append('&s' if 'void*' in p else ('u' if 'unsigned long long' in p else 'b' if 'unsigned char' in p else 'ints' if 'int' in p else 'values') if '*' in p else '2' if 'int' in p else '1.25')
 calls.append('assert(ugkwpGpuResidentStrict'+n+'('+','.join(args)+')==0);')
wire=D/'inputs/FSH-native-wire-ASan-S312.cpp';wire.write_text('#include "applications/FSH/gpu/GpuBackendClient.C"\n'+r'''#include <cassert>
#include <thread>
#include <cstdio>
int main(){int sockets[2];assert(socketpair(AF_UNIX,SOCK_STREAM,0,sockets)==0);ClientState s;s.fd=sockets[0];s.nCells=2;s.nFaces=3;s.nCellPlanes=1;std::size_t total=0;
 std::thread peer([&]{for(int n=0;n<10;++n){RequestHeader request{};assert(receiveObject(sockets[1],request));std::vector<unsigned char>body(request.payloadBytes);assert(readAll(sockets[1],body.data(),body.size()));total+=body.size();assert(sendObject(sockets[1],ResponseHeader{magic,protocolMajor,protocolMinor,0,0,0}));}close(sockets[1]);});
 double values[128];int ints[128];unsigned char b[128];unsigned long long u[128];for(int i=0;i<128;++i){values[i]=i+.25;ints[i]=i;b[i]=i;u[i]=1000+i;}
'''+ '\n'.join(calls)+r'''
 peer.join();close(sockets[0]);std::printf("PASS realPOSIX FSH nativeclient 10uploadframes payloadbytes=%zu normalcleanup\n",total);}
''')
binary=ROOT/'work/fixtures/FSH-native-wire-ASan-S312';t=run('BUILD-NATIVE-WIRE-ASAN-S312',cmd('g++','-std=c++17','-O1','-g','-pthread','-fsanitize=address,undefined','-fno-omit-frame-pointer','-ffunction-sections','-Wl,--gc-sections','-I'+str(ROOT),wire,'-o',binary),[19,20,26,28],900,'ActualFSH client/protocol +nativePOSIX socket peer; host-onlyASanUBSan, noGPUmock.',inputs=[wire,client,ROOT/'applications/FSH/gpu/GpuBackendProtocol.H'],kind='BUILD',variant='SUPPORT_S312');assert t['exit_code']==0
t=run('HOST-NATIVE-WIRE-ASAN-S312',cmd(binary),[19,20,26,28],300,'ActualrealIPC ten ordinaryuploadframes andnormalcleanup; noGPUbackendnumericalclaim.',inputs=[wire],binary=binary,kind='HOST_SANITIZER',variant='SUPPORT_S312');raw=(D/t['log']).read_text();assertion(t,t['exit_code']==0 and 'PASS realPOSIX' in raw,'Actualnativeclientprotocol10frames ASanUBSan/normalcleanup; server/GPU/faultconcurrency remainseparate.')
