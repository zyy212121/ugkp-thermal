#include "fvCFD.H"
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
