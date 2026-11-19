import subprocess
import pytest
from validation.tools.paths import SOURCE_ROOT

@pytest.mark.parametrize("real", ["float", "double"])
def test_raw_face_energy_is_preserved_on_publication(tmp_path, real):
    source=(SOURCE_ROOT/"common/gpu/ParticleRadiationMaterial.cu").read_text()
    kernels=source[source.index("__global__ void rawBoundaryEnergy"):source.index("\n}\nstruct ParticleRadiationMaterial::Implementation")]
    code=r"""
#include <cassert>
#include <cmath>
#define __global__
using RadiationReal=REAL;
struct {int x=0;} threadIdx,blockIdx;
struct {int x=1;} blockDim;
namespace diagnostics {struct DeviceContext{bool fatal=false;};bool hasFatal(DeviceContext*p){return p&&p->fatal;}}
KERNELS
int main(){
 const RadiationReal flux[]={RadiationReal(1.5822635e-4),RadiationReal(-1.5817220e-4),RadiationReal(2e-12)};
 const RadiationReal area[]={2,4,8};
 const unsigned char coupled[]={1,0,1};
 RadiationReal raw[3],boundary[3],escaped=0;
 diagnostics::DeviceContext context;
 const double deviceElapsed=.125;
 for(int mode=0;mode<3;++mode){
  context.fatal=mode==2;
  const auto elapsed=mode==0?RadiationReal(.5):RadiationReal(deviceElapsed);
  for(blockIdx.x=0;blockIdx.x<3;++blockIdx.x)
   rawBoundaryEnergy(3,RadiationReal(.5),area,flux,raw,&context,mode?&deviceElapsed:nullptr);
  blockIdx.x=0;
  publishMaterialBoundary(3,raw,coupled,boundary,&escaped);
  for(int i=0;i<3;++i){
   const RadiationReal expected=context.fatal?0:elapsed*area[i]*flux[i];
   assert(raw[i]==expected&&boundary[i]==(coupled[i]?expected:0));
  }
  assert(escaped==raw[1]);
  assert(escaped+boundary[0]+boundary[2]==raw[1]+raw[0]+raw[2]);
 }
 publishMaterialBoundary(0,nullptr,nullptr,nullptr,&escaped);
 assert(escaped==0);
}
""".replace("REAL",real).replace("KERNELS",kernels)
    cpp=tmp_path/"raw.cpp";exe=tmp_path/"raw";cpp.write_text(code)
    subprocess.run(["g++","-std=c++17",str(cpp),"-o",str(exe)],check=True)
    subprocess.run([str(exe)],check=True)
