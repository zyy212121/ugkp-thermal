pass
import subprocess
import pytest
from validation.tools.paths import SOURCE_ROOT

@pytest.mark.parametrize("real", ["float", "double"])
def test_raw_face_exchange_survives_prediction_and_actual_acceptance(tmp_path, real):
    source=(SOURCE_ROOT/"common/gpu/ParticleRadiationMaterial.cu").read_text()
    kernels=source[source.index("__global__ void publishMaterialBoundary"):source.index("\n}\nstruct ParticleRadiationMaterial::Implementation")]
    code=r"""
#include <cassert>
#include <cmath>
#define __global__
using RadiationReal=REAL;
struct {int x=0;} threadIdx;
namespace diagnostics {struct DeviceContext{bool fatal=false;};bool hasFatal(DeviceContext*p){return p&&p->fatal;}}
KERNELS
int main(){
 RadiationReal raw[]={RadiationReal(1.5822635e-4),RadiationReal(-1.5817220e-4),RadiationReal(2e-12)};
 unsigned char coupled[]={1,0,1};
 RadiationReal magnitude[]={1e-8,2e-8},total=0,boundary[3],escaped=0,accepted=0,gross=0;
 publishMaterialBoundary(2,3,raw,coupled,magnitude,&total,boundary,&escaped,&gross);
 assert(boundary[0]==raw[0]&&boundary[1]==0&&boundary[2]==raw[2]&&escaped==raw[1]);
 assert(gross==std::abs(raw[0])+std::abs(raw[1])+std::abs(raw[2]));
 diagnostics::DeviceContext context;
 RadiationReal actual[]={RadiationReal(3e-9),RadiationReal(-1e-9)};
 settleMaterialBoundary(2,actual,&accepted,&context);
 assert(accepted==actual[0]+actual[1]);
 assert(boundary[0]==raw[0]&&boundary[2]==raw[2]&&escaped==raw[1]);
 // Zero actual acceptance must not erase finite opposing face exchange.
 settleMaterialBoundary(2,nullptr,&accepted,&context);
 assert(accepted==0&&boundary[0]==raw[0]&&escaped==raw[1]);
 const RadiationReal balance=accepted+boundary[0]+boundary[2]+escaped;
 assert(balance!=0); // The real imbalance is not forcibly closed.
 context.fatal=true;settleMaterialBoundary(2,actual,&accepted,&context);
 assert(boundary[0]==raw[0]&&escaped==raw[1]);
}
""".replace("REAL",real).replace("KERNELS",kernels)
    cpp=tmp_path/"raw.cpp";exe=tmp_path/"raw";cpp.write_text(code)
    subprocess.run(["g++","-std=c++17",str(cpp),"-o",str(exe)],check=True)
    subprocess.run([str(exe)],check=True)
