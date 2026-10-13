#include <cstdio>
#include "GpuPrecisionTypes.H"
#include "GpuColdWallSolidification.H"
using namespace Foam::gpuThermal;
int main(){ColdWallSolidificationParameters p{2327,20,1.16e6,3990,1273,5.9,.25,0,0,12};GpuReal h[8],rings[8]{},frozen=0;GpuTime age=.002;for(auto &v:h)v=coldWallSpecificEnthalpyJkg(GpuReal(50000),p);auto r=advanceColdWallProfile(h,rings,age,frozen,p,GpuReal(1e-12),GpuReal(3e-9),GpuReal(2.2e-8),GpuReal(1e-8),.01,GpuReal(.4),1e-10,GpuReal(1000),GpuReal(10000),GpuReal(1e-5),GpuReal(50000),GpuReal(0));printf("valid=%d mean=%.17g surface=%.17g wall=%.17g age=%.17g node0=%.17g\n",r.valid,double(r.meanTemperatureK),double(r.surfaceTemperatureK),double(r.wallEnergyJ),age,double(h[0]));printf("frozen=%a result_frozen=%a\n",double(frozen),double(r.frozenFootprintAreaM2));for(int j=0;j<8;++j)printf("node%d=%a ring%d=%a\n",j,double(h[j]),j,double(rings[j]));}
