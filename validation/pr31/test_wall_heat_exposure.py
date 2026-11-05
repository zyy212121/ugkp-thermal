"""Common geometric heat exposure and generic owner adapter."""
from pathlib import Path
import subprocess
from validation.tools.paths import SOURCE_ROOT as ROOT
SOURCE=r'''
#include <cmath>
#include <cassert>
#include "gasTransport/WallHeatExposure.H"
#define __device__
#define GPU_R(x) double(x)
#define asm(x) assert(false)
using GpuReal=double;
struct GasOnly{};
struct Contacts{
 int nInternalFaces=1,nFaces=3,particleStuckModelConfigured=1;bool particlesMayBePresent=true;
 unsigned char mask[3]{0,1,1},*particleStuckCandidateMask=mask;
 double area[3]{2,2,2},*magSf=area,occupied[3]{0,1,5},*particleWallRepresentedContactArea=occupied;
 double scale[3]{1,.5,1},*particleWallContactAreaScale=scale;
};
#include "GpuGasWallExposedArea.cuh"
int main(){
 double f=-1;assert(ugkwp::wallHeatExposedFraction(2.,1.,.5,f)&&f==.75);
 assert(ugkwp::wallHeatExposedFraction(2.,5.,1.,f)&&f==0.);
 assert(ugkwp::wallHeatExposedFraction(2.,0.,1.,f)&&f==1.);
 for(double bad:{-1.,double(INFINITY),double(NAN)})assert(!ugkwp::wallHeatExposedFraction(2.,bad,1.,f));
 assert(!ugkwp::wallHeatExposedFraction(0.,0.,1.,f));
 assert(!ugkwp::wallHeatExposedFraction(2.,1.,0.,f));
 assert(!ugkwp::wallHeatExposedFraction(2.,1.,2.,f));
 GasOnly gas;Contacts contact;
 assert(gasWallExposedAreaFraction(gas,1)==1.);
 assert(gasWallExposedAreaFraction(contact,0)==1.);
 assert(gasWallExposedAreaFraction(contact,1)==.75);
 assert(gasWallExposedAreaFraction(contact,2)==0.);
 contact.particlesMayBePresent=false;assert(gasWallExposedAreaFraction(contact,1)==1.);
}
'''
def test_shared_area_formula_and_owner_dispatch(tmp_path):
 source=tmp_path/'area.cpp';source.write_text('#include <initializer_list>\n'+SOURCE);exe=tmp_path/'area'
 result=subprocess.run(['g++','-std=c++17','-O2','-I'+str(ROOT/'common'),str(source),'-o',str(exe)],capture_output=True,text=True)
 assert result.returncode==0,result.stdout+result.stderr
 result=subprocess.run([str(exe)],capture_output=True,text=True)
 assert result.returncode==0,result.stdout+result.stderr
