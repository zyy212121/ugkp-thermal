"""Independent material surface selection preserves legacy application semantics."""
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[2]

def test_surface_fields_are_independent_from_injected_gas(tmp_path):
    source=tmp_path/"surface.cpp"
    source.write_text(r'''
#include "thermal/ParticleSurfaceState.H"
#include <cassert>
struct Gas {double* gasBoundaryT;double* gasBoundaryUx;double* gasBoundaryUy;double* gasBoundaryUz;};
struct Coupled:Gas {ugkp::thermal::ParticleSurfaceState particleSurface;};
int main(){
 double t[2]={600,700},x[2]={4,5},y[2]={6,7},z[2]={8,9};Gas old{t,x,y,z};
 Coupled s;s.gasBoundaryT=t;s.gasBoundaryUx=x;s.gasBoundaryUy=y;s.gasBoundaryUz=z;
 for(int f=0;f<2;++f){assert(ugkwp::particleWallTemperature(old,f)==t[f]);
  assert(ugkwp::particleWallVelocityX(old,f)==x[f]);assert(ugkwp::particleWallVelocityY(old,f)==y[f]);assert(ugkwp::particleWallVelocityZ(old,f)==z[f]);
  assert(ugkwp::particleWallTemperature(s,f)==t[f]);assert(ugkwp::particleWallVelocityX(s,f)==x[f]);}
 double wt[2]={100,200},wx[2]={.1,.2},wy[2]={.3,.4},wz[2]={.5,.6};s.particleSurface={wt,wx,wy,wz};
 for(int f=0;f<2;++f){assert(ugkwp::particleWallTemperature(s,f)==wt[f]);assert(ugkwp::particleWallVelocityX(s,f)==wx[f]);
  assert(ugkwp::particleWallVelocityY(s,f)==wy[f]);assert(ugkwp::particleWallVelocityZ(s,f)==wz[f]);}
 t[0]=3000;x[0]=12345;y[0]=-9876;z[0]=5555;
 assert(ugkwp::particleWallTemperature(s,0)==100);assert(ugkwp::particleWallVelocityX(s,0)==.1);
 double wallE[2]={1000,2000};
 assert(ugkwp::particleWallEffusivity(old,0,4321.)==4321.);
 assert(ugkwp::particleWallEffusivity(s,0,4321.)==4321.);
 s.particleSurface.effusivity=wallE;assert(ugkwp::particleWallEffusivity(s,0,4321.)==1000.);
 s.particleSurface={};assert(ugkwp::particleWallTemperature(s,0)==3000);
}
''')
    binary=tmp_path/"surface"
    subprocess.run(["g++","-std=c++17","-O2","-Wall","-Wextra","-Werror","-I"+str(ROOT/"common"),str(source),"-o",str(binary)],check=True)
    subprocess.run([binary],check=True)
