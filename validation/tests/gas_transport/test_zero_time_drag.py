"""Zero-time production drag updates caches, never conserved inventories."""
import pytest
from test_mixture_state import compile_probe
from test_legacy_ale_transport import fixture


def test_zero_time_drag_preserves_homogeneous_conserved_bits(tmp_path):
    compile_probe(tmp_path, fixture()+r'''
struct DeviceState:State{
 Real epsSMin=1e-10,rhoSolid=1000,particleDiameterFallback=.001;
 Real couplingRhoOld[2]={},couplingUxOld[2]={},couplingUyOld[2]={},couplingUzOld[2]={},couplingTgasOld[2]={};
 Real thetaDragAlpha[2]={},momRhoP[2]={100,100},momRhoUPx[2]={},momRhoUPy[2]={},momRhoUPz[2]={},momRhoEP[2]={},momRhoPD[2]={.1,.1};
};
Real sqr3(Real x,Real y,Real z){return x*x+y*y+z*z;}
Real solidEpsFromMomentDevice(const DeviceState&s,int c){return s.momRhoP[c]/s.rhoSolid;}
template<int Model>Real cellDragInverseTime(DeviceState&,int,Real,Real,Real,Real,int){return Real(1);}
template<int DragModel>void drag(DeviceState*sp,double dt,int dragModel){
#include "GpuCellDragUpdate.inl"
}
int main(){DeviceState s;initialise(s);s.nCells=1;s.rho[0]=Real(1.1);s.rhoUx[0]=Real(19999920);s.Ux[0]=s.rhoUx[0]/s.rho[0];
s.rhoE[0]=Real(double(s.rhoUx[0])*double(s.rhoUx[0])/(2*double(s.rho[0]))+1e6);
s.momRhoUPx[0]=s.momRhoP[0]*s.Ux[0];s.momRhoEP[0]=Real(.5)*s.momRhoP[0]*s.Ux[0]*s.Ux[0];
const Real rho=s.rho[0],mx=s.rhoUx[0],energy=s.rhoE[0];drag<0>(&s,0.,0);
ck(std::memcmp(&rho,s.rho,sizeof rho)==0&&std::memcmp(&mx,s.rhoUx,sizeof mx)==0&&std::memcmp(&energy,s.rhoE,sizeof energy)==0,"zero-time drag changed conserved inventory bits");
ck(s.thetaDragAlpha[0]==1&&s.couplingUxOld[0]==s.Ux[0],"zero-time drag did not refresh required coupling caches");}
''')
