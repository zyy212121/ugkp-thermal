"""Execute each reachable CHT gas-energy source with independently typed energy."""
import pytest
from test_mixture_state import compile_probe
from test_authoritative_energy import energy_fixture


def source_fixture():
    return energy_fixture()+r'''
#include "wall/GpuAluminaProperties.H"
#include "gasNumerics/GpuParticlePhysicsAlgebra.cuh"
#include "gpu/GpuCouplingMath.H"
#define GPU_OPERATOR_THERMAL 1
struct DeviceState:State{
 Real gravityX=0,gravityY=0,gravityZ=0,epsSMin=1e-10,rhoSolid=1000,particleDiameterFallback=.001,TpMin=1,TpMax=10000;
 int solveParticleTemperature=1,particleGasHeatTransferModelId=1;
 Real epsGPrev[2]={.9,.9},couplingRhoOld[2]={},couplingUxOld[2]={},couplingUyOld[2]={},couplingUzOld[2]={},couplingTgasOld[2]={};
 Real thetaDragAlpha[2]={},momRhoP[2]={100,100},momRhoUPx[2]={},momRhoUPy[2]={},momRhoUPz[2]={},momRhoEP[2]={},momRhoPD[2]={.1,.1},momRhoHpP[2]={30000000,30000000};
};
Real sqr3(Real x,Real y,Real z){return x*x+y*y+z*z;}
Real solidEpsFromMomentDevice(const DeviceState&s,int c){return s.momRhoP[c]/s.rhoSolid;}
Real particleTemperatureFromSpecificEnthalpyDevice(Real hp){return hp/Real(1000);}
Real particleSpecificHeatDevice(Real){return Real(1000);}
template<int Model>Real cellDragInverseTime(DeviceState&,int,Real,Real,Real,Real,int){return Real(1);}
#include "GpuGravityUpdate.cuh"
#include "operators/applyGasVolumeFractionSourceKernel.cuh"
#include "operators/applyEulerianParticleMaterialHeatKernel.cuh"
template<int DragModel>void drag(DeviceState*sp,double dt,int dragModel){
#include "GpuCellDragUpdate.inl"
}
void setup(DeviceState&s){initialise(s);s.nCells=1;s.rhoE[0]=1e9+.03;s.momRhoEP[0]=Real(.03);s.epsGPrev[0]=Real(1)-solidEpsFromMomentDevice(s,0);s.couplingRhoOld[0]=s.rho[0];s.couplingTgasOld[0]=300;}
'''


@pytest.mark.parametrize('source',['gravity','volume','heat','drag'])
def test_actual_sources_preserve_current_energy_offset(tmp_path,source):
    actions={
        'gravity':'applyGasGravityCell<true>(s,0,.25);',
        'volume':'s.cellPlaneCount[0]=0;applyGasVolumeFractionSourceKernel(&s,.25);',
        'heat':'applyEulerianParticleMaterialHeatKernel(&s,.25);',
        'drag':'s.momRhoEP[0]=0;drag<0>(&s,.25,0);',
    }
    compile_probe(tmp_path,source_fixture()+r'''
int main(){DeviceState s;setup(s);const double before=s.rhoE[0];ACTION
closeEnergy(s.rhoE[0],before,"SOURCE source narrowed current energy");}
'''.replace('ACTION',actions[source]).replace('SOURCE',source))


def test_gravity_retains_small_mechanical_work(tmp_path):
    compile_probe(tmp_path,source_fixture()+r'''
int main(){DeviceState s;setup(s);s.gravityX=Real(.2);const double before=s.rhoE[0];applyGasGravityCell<false>(s,0,.5);
const long double work=.5L*(long double)s.rho[0]*.5L*.5L*(long double)s.gravityX*(long double)s.gravityX;
closeEnergy(s.rhoE[0],(long double)before+work,"gravity work lost");}
''')


def test_drag_gas_gain_retains_resolved_internal_energy_transfer(tmp_path):
    compile_probe(tmp_path,source_fixture()+r'''
int main(){DeviceState s;setup(s);const double before=s.rhoE[0];const Real solidBefore=s.momRhoEP[0];drag<0>(&s,.25,0);
const Real epsG=Real(1)-solidEpsFromMomentDevice(s,0);const Real configuredParticleLoss=solidBefore-solidBefore*s.thetaDragAlpha[0];
closeEnergy(s.rhoE[0],(long double)before+(long double)configuredParticleLoss/(long double)epsG,"drag gas gain does not match configured internal-energy transfer");}
''')


def test_material_heat_gas_decrement_retains_configured_gas_side_exchange(tmp_path):
    compile_probe(tmp_path,source_fixture()+r'''
int main(){DeviceState s;setup(s);s.gasMu=Real(1e-5);s.couplingTgasOld[0]=500;const double dt=1e-8,before=s.rhoE[0];
const Real epsG=Real(1)-solidEpsFromMomentDevice(s,0),rhoG=s.couplingRhoOld[0];
const Real diameter=s.momRhoPD[0]/s.momRhoP[0],cpP=particleSpecificHeatDevice(300);
const Real gasCv=s.Rgas/clampMin(s.gammaGas-Real(1),OfSmall);
const Real nu=ugkwp::ranzMarshallNuFromPr(Real(0),s.gasPrClamped),k=molecularGasConductivity(s);
const Real rate=Real(6)*nu*k/(s.rhoSolid*cpP*diameter*diameter+Real(1e-30));
const Real gasSideExchange=gpuFiniteCapacityHeatExchange(epsG*rhoG*gasCv,s.momRhoP[0]*cpP,s.couplingTgasOld[0],Real(300),rate,dt);
ck(gasSideExchange>0&&gasSideExchange<1,"material heat test must exercise sub-ULP exchange");
applyEulerianParticleMaterialHeatKernel(&s,dt);
closeEnergy(s.rhoE[0],(long double)before-(long double)gasSideExchange/(long double)epsG,"material gas decrement lost configured gas-side exchange");}
''')


def test_volume_fraction_energy_uses_unrounded_frozen_source_factor(tmp_path):
    compile_probe(tmp_path,source_fixture()+r'''
int main(){DeviceState s;setup(s);s.cellPlaneCount[0]=0;const double dt=.25,before=s.rhoE[0];
const Real epsG=Real(1)-solidEpsFromMomentDevice(s,0);s.epsGPrev[0]=epsG+Real(.0001);
const Real ceps=-((epsG-s.epsGPrev[0])/(dt+OfSmall))/epsG;
const long double expected=(long double)before+(long double)dt*(long double)ceps*((long double)before+(long double)s.p[0]);
applyGasVolumeFractionSourceKernel(&s,dt);closeEnergy(s.rhoE[0],expected,"volume-fraction energy factor narrowed",4e-7);}
''')


def test_homogeneous_double_gravity_preserves_legacy_time_arithmetic(tmp_path):
    source=source_fixture().replace('GasStateView<Real,Time,ugkwp::SstCoefficients,double>','GasStateView<Real,Time,ugkwp::SstCoefficients>').replace('s.rhoE=new double[8]{};','s.rhoE=new Real[8]{};').replace('s.rhoENext=new double[8]{};','s.rhoENext=new Real[8]{};')
    compile_probe(tmp_path,source+r'''
int main(){DeviceState s;setup(s);s.rho[0]=1;s.rhoUx[0]=Real(.3);s.gravityX=Real(.2);s.rhoE[0]=0;const double dt=.1;
const Real work=dt*(s.rhoUx[0]*s.gravityX);const Real expected=work+Real(.5)*s.rho[0]*dt*dt*sqr3(s.gravityX,s.gravityY,s.gravityZ);
applyGasGravityCell<false>(s,0,dt);ck(s.rhoE[0]==expected,"homogeneous double gravity arithmetic changed");}
''')


def test_zero_time_drag_preserves_high_kinetic_conserved_state(tmp_path):
    compile_probe(tmp_path,source_fixture()+r'''
int main(){DeviceState s;setup(s);s.rho[0]=Real(1.1);s.rhoUx[0]=Real(19999920);s.Ux[0]=s.rhoUx[0]/s.rho[0];
s.rhoE[0]=double(s.rhoUx[0])*double(s.rhoUx[0])/(2*double(s.rho[0]))+1e6;
s.momRhoUPx[0]=s.momRhoP[0]*s.Ux[0];s.momRhoEP[0]=Real(.5)*s.momRhoP[0]*s.Ux[0]*s.Ux[0];
const Real rho=s.rho[0],mx=s.rhoUx[0];const double energy=s.rhoE[0];drag<0>(&s,0.,0);
ck(s.rho[0]==rho&&s.rhoUx[0]==mx&&s.rhoE[0]==energy,"zero-time drag changed high-kinetic authority");}
''')
