"""Real shared boundary operators using CHMT marshaling and the host CUDA shim.

This checks host numerical behavior, not native GPU execution.
"""
from pathlib import Path
import runpy
import subprocess

APP = Path(__file__).resolve().parents[1]
HELPER = runpy.run_path(str(APP / 'tests/test_shared_backend_compile.py'))
FIXTURE = runpy.run_path(str(APP / 'tests/test_coupled_runtime_transactions.py'))['FIXTURE']


def run_boundary_probe(tmp_path, body):
    source = HELPER['prepare_backend'](tmp_path)
    source.write_text(source.read_text() + FIXTURE + r'''
#include <cstdio>
void require(bool ok,const char* message){if(!ok){std::fprintf(stderr,"%s\n",message);std::abort();}}
int main(){for(bool single:{false,true}){
 Fixture f;
 if(single){assert(bindSingleGasModel(f.gas,"A",f.model,f.error));f.gas.mode=ugkwp::GasMode::SingleLegacy;}
 f.model.physics.gasViscosity=.02;
 f.h.gas[0].momentum={7,5,-3};
 f.h.gas[0].energy=(f.model.physics.species[0].cp0-f.model.physics.species[0].R)*600+41.5;
 auto& m=f.h.gasMesh;m.boundaryKind.assign(2,BoundaryKind::Slip);
 m.thermalBoundary.assign(2,ThermalBoundaryKind::ZeroGradient);
 m.thermalBoundary[0]=ThermalBoundaryKind::FixedValue;m.boundaryPrimitive[0].temperature=300;
''' + body + '\n}}\n')
    binary = tmp_path / 'boundary'
    subprocess.run(HELPER['compiler_flags'](tmp_path) + [str(source), str(APP / 'mesh/Geometry.C'), '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


def test_slip_fixed_temperature_conducts_without_mass_or_shear(tmp_path):
    run_boundary_probe(tmp_path, r'''
 SharedGasDeviceStorage storage;assert(storage.configure(f.model,f.gas,{},f.h,f.error));auto& v=storage.hostView();
 v.gasFluxScheme=1;v.gasReconstruction=0;blockDim.x=1;threadIdx.x=0;
 transport::recoverGasPrimitivesKernel(&v);
 transport::updateLegacyGasBoundaryMirrorKernel(&v,0);transport::updateRiemannBoundaryMirrorKernel(&v);
 // Nonzero gradients ensure a no-slip mapping or accidental shear term is detected.
 v.gradUxX[0]=2;v.gradUyX[0]=3;v.gradUzX[0]=-4;
 double mass,mx,my,mz,energy;
 require(transport::computeRiemannGasFaceFluxDevice<false>(v,0,mass,mx,my,mz,energy),"slip face rejected");
 std::fprintf(stderr,"single=%d slip heat=%g W\n",single,energy);
 require(std::abs(energy-300)<1e-9,"fixed-temperature slip lost the 300 W conductive flux");
 require(mass==0&&std::abs(mx+v.p[0])<1e-9&&my==0&&mz==0,"slip gained penetration or viscous traction");
 // The unchanged adiabatic slip face still has exactly zero energy flux.
 require(transport::computeRiemannGasFaceFluxDevice<false>(v,1,mass,mx,my,mz,energy),"adiabatic face rejected");
 require(mass==0&&energy==0&&my==0&&mz==0,"adiabatic slip changed");
 ''')


def test_slip_fixed_temperature_gradient_keeps_tangential_velocity(tmp_path):
    run_boundary_probe(tmp_path, r'''
 SharedGasDeviceStorage storage;assert(storage.configure(f.model,f.gas,{},f.h,f.error));auto& v=storage.hostView();
 blockDim.x=1;threadIdx.x=0;transport::recoverGasPrimitivesKernel(&v);
 const auto wall=transport::riemannFacePrimitiveForGradient(v,0,0);
 require(wall.T==300,"slip gradient ignored fixed temperature");
 require(wall.ux==0&&wall.uy==5&&wall.uz==-3,"slip gradient lost tangential motion");
 require(std::abs(wall.rho-2)<1e-12&&wall.p==v.p[0],"fixed-temperature slip EOS is inconsistent");
 const auto adiabatic=transport::riemannFacePrimitiveForGradient(v,0,1);
 require(std::abs(adiabatic.T-600)<1e-10&&adiabatic.rho==1,"adiabatic slip primitive changed");
 ''')


def test_outlet_pressure_survives_both_refreshes_and_changes_actual_flux(tmp_path):
    run_boundary_probe(tmp_path, r'''
 m.boundaryKind[0]=BoundaryKind::Outlet;m.thermalBoundary[0]=ThermalBoundaryKind::ZeroGradient;
 const double outletPressure=90000;m.boundaryPrimitive[0].pressure=outletPressure;
 SharedGasDeviceStorage storage;assert(storage.configure(f.model,f.gas,{},f.h,f.error));auto& v=storage.hostView();
 v.gasFluxScheme=1;v.gasReconstruction=0;blockDim.x=1;threadIdx.x=0;
 transport::recoverGasPrimitivesKernel(&v);
 transport::updateLegacyGasBoundaryMirrorKernel(&v,0);transport::updateRiemannBoundaryMirrorKernel(&v);
 std::fprintf(stderr,"single=%d outlet mirror=%g Pa\n",single,v.riemannBoundaryP[0]);
 require(v.gasBoundaryP[0]==outletPressure&&v.riemannBoundaryP[0]==outletPressure,"outlet pressure overwritten by owner refresh");
 const auto owner=transport::gasCellPrimitive(v,0);const auto outlet=transport::riemannBoundaryState(v,0,owner);
 require(outlet.p==outletPressure&&outlet.T==owner.T,"outlet primitive did not preserve prescribed pressure and owner temperature");
 require(outlet.ux==owner.ux&&outlet.uy==owner.uy&&outlet.uz==owner.uz,"outlet velocity was fixed");
 double mass,mx,my,mz,energy,ownerMass,ownerMx,ownerMy,ownerMz,ownerEnergy;
 require(transport::computeRiemannGasFaceFluxDevice<false>(v,0,mass,mx,my,mz,energy),"outlet flux rejected");
 v.riemannBoundaryPFix[0]=0;v.riemannBoundaryP[0]=owner.p;
 require(transport::computeRiemannGasFaceFluxDevice<false>(v,0,ownerMass,ownerMx,ownerMy,ownerMz,ownerEnergy),"owner-pressure flux rejected");
 require(std::abs(mass-ownerMass)>1e-6&&std::abs(energy-ownerEnergy)>1e-6,"outlet pressure never reached the production Riemann flux");
 ''')
