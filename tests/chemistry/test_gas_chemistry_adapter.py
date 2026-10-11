"""The shared production chemistry cell adapter, on real GasStateView storage.

Serial host execution is not a substitute for a native CUDA/application run.
"""
from pathlib import Path
import shutil
import subprocess
import pytest
ROOT=Path(__file__).resolve().parents[2]
SOURCE=r'''
#include "common/operators/advanceGasChemistryKernel.cuh"
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
using namespace ugkwp;
using Real=TEST_REAL;
struct State:GasStateView<Real,double,int>{GasSpeciesState<Real,2> gasSpecies;};
struct Fixture {
 State state;
 Real rho[2]={Real(2),Real(4)},mx[2]={Real(6),Real(-4)},my[2]={Real(-2),Real(8)},mz[2]={Real(1),Real(2)};
 Real energy[2],volume[2]={Real(8),Real(16)},temperature[2]={Real(-77),Real(-88)};
 Real partial[4]={Real(1.5),Real(3),Real(.5),Real(1)};
 int cellStatus[2]={};ChemistryStatus chemistryStatus[2];ChemistryAudit<Real,2> audit[2];
 SpeciesThermoData<Real> species[2];Real coefficients[6]={Real(1000),Real(0),Real(0),Real(1000),Real(0),Real(-1e5)};
 Real elements[2]={Real(1),Real(1)},basis[2]={Real(-1),Real(1)};
 GasReactionData<Real> reaction[1];GasStoichTerm<Real> reactants[1],products[1];
 Fixture(){
  state.nCells=2;state.rho=rho;state.rhoUx=mx;state.rhoUy=my;state.rhoUz=mz;state.rhoE=energy;state.V=volume;state.Tgas=temperature;
  auto& g=state.gasSpecies;g.mode=GasMode::MixtureChemistry;g.rho=partial;g.cellStatus=cellStatus;g.chemistryStatus=chemistryStatus;g.chemistryAudit=audit;
  for(int s=0;s<2;++s){species[s].coefficientOffset=3*s;species[s].molarMass=Real(.028);species[s].minTemperature=Real(200);species[s].maxTemperature=Real(4000);species[s].referencePressure=Real(101325);}
  g.thermo.species=species;g.thermo.coefficients=coefficients;g.thermo.coefficientCount=6;g.thermo.elementCount=1;g.thermo.elementComposition=elements;
  reaction[0].reactantCount=reaction[0].productCount=1;reaction[0].highRate.preExponential=Real(2);
  reactants[0].species=0;reactants[0].coefficient=Real(1);products[0].species=1;products[0].coefficient=Real(1);
  g.mechanism.reactions=reaction;g.mechanism.reactionCount=1;g.mechanism.reactants=reactants;g.mechanism.reactantTermCount=1;g.mechanism.products=products;g.mechanism.productTermCount=1;g.mechanism.stoichiometricBasis=basis;g.mechanism.independentRank=1;g.mechanism.referencePressure=Real(101325);
  if(sizeof(Real)==4){auto& c=g.chemistryControls;c.relativeTolerance=Real(1e-3);c.absoluteMassFractionTolerance=Real(1e-7);c.absoluteTemperatureTolerance=Real(.01);c.relativeConservationTolerance=Real(1e-5);c.thermo.relativeEnergyTolerance=Real(1e-5);c.thermo.relativeTemperatureTolerance=Real(1e-5);c.thermo.absoluteTemperatureTolerance=Real(.01);}
  for(int c=0;c<2;++c){Real masses[2]={partial[c],partial[2+c]};energy[c]=mixtureEnergy(masses,Real(700),g.thermo)+(mx[c]*mx[c]+my[c]*my[c]+mz[c]*mz[c])/(Real(2)*rho[c]);audit[c].integratedTime=Real(-91);}
 }
};
int main(){
 Fixture f;auto& s=f.state;const auto& g=s.gasSpecies;
 Real bulk[10]={f.rho[0],f.rho[1],f.mx[0],f.mx[1],f.my[0],f.my[1],f.mz[0],f.mz[1],f.energy[0],f.energy[1]};
 // Explicit chemistry volume differs from state.V; density->inventory must use
 // this value consistently. Binary volumes make conversion exactly reversible.
 const Real actualVolume=Real(.5),interval=Real(.1);ClosedReactorInput<Real,2> input;
 input.speciesMass[0]=f.partial[0]*actualVolume;input.speciesMass[1]=f.partial[2]*actualVolume;input.totalMass=f.rho[0]*actualVolume;input.volume=actualVolume;
 Real ux=f.mx[0]/f.rho[0],uy=f.my[0]/f.rho[0],uz=f.mz[0]/f.rho[0];input.internalEnergy=(f.energy[0]-Real(.5)*f.rho[0]*(ux*ux+uy*uy+uz*uz))*actualVolume;
 ClosedReactorResult<Real,2> expected;ChemistryAudit<Real,2> expectedAudit;assert(advanceClosedReactor(input,interval,g.thermo,g.mechanism,g.chemistryControls,expected,expectedAudit));
 assert(advanceGasChemistryCell(s,0,double(interval),actualVolume));
 assert(f.partial[0]==expected.speciesMass[0]/actualVolume);assert(f.partial[2]==expected.speciesMass[1]/actualVolume);assert(f.partial[1]==Real(3)&&f.partial[3]==Real(1));
 assert(f.audit[0].integratedTime==interval);assert(f.audit[0].speciesMassChange[0]==expectedAudit.speciesMassChange[0]);assert(f.audit[1].integratedTime==Real(-91));
 assert(f.chemistryStatus[0] && f.cellStatus[0]==0);assert(f.temperature[0]==Real(-77));
 Real after[10]={f.rho[0],f.rho[1],f.mx[0],f.mx[1],f.my[0],f.my[1],f.mz[0],f.mz[1],f.energy[0],f.energy[1]};assert(std::memcmp(bulk,after,sizeof bulk)==0);
 // No audit allocation is required in production.
 s.gasSpecies.chemistryAudit=nullptr;assert(advanceGasChemistryCell(s,1,.1,Real(2)));assert(f.partial[1]<Real(3));
 // A failed cell does not publish any species, primitives, or accepted audit.
 s.gasSpecies.chemistryAudit=f.audit;Real savedSpecies[4];std::memcpy(savedSpecies,f.partial,sizeof savedSpecies);auto savedAudit=f.audit[0];
 assert(!advanceGasChemistryCell(s,0,.1,Real(0)));assert(std::memcmp(savedSpecies,f.partial,sizeof savedSpecies)==0);assert(std::memcmp(&savedAudit,&f.audit[0],sizeof savedAudit)==0);assert(!f.chemistryStatus[0]);assert(f.cellStatus[0]==int(GasTransportCode::ChemistryFailure));
 auto failure=f.chemistryStatus[0];assert(!advanceGasChemistryCell(s,0,.1,Real(.5)));assert(std::memcmp(&failure,&f.chemistryStatus[0],sizeof failure)==0);
 // Existing transport failures are also latched; chemistry cannot erase them.
 f.cellStatus[1]=int(GasTransportCode::NegativeInventory);auto oldStatus=f.chemistryStatus[1];assert(!advanceGasChemistryCell(s,1,.1,Real(2)));assert(f.cellStatus[1]==int(GasTransportCode::NegativeInventory));assert(std::memcmp(&oldStatus,&f.chemistryStatus[1],sizeof oldStatus)==0);
 // Disabled modes are exact no-ops, even without model or diagnostic storage.
 State disabled;disabled.gasSpecies.mode=GasMode::MixtureFrozen;assert(advanceGasChemistryCell(disabled,0,std::numeric_limits<double>::quiet_NaN(),Real(0)));disabled.gasSpecies.mode=GasMode::SingleLegacy;assert(advanceGasChemistryCell(disabled,0,1.,Real(0)));
 GasStateView<Real,double,int> legacy;assert(advanceGasChemistryCell(legacy,0,1.,Real(0)));
 // A caller may discard a multi-cell candidate even after another cell passed.
 Fixture trial;Real acceptedSpecies[4];std::memcpy(acceptedSpecies,trial.partial,sizeof acceptedSpecies);assert(advanceGasChemistryCell(trial.state,0,.1,Real(.5)));trial.partial[3]=Real(-1);assert(!advanceGasChemistryCell(trial.state,1,.1,Real(2)));assert(acceptedSpecies[0]==Real(1.5)&&acceptedSpecies[3]==Real(1));
 // Integration failure after attempted work also preserves this cell's state.
 Fixture exhausted;exhausted.state.gasSpecies.chemistryControls.maximumSteps=1;auto exhaustedAudit=exhausted.audit[0];Real exhaustedSpecies[4];std::memcpy(exhaustedSpecies,exhausted.partial,sizeof exhaustedSpecies);
 assert(!advanceGasChemistryCell(exhausted.state,0,1.,Real(.5)));assert(exhausted.chemistryStatus[0].attemptedSteps==1);assert(std::memcmp(exhaustedSpecies,exhausted.partial,sizeof exhaustedSpecies)==0);assert(std::memcmp(&exhaustedAudit,&exhausted.audit[0],sizeof exhaustedAudit)==0);
 // An interval too small to represent in the selected chemistry precision may
 // not silently turn into successful zero-time integration.
 if(sizeof(Real)==4){Fixture underflow;assert(!advanceGasChemistryCell(underflow.state,0,1e-100,Real(.5)));assert(underflow.partial[0]==Real(1.5));}
 // Missing mandatory typed status rejects before touching species.
 Fixture missing;missing.state.gasSpecies.chemistryStatus=nullptr;assert(!advanceGasChemistryCell(missing.state,0,.1,Real(.5)));assert(missing.partial[0]==Real(1.5));assert(missing.cellStatus[0]==int(GasTransportCode::ChemistryFailure));
 // Interval zero validates but must not introduce density roundoff changes.
 Fixture zero;Real zeroSnapshot[4];std::memcpy(zeroSnapshot,zero.partial,sizeof zeroSnapshot);assert(advanceGasChemistryCell(zero.state,0,0.,Real(.3)));assert(std::memcmp(zeroSnapshot,zero.partial,sizeof zeroSnapshot)==0);
}
'''
@pytest.mark.parametrize('real',['double','float'])
def test_shared_density_chemistry_adapter(tmp_path,real):
    src=tmp_path/'adapter.cpp';src.write_text(SOURCE.replace('TEST_REAL',real));exe=tmp_path/'adapter'
    build=subprocess.run(['g++','-std=c++17','-O2','-Wall','-Wextra','-Werror','-pedantic','-I',str(ROOT),str(src),'-o',str(exe)],capture_output=True,text=True)
    assert build.returncode==0,build.stderr
    run=subprocess.run([str(exe)],capture_output=True,text=True)
    assert run.returncode==0,run.stderr


def test_cuda_shared_chemistry_adapter_compiles_when_available(tmp_path):
    nvcc=shutil.which('nvcc')
    if nvcc is None:pytest.skip('nvcc unavailable: no native CUDA execution claimed')
    src=tmp_path/'adapter.cu'
    src.write_text(r'''
#include "common/operators/advanceGasChemistryKernel.cuh"
struct State:ugkwp::GasStateView<double,double,int>{ugkwp::GasSpeciesState<double,10> gasSpecies;};
void launch(State* state,const double* volume){advanceGasChemistryKernel<<<1,32>>>(state,1e-6,volume);}
''')
    build=subprocess.run([nvcc,'-std=c++17','-I',str(ROOT),'-c',str(src),'-o',str(tmp_path/'adapter.o')],capture_output=True,text=True)
    assert build.returncode==0,build.stderr
