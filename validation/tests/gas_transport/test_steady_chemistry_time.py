"""Production chemistry launch body on CPU; adapters retain physical-time API."""
import importlib.util
import subprocess
import pytest
from test_mixture_state import ROOT, compile_probe


def chemistry_fixture():
    path=ROOT/'validation/tests/chemistry/test_gas_chemistry_adapter.py'
    spec=importlib.util.spec_from_file_location('chemical_adapter_fixture',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    fixture=module.SOURCE.split('int main(){',1)[0].replace('TEST_REAL','double')
    source=(ROOT/'common/operators/advanceGasChemistryKernel.cuh').read_text()
    start=source.index('template<class GasState, class Time>\n__global__ void advanceGasChemistryKernel')
    # Only expose the CUDA-conditional production launch body to the serial
    # emulator; the body itself and real closed-reactor adapter are unchanged.
    kernel=source[start:source.rfind('#endif')]
    return fixture+r'''
#define __global__
#define __host__
#define __device__
struct Index{int x=0;}blockIdx,threadIdx,blockDim;
struct LocalState:State{double*gasPseudoDt=nullptr;double gasPseudoReferenceDt=2;};
'''+kernel


def compile_chemistry(tmp_path,body):
    src=tmp_path/'chem.cpp';src.write_text(chemistry_fixture()+body)
    exe=tmp_path/'chem'
    result=subprocess.run(['g++','-std=c++17','-O0','-I'+str(ROOT),'-I'+str(ROOT/'common'),str(src),'-o',str(exe)],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    run=subprocess.run([str(exe)],capture_output=True,text=True)
    assert run.returncode==0,run.stdout+run.stderr


def test_kernel_scales_strang_half_step_exactly_once_and_keeps_direct_adapter(tmp_path):
    compile_chemistry(tmp_path,r'''
int main(){Fixture actual,expected,direct;LocalState s,d;static_cast<State&>(s)=actual.state;static_cast<State&>(d)=direct.state;
double localDt[2]={.01,.04},volumes[2]={.5,2};s.gasPseudoDt=d.gasPseudoDt=localDt;blockDim.x=1;
for(int c=0;c<2;++c){threadIdx.x=c;advanceGasChemistryKernel(&s,.5,volumes);advanceGasChemistryCellWithVolumes(expected.state,c,.5*localDt[c]/2,volumes);
assert(actual.partial[c]==expected.partial[c]&&actual.partial[2+c]==expected.partial[2+c]);assert(actual.chemistryStatus[c]&&actual.cellStatus[c]==0);
// Direct adapter receives an actual interval and must not apply localDt again.
advanceGasChemistryCellWithVolumes(d,c,.5*localDt[c]/2,volumes);assert(direct.partial[c]==expected.partial[c]&&direct.partial[2+c]==expected.partial[2+c]);}
assert(actual.partial[0]!=1.5&&actual.partial[1]!=3);assert(actual.energy[0]==expected.energy[0]&&actual.energy[1]==expected.energy[1]);}
''')


def test_kernel_zero_interval_and_default_layout_are_exact(tmp_path):
    compile_chemistry(tmp_path,r'''
int main(){Fixture zero,legacy,nullable,expected;LocalState z,n;static_cast<State&>(z)=zero.state;static_cast<State&>(n)=nullable.state;double localDt[2]={.01,.04},volumes[2]={.5,2};z.gasPseudoDt=localDt;double saved[4];std::memcpy(saved,zero.partial,sizeof(saved));blockDim.x=1;
for(int c=0;c<2;++c){threadIdx.x=c;advanceGasChemistryKernel(&z,0.,volumes);advanceGasChemistryKernel(&legacy.state,.01,volumes);advanceGasChemistryKernel(&n,.01,volumes);advanceGasChemistryCellWithVolumes(expected.state,c,.01,volumes);}
assert(std::memcmp(saved,zero.partial,sizeof(saved))==0);assert(std::memcmp(legacy.partial,expected.partial,sizeof(saved))==0);assert(std::memcmp(nullable.partial,expected.partial,sizeof(saved))==0);}
''')


def test_uniform_chemical_residual_is_nonzero_and_cancels_transport(tmp_path):
    header=str(ROOT/'applications/gasUGKPSteady/private_backend/SteadyChemistryResidual.cuh')
    compile_chemistry(tmp_path,'#include "'+header+'"\n'+r'''
int main(){Fixture f;f.temperature[0]=700;f.temperature[1]=700;double before[2]={f.energy[0],f.energy[1]},r[2]={};
assert(steadyAddChemistryResidual(f.state,0,r));assert(std::abs(r[0]+3)<1e-12&&std::abs(r[1]-3)<1e-12);
double balanced[2]={3,-3};assert(steadyAddChemistryResidual(f.state,0,balanced));assert(std::abs(balanced[0])<1e-12&&std::abs(balanced[1])<1e-12);
assert(f.energy[0]==before[0]&&f.energy[1]==before[1]);assert(f.partial[0]==1.5&&f.partial[2]==.5);
f.state.gasSpecies.mode=GasMode::MixtureFrozen;double unchanged[2]={2,-2};assert(steadyAddChemistryResidual(f.state,0,unchanged));assert(unchanged[0]==2&&unchanged[1]==-2);}
''')


def test_raw_kernel_reports_homogeneous_reaction_and_transport_source_balance(tmp_path):
    from test_steady_gas_residual import kernel_fixture
    compile_probe(tmp_path,kernel_fixture()+r'''
int main(){DeviceState s;initialise(s);double residual[18]={};s.gasSteadyResidual=residual;s.V[0]=.5;s.V[1]=2;
ugkwp::SpeciesThermoData<Real> species[2];Real coefficients[6]={1000,0,-1e7,1000,0,-1.01e7};
Real partial[4]={1.5,1.5,.5,.5},flux[6]={},elements[2]={1,1},basis[2]={-1,1};
ugkwp::GasReactionData<Real> reaction[1];ugkwp::GasStoichTerm<Real> reactant[1],product[1];
auto&g=s.gasSpecies;g.mode=ugkwp::GasMode::MixtureChemistry;g.rho=partial;g.flux=flux;
for(int k=0;k<2;++k){species[k].coefficientOffset=3*k;species[k].molarMass=.028;species[k].minTemperature=200;species[k].maxTemperature=4000;species[k].referencePressure=101325;}
g.thermo.species=species;g.thermo.coefficients=coefficients;g.thermo.coefficientCount=6;g.thermo.elementCount=1;g.thermo.elementComposition=elements;
reaction[0].reactantCount=reaction[0].productCount=1;reaction[0].highRate.preExponential=2;reactant[0].species=0;reactant[0].coefficient=1;product[0].species=1;product[0].coefficient=1;
g.mechanism.reactions=reaction;g.mechanism.reactionCount=1;g.mechanism.reactants=reactant;g.mechanism.reactantTermCount=1;g.mechanism.products=product;g.mechanism.productTermCount=1;g.mechanism.stoichiometricBasis=basis;g.mechanism.independentRank=1;g.mechanism.referencePressure=101325;
double oldE[2];
for(int c=0;c<2;++c){Real rhoY[2]={1.5,.5};s.Tgas[c]=700;s.rhoE[c]=ugkwp::mixtureEnergy(rhoY,Real(700),g.thermo);oldE[c]=s.rhoE[c];threadIdx.x=c;steadyGasSpatialResidualKernel(&s);
near(residual[10+c],3,"uniform reacting source omitted");near(residual[12+c],3,"uniform product source omitted");near(residual[8+c],0,"formation-inclusive chemistry added heat RHS");}
// Boundary influx of reactant and outflux of product cancels the volumetric
// reaction source in both unequal-volume cells, before absolute-value reduction.
flux[1]=-3*s.V[0];flux[4]=3*s.V[0];flux[2]=-3*s.V[1];flux[5]=3*s.V[1];
for(int c=0;c<2;++c){threadIdx.x=c;steadyGasSpatialResidualKernel(&s);near(residual[10+c],0,"transport/reaction A cancellation");near(residual[12+c],0,"transport/reaction B cancellation");ck(s.rhoE[c]==oldE[c]&&s.rhoE[c]<0,"raw chemistry changed formation energy");}
}
''')
