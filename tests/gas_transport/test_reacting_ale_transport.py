"""Actual local chemistry/common ALE operator composition on disposable storage."""
import pytest
from test_mixture_state import compile_probe
from test_mixture_transport import fixture

@pytest.mark.parametrize('bits',[32,64])
@pytest.mark.parametrize('scheme',[1,2])
def test_reacting_moving_euler_uses_formation_energy_and_stage_volumes(tmp_path,bits,scheme):
    compile_probe(tmp_path,fixture()+r'''
#include "operators/advanceGasChemistryKernel.cuh"
int main(){State s;initialise(s);s.gasFluxScheme=SCHEME;s.gasSpecies.mode=ugkwp::GasMode::MixtureChemistry;
auto& g=s.gasSpecies;ugkwp::ChemistryStatus status[2];ugkwp::ChemistryAudit<Real,2> audit[2];g.chemistryStatus=status;g.chemistryAudit=audit;
Real elements[2]={1,1},basis[2]={1,-1};g.thermo.elementCount=1;g.thermo.elementComposition=elements;
ugkwp::GasReactionData<Real> reaction;reaction.reactantCount=reaction.productCount=1;reaction.highRate.preExponential=2;
ugkwp::GasStoichTerm<Real> reactant,product;reactant.species=1;reactant.coefficient=1;product.species=0;product.coefficient=1;
g.mechanism.referencePressure=101325;g.mechanism.reactions=&reaction;g.mechanism.reactionCount=1;g.mechanism.reactants=&reactant;g.mechanism.reactantTermCount=1;g.mechanism.products=&product;g.mechanism.productTermCount=1;g.mechanism.stoichiometricBasis=basis;g.mechanism.independentRank=1;
if(sizeof(Real)==4){auto& q=g.chemistryControls;q.relativeTolerance=1e-3;q.absoluteMassFractionTolerance=1e-7;q.absoluteTemperatureTolerance=.01;q.relativeConservationTolerance=1e-5;q.thermo.relativeEnergyTolerance=1e-5;q.thermo.relativeTemperatureTolerance=1e-5;q.thermo.absoluteTemperatureTolerance=.01;}
for(int c=0;c<2;++c){g.rho[c]=.4;g.rho[2+c]=.6;Real r[2]={.4,.6};s.rhoE[c]=ugkwp::mixtureEnergy(r,Real(700),g.thermo);threadIdx.x=c;recoverGasPrimitivesKernel(&s);}
Real oldV[2]={1,1},newV[2]={1.1,1.1},sweep[3]={0,.1,.1};s.riemannBoundaryKind[1]=s.riemannBoundaryKind[2]=0;
s.gasGeometry.enabled=true;s.gasGeometry.oldVolume=oldV;s.gasGeometry.newVolume=newV;s.gasGeometry.faceSweptVolume=sweep;s.gasGeometry.interval=.001;s.gasGeometry.absoluteGeometryTolerance=1e-7;s.gasGeometry.relativeGeometryTolerance=1e-6;
const Real energy=s.rhoE[0];
for(int c=0;c<2;++c){ck(bool(ugkwp::advanceGasChemistryCell(s,c,.0005,oldV[c])),"first reacting half rejected");threadIdx.x=c;recoverGasPrimitivesKernel(&s);}
transport(s,.001);
for(int c=0;c<2;++c){ck(s.gasSpecies.cellStatus[c]==0,"reacting ALE transport rejected");const Real speciesBefore=g.rho[c];ck(bool(ugkwp::advanceGasChemistryCell(s,c,.0005,newV[c])),"second reacting half rejected");threadIdx.x=c;recoverGasPrimitivesKernel(&s);
ck(std::abs(g.rho[c]-(Real(.4)+Real(.6)*(1-std::exp(Real(-.002)))))<Real(3e-6),"reacting ALE species advancement wrong");
ck(std::abs(s.rhoE[c]-energy)<Real(2e-6)*std::abs(energy),"chemical formation energy was added twice");ck(s.Tgas[c]>704&&s.Tgas[c]<706,"formation conversion did not change temperature");
ck(std::abs(audit[c].speciesMassChange[0]-(g.rho[c]-speciesBefore)*newV[c])<Real(3e-7),"second-half audit did not use endpoint volume");
}
}
'''.replace('SCHEME',str(scheme)),bits)
