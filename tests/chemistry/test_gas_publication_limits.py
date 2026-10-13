"""Publication uses original finite values if optional repair/audits cannot finish."""
from pathlib import Path
import subprocess
import pytest
ROOT=Path(__file__).resolve().parents[2]
SOURCE=r'''
#include "common/operators/advanceGasChemistryKernel.cuh"
#include <cassert>
#include <cmath>
#include <cstring>
#include <string>
using namespace ugkwp;
template<class R,int N>struct State:GasStateView<R,double,int,double>{GasSpeciesState<R,N> gasSpecies;};
int main(int argc,char**argv){
 assert(argc==2);std::string mode=argv[1];
 if(mode=="repair" || mode=="audit"){
  State<double,3> s;double rho=0x1.e9a02fc7a3981p+5,zero=0,energy=0;
  double partial[3]={0x1.d26bf8769ec2dp+0,0x1.db0b39192641bp+5,0x1.96eac8605681fp-11};
  double volume=0x1.f991bc5586445p-5;
  SpeciesThermoData<double> sp[3];double elements[6]={};double coefficients[9]={1000,0,0,1000,0,0,1000,0,0};
  for(int i=0;i<3;++i){sp[i].coefficientOffset=i*3;sp[i].molarMass=.028;sp[i].minTemperature=200;sp[i].maxTemperature=4000;sp[i].referencePressure=101325;}
  auto& g=s.gasSpecies;g.thermo.species=sp;g.thermo.coefficients=coefficients;g.thermo.coefficientCount=9;g.mode=GasMode::MixtureChemistry;g.mechanism.referencePressure=101325;g.rho=partial;g.densityClosureTolerance=0;
  if(mode=="audit"){rho=1;volume=1;g.thermo.elementCount=2;g.thermo.elementComposition=elements;for(int i=0;i<3;++i){partial[i]=std::numeric_limits<double>::max()/2;sp[i].molarMass=1;coefficients[3*i]=universalGasConstant<double>();elements[2*i]=std::numeric_limits<double>::max();}}
  ChemistryStatus status;ChemistryAudit<double,3> audit;audit.integratedTime=-777;int cell=0;g.chemistryStatus=&status;g.chemistryAudit=&audit;g.cellStatus=&cell;
  s.nCells=1;s.rho=&rho;s.rhoUx=s.rhoUy=s.rhoUz=&zero;s.rhoE=&energy;
  energy=mixtureEnergy(partial,700.,g.thermo);
  double original[3];for(int i=0;i<3;++i)original[i]=(partial[i]*volume)/volume;
  assert(advanceGasChemistryCell(s,0,1e-6,volume));assert(status&&cell==0&&audit.integratedTime==1e-6);
  for(int i=0;i<3;++i)assert(partial[i]==original[i]);
  if(mode=="repair"){
   // The attempted largest-carrier correction crosses the target by one ULP.
   double independent=original[0]+original[2],proposed=rho-independent;
   assert(proposed!=original[1]);assert(original[0]+proposed+original[2]!=rho);
   assert(original[0]+original[1]+original[2]!=rho);
  }else {assert(!gasFinite(audit.massResidual));assert(!gasFinite(audit.maximumElementResidual));for(double x:partial)assert(gasFinite(x));}
 }else{
  State<double,2> s;double rho=1,zero=0,partial[2]={1,0};double energy;
  SpeciesThermoData<double> sp[2];double coefficients[6]={1000,0,0,1000,0,0},basis[2]={0,1};
  for(int i=0;i<2;++i){sp[i].coefficientOffset=3*i;sp[i].molarMass=.028;sp[i].minTemperature=200;sp[i].maxTemperature=4000;sp[i].referencePressure=101325;}
  auto& g=s.gasSpecies;g.thermo.species=sp;g.thermo.coefficients=coefficients;g.thermo.coefficientCount=6;g.rho=partial;g.mode=GasMode::MixtureChemistry;g.mechanism.referencePressure=101325;
  GasReactionData<double> reaction;GasStoichTerm<double> product{1,1};reaction.productCount=1;reaction.highRate.preExponential=1e38;
  g.mechanism.reactions=&reaction;g.mechanism.reactionCount=1;g.mechanism.products=&product;g.mechanism.productTermCount=1;g.mechanism.stoichiometricBasis=basis;g.mechanism.independentRank=1;
  ChemistryStatus status;ChemistryAudit<double,2> audit;audit.integratedTime=-777;int cell=0;g.chemistryStatus=&status;g.chemistryAudit=&audit;g.cellStatus=&cell;
  s.nCells=1;s.rho=&rho;s.rhoUx=s.rhoUy=s.rhoUz=&zero;s.rhoE=&energy;double mass[2]={1,0};energy=mixtureEnergy(mass,700.,g.thermo);
  ClosedReactorInput<double,2> input;input.totalMass=1;input.volume=1;input.speciesMass[0]=1;input.internalEnergy=energy;
  ClosedReactorResult<double,2> output;ChemistryAudit<double,2> unrounded;assert(advanceClosedReactor(input,1000.,g.thermo,g.mechanism,castChemistryControls<double>(g.chemistryControls),output,unrounded));assert(output.speciesMass[1]>1e38&&std::isfinite(output.speciesMass[1]));
  assert(advanceGasChemistryCell(s,0,1000.,1.));assert(status&&cell==0&&audit.integratedTime==1000.);assert(partial[0]==1&&partial[1]==output.speciesMass[1]);
 }
}
'''
@pytest.fixture(scope='module')
def runner(tmp_path_factory):
 p=tmp_path_factory.mktemp('publication-limits');src=p/'probe.cpp';exe=p/'probe';src.write_text(SOURCE)
 b=subprocess.run(['g++','-std=c++17','-O2','-ffp-contract=off','-Wall','-Wextra','-Werror','-I',str(ROOT),str(src),'-o',str(exe)],capture_output=True,text=True)
 assert b.returncode==0,b.stderr
 return exe
@pytest.mark.parametrize('scenario',['repair','audit','large_finite'])
def test_finite_publication_and_large_double_inventory(runner,scenario):
 r=subprocess.run([str(runner),scenario],capture_output=True,text=True,timeout=20)
 assert r.returncode==0,r.stderr
