"""Finite chemistry outputs survive overflow of their diagnostic density sum."""
import subprocess
from validation.tools.paths import SOURCE_ROOT

SOURCE = r'''

#include "common/chemistry/GasRates.H"
#include "common/chemistry/GasJacobian.H"

#include <cassert>
#include <cmath>
#include <limits>
using namespace ugkwp;
struct Model {
 SpeciesThermoData<double> sp[2]; double coef[6]={1000,0,0,1000,0,0};
 double elements[2]={1,1}, basis[2]={-1,1};
 GasReactionData<double> rx[2]; GasStoichTerm<double> re[2],pr[2];
 GasColliderEfficiency<double> eff[1];
 SpeciesThermoView<double,2> t; GasMechanismView<double,2> m;
 Model(){for(int s=0;s<2;++s){sp[s].coefficientOffset=3*s;sp[s].molarMass=.028;sp[s].minTemperature=200;sp[s].maxTemperature=4000;sp[s].referencePressure=101325;sp[s].referenceTemperature=300;sp[s].referenceEntropy=1500;sp[s].hasEntropyReference=true;}
 t.species=sp;t.coefficients=coef;t.coefficientCount=6;t.elementCount=1;t.elementComposition=elements;t.speciesOrderHash=1;
 for(int r=0;r<2;++r){rx[r].reactantOffset=r;rx[r].reactantCount=1;rx[r].productOffset=r;rx[r].productCount=1;rx[r].sourceIndex=r;rx[r].highRate.preExponential=2;re[r].species=0;re[r].coefficient=1;pr[r].species=1;pr[r].coefficient=1;}
 m.reactions=rx;m.reactionCount=1;m.reactants=re;m.reactantTermCount=2;m.products=pr;m.productTermCount=2;m.referencePressure=101325;m.speciesOrderHash=1;m.stoichiometricBasis=basis;m.independentRank=1;
 }
};
bool near(double a,double b,double r=1e-10){return std::abs(a-b)<1e-12+r*std::abs(b);}


#include "common/chemistry/LocalStiffIntegrator.H"

#include "common/operators/advanceGasChemistryKernel.cuh"
#include <cstdio>
int main(){
 Model model; ugkp::diagnostics::DeviceContext d{};
 struct State:GasStateView<double,double,int>{GasSpeciesState<double,2> gasSpecies;} s;
 double rho=1e308,u=0,energy=1e308,partial[2]={1e308,1e308};int coarse=0;
 s.nCells=1;s.diagnostics=&d;s.rho=&rho;s.rhoUx=s.rhoUy=s.rhoUz=&u;s.rhoE=&energy;
 s.gasSpecies.mode=GasMode::MixtureChemistry;s.gasSpecies.rho=partial;s.gasSpecies.cellStatus=&coarse;s.gasSpecies.thermo=model.t;
 bool done=false;auto status=advanceGasChemistryCell(s,0,0.,1e-300,false,&done);
 printf("done=%d fatal=%u site=%u observed=%g status=%d\n",done,d.fatalClaim,d.firstFatal.sourceSiteId,d.firstFatal.observed,int(status.code));
 if(d.fatalClaim || !done)return 1;
 if(d.warnings[0].count==0)return 2;
 if(partial[0]!=1e308 || partial[1]!=1e308)return 3;
 return 0;
}
'''

def test_chemistry_diagnostic_sum_overflow_is_warning(tmp_path):
    source=tmp_path/'chemistry.cpp'
    source.write_text(SOURCE)
    binary=tmp_path/'chemistry'
    subprocess.run(['g++','-std=c++17','-I'+str(SOURCE_ROOT),str(source),'-o',str(binary)],check=True)
    result=subprocess.run([str(binary)],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr


def test_chemistry_invalid_input_provenance(tmp_path):
    body = SOURCE[:SOURCE.index("int main(){")] + r'''int main(){
 Model model; ugkp::diagnostics::DeviceContext d{};
 struct State:GasStateView<double,double,int>{GasSpeciesState<double,2> gasSpecies;} s;
 double rho=1,u=0,energy=1e6,partial[2]={1,0};int coarse=0;
 s.nCells=1;s.diagnostics=&d;s.rho=&rho;s.rhoUx=s.rhoUy=s.rhoUz=&u;s.rhoE=&energy;
 s.gasSpecies.mode=GasMode::MixtureChemistry;s.gasSpecies.rho=partial;s.gasSpecies.cellStatus=&coarse;s.gasSpecies.thermo=model.t;
 bool done=false;
 advanceGasChemistryCell(s,0,std::numeric_limits<double>::quiet_NaN(),1.,false,&done);
 if(!d.fatalClaim || !std::isnan(d.firstFatal.observed))return 1;
 d={};u=std::numeric_limits<double>::infinity();
 advanceGasChemistryCell(s,0,0.,1.,false,&done);
 if(!d.fatalClaim || !std::isinf(d.firstFatal.observed) || d.firstFatal.component!=0)return 2;
 return 0;
}'''
    source=tmp_path/'provenance.cpp';source.write_text(body);binary=tmp_path/'probe'
    subprocess.run(['g++','-std=c++17','-I'+str(SOURCE_ROOT),str(source),'-o',str(binary)],check=True)
    subprocess.run([str(binary)],check=True)
