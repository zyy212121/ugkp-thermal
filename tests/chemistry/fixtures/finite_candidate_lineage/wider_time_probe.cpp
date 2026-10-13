#include "probe_hooks.hpp"
#include "common/operators/advanceGasChemistryKernel.cuh"
#include <cstring>
using namespace ugkwp;
struct State:GasStateView<double,double,int>{GasSpeciesState<double,2> gasSpecies;};
int main(int argc,char**argv){
 State state;double density=1,momentum=0,energy=0,partial[2]={1,0};int cell=0;ChemistryStatus status;ChemistryAudit<double,2> audit;
 SpeciesThermoData<double> sp[2];double coefficients[6]={1000,0,0,1000,0,0},elements[2]={1,1},basis[2]={-1,1};GasReactionData<double> rx[1];GasStoichTerm<double> reactants[1]={{0,1}},products[1]={{1,1}};
 for(int s=0;s<2;++s){sp[s].coefficientOffset=3*s;sp[s].molarMass=1;sp[s].minTemperature=200;sp[s].maxTemperature=4000;sp[s].referencePressure=101325;}
 state.nCells=1;state.rho=&density;state.rhoUx=state.rhoUy=state.rhoUz=&momentum;state.rhoE=&energy;auto& g=state.gasSpecies;
 g.mode=GasMode::MixtureChemistry;g.rho=partial;g.cellStatus=&cell;g.chemistryStatus=&status;g.chemistryAudit=&audit;g.thermo.species=sp;g.thermo.coefficients=coefficients;g.thermo.coefficientCount=6;g.thermo.elementCount=1;g.thermo.elementComposition=elements;
 rx[0].reactantCount=rx[0].productCount=1;rx[0].highRate.preExponential=1e30;g.mechanism.reactions=rx;g.mechanism.reactionCount=1;g.mechanism.reactants=reactants;g.mechanism.products=products;g.mechanism.reactantTermCount=g.mechanism.productTermCount=1;g.mechanism.stoichiometricBasis=basis;g.mechanism.independentRank=1;g.mechanism.referencePressure=101325;
 auto& c=g.chemistryControls;c.relativeTolerance=1e-3;c.absoluteMassFractionTolerance=1e-7;c.absoluteTemperatureTolerance=.01;c.relativeConservationTolerance=1e-5;c.thermo.relativeEnergyTolerance=1e-5;c.thermo.absoluteTemperatureTolerance=.01;c.thermo.relativeTemperatureTolerance=1e-5;
 energy=mixtureEnergy(partial,700.,g.thermo);independent::reset();const double interval=argc>1?std::strtod(argv[1],nullptr):1e-50;
 auto st=advanceGasChemistryCell(state,0,interval,1.);std::cout<<std::setprecision(21)<<"success="<<bool(st)<<" code="<<int(st.code)<<" interval="<<interval<<" stored_interval="<<double(interval)<<" species="<<partial[0]<<","<<partial[1]<<" audit_storage_time="<<audit.integratedTime<<"\n";independent::dump();
 bool ok=bool(st)&&partial[1]>0&&std::abs(double(partial[1])/(interval*double(rx[0].highRate.preExponential))-1)<2e-5&&independent::counters.rates>0&&independent::counters.prepare==1;
 if(!independent::steps.empty())ok=ok&&independent::steps[0].h==interval;
 std::cout<<"RESULT "<<(ok?"PASS":"FAIL")<<"\n";return ok?0:1;
}
