// Checkpoint 2 only: real FP32 bulk with FP64 tables, geometry and energy.
#include "common/operators/advanceGasChemistryKernel.cuh"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <limits>
#include <type_traits>
using namespace ugkwp;
struct State:GasStateView<float,double,int,double,double,double> {
    GasSpeciesState<float,2,double> gasSpecies;
};
int main() {
    State state;
    ugkp::diagnostics::DeviceContext diagnostics{};
    float rho[1]={2},mx[1]={6},my[1]={-2},mz[1]={1},temperature[1]={700};
    float partial[2]={1.5f,.5f};
    double energy[1],volume[1]={8.000000000000002};
    int cellStatus[1]={};ChemistryStatus chemistryStatus[1];
    SpeciesThermoData<double> table[2];
    double coefficients[6]={1000,0,0,1000,0,-1e5};
    double elements[2]={1,1},basis[2]={-1,1};
    GasReactionData<double> reaction[1];
    GasStoichTerm<double> reactants[1],products[1];
    state.diagnostics=&diagnostics;state.nCells=1;
    state.rho=rho;state.rhoUx=mx;state.rhoUy=my;state.rhoUz=mz;
    state.rhoE=energy;state.V=volume;state.Tgas=temperature;
    auto& species=state.gasSpecies;species.mode=GasMode::MixtureChemistry;
    species.rho=partial;species.cellStatus=cellStatus;species.chemistryStatus=chemistryStatus;
    for(int j=0;j<2;++j) {
        table[j].coefficientOffset=3*j;table[j].molarMass=.028;
        table[j].minTemperature=200;table[j].maxTemperature=4000;
        table[j].referencePressure=101325;
    }
    species.thermo.species=table;species.thermo.coefficients=coefficients;
    species.thermo.coefficientCount=6;species.thermo.elementCount=1;
    species.thermo.elementComposition=elements;
    reaction[0].reactantCount=reaction[0].productCount=1;
    reaction[0].highRate.preExponential=2;
    reactants[0].species=0;reactants[0].coefficient=1;
    products[0].species=1;products[0].coefficient=1;
    species.mechanism.reactions=reaction;species.mechanism.reactionCount=1;
    species.mechanism.reactants=reactants;species.mechanism.reactantTermCount=1;
    species.mechanism.products=products;species.mechanism.productTermCount=1;
    species.mechanism.stoichiometricBasis=basis;species.mechanism.independentRank=1;
    species.mechanism.referencePressure=101325;
    static_assert(std::is_same<GasEnergyScalar<State>,double>::value);
    static_assert(std::is_same<GasChemistryVolumeReal<State>,double>::value);
    static_assert(std::is_same<decltype(speciesCp(0,700.f,species.thermo)),float>::value);
    static_assert(std::is_same<decltype(mixtureGasConstant(partial,species.thermo)),float>::value);
    static_assert(std::is_same<decltype(molecularViscosity<2>(state.gasViscosity,1.8e-5f,700.f,partial)),float>::value);
    const float cp=speciesCp(0,700.f,species.thermo);
    const float gasConstant=mixtureGasConstant(partial,species.thermo);
    assert(std::isfinite(cp)&&cp>gasConstant&&gasConstant>0);
    const double initialSpecies[2]={partial[0],partial[1]};
    const double kinetic=.5*(double(mx[0])*mx[0]+double(my[0])*my[0]+double(mz[0])*mz[0])/rho[0];
    energy[0]=mixtureEnergy(initialSpecies,700.,species.thermo)+kinetic;
    const double oldEnergy=energy[0];
    const double actualVolume=.5000000000000001;
    bool evaluated=false;
    const auto status=advanceGasChemistryCell(state,0,.1,actualVolume,false,&evaluated);
    assert(evaluated&&status&&!ugkp::diagnostics::hasFatal(&diagnostics));
    assert(std::isfinite(partial[0])&&std::isfinite(partial[1]));
    assert(partial[0]>0&&partial[0]<initialSpecies[0]);
    assert(partial[1]>initialSpecies[1]);
    const double massResidual=double(partial[0])+partial[1]-rho[0];
    assert(std::abs(massResidual)<=64*std::numeric_limits<float>::epsilon()*rho[0]);
    // Chemistry never publishes rhoE or cached temperature. Recovery consumes
    // the unchanged double energy with the newly quantized species densities.
    assert(energy[0]==oldEnergy&&temperature[0]==700.f&&rho[0]==2.f);
    double acceptedSpecies[2]={partial[0],partial[1]},recoveredTemperature=700;
    bool recovered=false;
    const auto thermoStatus=invertMixtureEnergy(acceptedSpecies,energy[0]-kinetic,
        species.thermo,species.thermoControls,recoveredTemperature,&recovered);
    assert(recovered&&thermoStatus==ThermoStatus::Success);
    assert(std::isfinite(recoveredTemperature)&&recoveredTemperature>700&&recoveredTemperature<4000);
    const double recoveredEnergy=mixtureEnergy(acceptedSpecies,recoveredTemperature,species.thermo)+kinetic;
    const double energyRelative=std::abs(recoveredEnergy-oldEnergy)/std::abs(oldEnergy);
    assert(energyRelative<=64*std::numeric_limits<float>::epsilon());
    std::printf("mixed chemistry: mass residual %.17g, relative energy residual %.17g, T %.17g, steps %d\n",
        massResidual,energyRelative,recoveredTemperature,status.attemptedSteps);
}
