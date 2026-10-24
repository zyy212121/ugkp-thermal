#include "materials/dry/BulkResponse.H"
#include "materials/dry/PoreTransport.H"
#include "materials/GasThermoAdapter.H"
#include "materials/Thermodynamics.H"
#include <cassert>
#include <iostream>
#include <algorithm>
using namespace chmt::dry;
// Test-only thin adapter proves the new math uses the existing shared gas
// thermochemistry unchanged. No duplicate EOS/property implementation.
struct Gas {
    chmt::SpeciesThermo species[2];
    Gas() {
        for(auto& s:species) {s.R=287;s.cp0=1000;s.Tmin=50;s.Tmax=6000;}
        species[0].e0=2e5;species[1].e0=-3e5;
    }
    double internalEnergy(std::size_t s,double T)const{return chmt::speciesE(species[s],T);}
    double heatCapacityCv(std::size_t s,double T)const{return chmt::speciesCv(species[s],T);}
    double enthalpy(std::size_t s,double T)const{return chmt::speciesH(species[s],T);}
    double gasConstant(std::size_t s)const{return species[s].R;}
};
void close(double a,double b,double tol=2e-12) {
    assert(std::abs(a-b)<=tol*std::max(1.0,std::max(std::abs(a),std::abs(b))));
}
double mass(const MaterialState<2,2>& q) {
    double m=0;for(double v:q.condensedMass)m+=v;for(double v:q.poreMass)m+=v;return m;
}
double firstOrderError(int steps) {
    MaterialState<2,2> q;q.condensedMass[0]=1;q.energy=123;
    Reaction<2,2> reaction;reaction.preExponential=2;reaction.condensedOrder[0]=1;
    reaction.production.condensed={{-1,.3}};reaction.production.gas={{.2,.5}};
    for(int n=0;n<steps;++n) {
        const auto source=reactionSourceFromConcentrations(std::vector<Reaction<2,2>>{reaction},
            std::array<double,2>{{q.condensedMass[0],0}},std::array<double,2>{},300,1);
        close(reactionEuler(q,source,1.0/steps),1);
    }
    close(mass(q),1);close(q.energy,123);
    return std::abs(q.condensedMass[0]-std::exp(-2.0));
}
int main() {
    Gas gas;
    std::array<CondensedThermo,2> solids;
    for(auto& s:solids) {s.density=1000;s.cp0=1000;s.conductivity=2;}
    solids[1].formationEnergy=-2e5;solids[1].cp1=.1;
    MaterialState<2,2> q;q.condensedMass={{.4,.1}};q.poreMass={{.001,.002}};
    q.energy=caloricState(q,solids,gas,600).energy;
    const auto recovered=recoverTemperature(q,solids,gas,300);
    assert(recovered.converged);close(recovered.temperature,600);
    const auto p=primitive(q,.001,600,solids,gas,.04);
    close(p.porosity,.5);close(p.poreDensity,6);close(p.pressure,6*287*600);
    close(p.conductivity,1.02);

    // Manufactured two-element reaction: mass/element invariants are exact,
    // but these yields/kinetics are NOT an experimentally validated material.
    Stoichiometry<2,2> nu;nu.condensed={{-1,.4}};nu.gas={{.2,.4}};
    close(massResidual(nu),0);
    const std::array<std::array<double,2>,2> elementSolid{{{{.6,.4}},{{1,0}}}};
    const std::array<std::array<double,2>,2> elementGas{{{{1,0}},{{0,1}}}};
    const auto elements=elementResidual(nu,elementSolid,elementGas);
    close(elements[0],0);close(elements[1],0);
    const double initialMass=mass(q),initialEnergy=q.energy;
    applyExtent(q,nu,.1);
    close(mass(q),initialMass);close(q.energy,initialEnergy);
    const auto reacted=recoverTemperature(q,solids,gas,600);
    assert(reacted.converged);assert(std::abs(reacted.temperature-600)>1);
    close(caloricState(q,solids,gas,reacted.temperature).energy,initialEnergy);

    // Shift every species formation energy by elemental reference constants.
    // The temperature from conserved energy must not change.
    auto shiftedSolid=solids;auto shiftedGas=gas;auto shiftedState=q;
    const std::array<double,2> shift{{8e5,-1.2e6}};
    double totalShift=0;
    for(std::size_t c=0;c<2;++c) {
        const double de=elementSolid[c][0]*shift[0]+elementSolid[c][1]*shift[1];
        shiftedSolid[c].formationEnergy+=de;totalShift+=q.condensedMass[c]*de;
    }
    for(std::size_t s=0;s<2;++s) {
        const double de=elementGas[s][0]*shift[0]+elementGas[s][1]*shift[1];
        shiftedGas.species[s].e0+=de;totalShift+=q.poreMass[s]*de;
    }
    shiftedState.energy+=totalShift;
    const auto shifted=recoverTemperature(shiftedState,shiftedSolid,shiftedGas,500);
    assert(shifted.converged);close(shifted.temperature,reacted.temperature);

    MaterialState<2,2> small;small.condensedMass={{.01,.0}};small.energy=9;
    MaterialState<2,2> rate;rate.condensedMass={{-1,.4}};rate.poreMass={{.2,.4}};
    close(reactionEuler(small,rate,1),.01);close(small.condensedMass[0],0);
    close(mass(small),.01);close(small.energy,9);
    // Preserve the actual legacy mass-concentration kinetic parameterization.
    // Compare against BOTH the independent formula and existing CHMT evaluator
    // with unchanged A, b, Ea, orders and mass-yield coefficients.
    chmt::Reaction oldReaction;
    oldReaction.A=7.3;oldReaction.temperaturePower=.35;oldReaction.activationEnergy=22000;
    oldReaction.order[0]=.7;oldReaction.order[1]=1.2;
    oldReaction.condensedNu[0]=-1;oldReaction.condensedNu[1]=.4;
    oldReaction.gasNu[0]=.2;oldReaction.gasNu[1]=.4;
    Reaction<2,2> preserved;
    preserved.preExponential=oldReaction.A;preserved.temperaturePower=oldReaction.temperaturePower;
    preserved.activationEnergy=oldReaction.activationEnergy;
    preserved.condensedOrder={{oldReaction.order[0],oldReaction.order[1]}};
    preserved.production.condensed={{-1,.4}};preserved.production.gas={{.2,.4}};
    for(double T: {350.,800.,1300.}) for(double V: {.001,.01,.1}) {
        const double independent=V*7.3*std::pow(T,.35)*std::exp(-22000/(8.31446261815324*T))
            *std::pow(q.condensedMass[0]/V,.7)*std::pow(q.condensedMass[1]/V,1.2);
        const double oldRate=chmt::reactionRate(oldReaction,q.condensedMass.data(),V,T);
        const auto preservedSource=reactionSource(q,std::vector<Reaction<2,2>>{preserved},
            V,solids,T);
        close(oldRate,independent);close(preservedSource.condensedMass[0],-oldRate);
        close(preservedSource.condensedMass[1],.4*oldRate);
        close(preservedSource.poreMass[0],.2*oldRate);close(preservedSource.poreMass[1],.4*oldRate);
    }
    // Explicit optional molar-concentration input is equivalent ONLY with
    // A_molar=A_mass*product(W_i^order_i), same extent/production basis.
    const std::array<double,2> W{{.025,.12}},rho{{3,7}},molar{{3/.025,7/.12}};
    auto molarReaction=preserved;
    molarReaction.preExponential*=std::pow(W[0],.7)*std::pow(W[1],1.2);
    close(reactionRate(preserved,rho,std::array<double,2>{},800),
        reactionRate(molarReaction,molar,std::array<double,2>{},800));
    // Independently configured reaction stages retain distinct gas yields.
    Reaction<2,2> stage1,stage2;
    stage1.production.condensed={{-.1,.05}};stage1.production.gas={{.05,0}};
    stage2.production.condensed={{0,-.1}};stage2.production.gas={{0,.1}};
    stage1.preExponential=2;stage2.preExponential=3;
    stage1.condensedOrder[0]=1;stage2.condensedOrder[1]=1;
    const auto multi=reactionSource(q,std::vector<Reaction<2,2>>{stage1,stage2},
        .001,solids,600);
    close(multi.poreMass[0],q.condensedMass[0]*2*.05);
    close(multi.poreMass[1],q.condensedMass[1]*3*.1);
    close(mass(multi),0);
    MaterialState<3,3> many;many.condensedMass={{1,2,3}};
    Stoichiometry<3,3> manyNu;manyNu.condensed={{-.5,-.25,-.25}};
    manyNu.gas={{.1,.2,.7}};applyExtent(many,manyNu,.1);
    close(many.condensedMass[2],2.975);close(many.poreMass[2],.07);
    const auto frozen=reactionSourceFromConcentrations(std::vector<Reaction<2,2>>{},
        std::array<double,2>{},std::array<double,2>{},300,1);
    close(mass(frozen),0);
    const double e100=firstOrderError(100),e200=firstOrderError(200);
    assert(e100/e200>1.98&&e100/e200<2.03);

    close(darcyVelocity(1e-12,1e-5,-1e6,2),.1);
    close(darcyVelocity(1e-12,1e-5,2*9.81,2,9.81),0);
    PoreFaceState<2> left,right;
    left.massFraction={{.8,.2}};right.massFraction={{.3,.7}};
    left.density=2;right.density=1;left.temperature=500;right.temperature=700;
    const auto f=poreFaceFlux(left,right,.1,.01,std::array<double,2>{{.02,.04}},.5,gas);
    close(f.diffusion[0]+f.diffusion[1],0);
    close(f.species[0]+f.species[1],.02);
    close(f.energy,f.conductionEnergy+f.speciesEnergy[0]+f.speciesEnergy[1]);
    const auto reversed=poreFaceFlux(right,left,.1,-.01,std::array<double,2>{{.02,.04}},.5,gas);
    for(std::size_t s=0;s<2;++s)close(f.species[s],-reversed.species[s]);
    close(f.energy,-reversed.energy);
    auto a=q,b=q;const auto before=a;
    exchangePoreFlux(a,b,f,.2,1e-3);
    for(std::size_t s=0;s<2;++s)close(a.poreMass[s]+b.poreMass[s],2*before.poreMass[s]);
    close(a.energy+b.energy,2*before.energy);
    // Single-species diffusion cancels exactly; three species don't require a
    // second component placeholder or fixed two-species matrix.
    close(correctedDiffusion(1,std::array<double,1>{{1}},
        std::array<double,1>{{1}},std::array<double,1>{{1}})[0],0);
    const auto j3=correctedDiffusion(2,std::array<double,3>{{.2,.3,.5}},
        std::array<double,3>{{-.1,.3,-.2}},std::array<double,3>{{.1,.2,.3}});
    close(j3[0]+j3[1]+j3[2],0);
    std::cout<<"bulk: calorics, reaction mass/elements/energy, reference invariance, "
        "Euler refinement, bounded extent, frozen chemistry, Darcy, corrected diffusion, "
        "oriented face energy and closed-pair conservation PASS; Euler ratio="<<e100/e200<<'\n';
}
