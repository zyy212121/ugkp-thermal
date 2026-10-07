#include "tests/TestSupport.H"
#include "materials/MaterialCaloric.H"
#include "materials/ReactionStepControl.H"
#include "ablation/InterfaceMath.H"
#include <type_traits>
#include <cfenv>
using namespace chmt;
using namespace chmt_test;

static SpeciesThermo nasa() {
    SpeciesThermo s;
    s.model=ugkwp::SpeciesThermoModel::NASA7;
    s.R=300;s.Tmin=200;s.Tmid=1000;s.Tmax=3000;
    // Synthetic polynomial: cp/R=3.5+0.001*T below 1000 K, 4+0.0005*T above.
    // Different integration constants make both h branches continuous.
    const Real low[7]={3.5,.001,0,0,0,-10000,2};
    const Real high[7]={4,.0005,0,0,0,-10250,2};
    for(int j=0;j<7;++j){s.nasa7[j]=low[j];s.nasa7[j+7]=high[j];}
    return s;
}
static Real expectedE(Real T) {
    return T<=1000 ? 300*(-10000+2.5*T+.0005*T*T)
        : 300*(-10250+3*T+.00025*T*T);
}
static Real expectedCv(Real T) {return 300*(T<=1000 ? 2.5+.001*T : 3+.0005*T);}
int main() {
    static_assert(std::is_trivially_copyable<SpeciesThermo>::value,"checkpoint thermo stays POD");
    // OpenFOAM enables traps for invalid arithmetic. Unconfigured inactive
    // species and rejected tiny R values must not divide by zero or overflow.
    for(Real R:{Real(0),Real(-1),std::numeric_limits<Real>::denorm_min(),
        std::numeric_limits<Real>::infinity(),std::numeric_limits<Real>::quiet_NaN()}) {
        SpeciesThermo invalid;invalid.R=R;
        std::feclearexcept(FE_ALL_EXCEPT);
        const GasThermoAdapter<1> adapter(&invalid);
        check(std::fetestexcept(FE_DIVBYZERO|FE_INVALID|FE_OVERFLOW)==0,"invalid R adapter has no floating exceptions");
        check(adapter.data[0].molarMass==0,"invalid R creates invalid metadata");
        check(ugkwp::validateSpeciesThermoView(adapter.view())==ugkwp::ThermoStatus::InvalidModel,"active invalid R rejects");
    }
    auto p=physics();
    SolidQ condensedOnly;condensedOnly.condensed[0]=1;condensedOnly.porosity=.1;
    condensedOnly.energy=condensedE(p.condensed[0],600);
    for(auto& species:p.species)species=SpeciesThermo{};
    std::feclearexcept(FE_ALL_EXCEPT);Real condensedTemperature=0;
    check(recoverSolid(condensedOnly,p,condensedTemperature),"unconfigured zero-pore gas remains inactive");
    check(std::fetestexcept(FE_DIVBYZERO|FE_INVALID|FE_OVERFLOW)==0,"zero-pore recovery has no floating exceptions");
    near(condensedTemperature,600,"zero-pore condensed temperature");
    SolidQ dryRight=condensedOnly;dryRight.energy=condensedE(p.condensed[0],700);
    MaterialPrimitive dryLeftState,dryRightState;
    check(recoverMaterial(condensedOnly,.002,p,dryLeftState)
        &&recoverMaterial(dryRight,.002,p,dryRightState),"dry conduction material states");
    SolidQ dryFlux;std::feclearexcept(FE_ALL_EXCEPT);
    check(darcyConductionFlux(condensedOnly,dryRight,dryLeftState,dryRightState,
        .002,.002,1,.1,Vec3{1,0,0},p,dryFlux),"dry zero-pore conduction needs no gas thermo");
    near(dryFlux.energy,-1000,"dry conduction carries condensed heat only");
    for(Real massRate:dryFlux.pore)check(massRate==0,"dry pore transport is zero");
    check(std::fetestexcept(FE_DIVBYZERO|FE_INVALID|FE_OVERFLOW)==0,"dry conduction has no floating exceptions");
    SolidQ invalidDonor=condensedOnly;invalidDonor.pore[0]=.1;
    p.permeability=1;p.poreViscosity=1;dryFlux.energy=123;
    check(!darcyConductionFluxWithGradient(invalidDonor,dryRight,dryLeftState,dryRightState,
        .002,.002,1,.1,Vec3{1,0,0},-1,p,dryFlux),"active Darcy transport rejects missing gas thermo");
    check(dryFlux.energy==123,"failed Darcy flux preserves output");
    p=physics();
    for(int s=0;s<Ns;++s)p.species[s]=nasa();
    check(validSpecies(p.species[0]),"NASA7 accepted without LinearCp intercept rule");
    for(Real T:{200.,750.,1000.,1000.00001,1500.,3000.}) {
        near(speciesE(p.species[0],T),expectedE(T),"NASA gas energy");
        near(speciesH(p.species[0],T),expectedE(T)+300*T,"NASA gas enthalpy");
        near(speciesCv(p.species[0],T),expectedCv(T),"NASA gas cv");
        near(speciesCp(p.species[0],T),expectedCv(T)+300,"NASA gas cp");
        GasPrimitive interface;Real Y[Ns]{};Y[0]=1;
        check(makeInterfacePrimitive(Y,T,101325,Vec3{},Vec3{1,0,0},0,0,p,interface),"NASA interface primitive");
        near(interface.soundSpeed,::sqrt((1+300/expectedCv(T))*300*T),"NASA interface capacity dispatch");
        GasPrimitive w;w.rho=1.2;w.temperature=T;w.velocity={7,-3,1};w.Y[0]=.4;w.Y[1]=.6;
        for(Real volume:{1e-12,1.,1e12}) {
            const auto q=conservativeGas(w,volume,p);GasPrimitive recovered;
            check(recoverGas(q,volume,p,recovered),"NASA gas recovery");
            near(recovered.temperature,T,"NASA gas temperature",1e-10,1e-7);
            near(recovered.soundSpeed,::sqrt((1+300/expectedCv(T))*300*T),"NASA sound speed");
        }
    }
    auto bad=p.species[0];bad.nasa7[0]=.2;bad.nasa7[1]=0;
    check(!validSpecies(bad),"nonpositive NASA cv rejects");
    check(!chmt::finite(speciesE(p.species[0],199)),"NASA extrapolation rejects");
    SolidQ solid;solid.condensed[0]=.8;solid.condensed[1]=.1;solid.pore[0]=.2;solid.porosity=.2;
    for(Real T:{200.,700.,1000.,1400.,2000.}) {
        solid.energy=.8*condensedE(p.condensed[0],T)+.1*condensedE(p.condensed[1],T)+.2*expectedE(T);
        Real recovered=999;check(recoverSolid(solid,p,recovered),"mixed condensed NASA recovery");
        near(recovered,T,"mixed temperature",1e-10,1e-7);
        MaterialCaloric caloric;std::string error;check(materialCaloric(solid,p,caloric,error),"NASA material caloric object");
        near(caloric.energy(T),solid.energy,"material energy dispatch");
        near(caloric.capacity(T),.8*(p.condensed[0].cp0+p.condensed[0].cp1*T)+.1*(p.condensed[1].cp0+p.condensed[1].cp1*T)+.2*expectedCv(T),"material capacity dispatch");
        near(caloric.secant(750,1500),(caloric.energy(1500)-caloric.energy(750))/750,"material secant across NASA break");
        near(caloric.secant(T,T),caloric.capacity(T),"coincident secant is capacity");
        ReactionStepControl control;check(reactionStepControl(solid,.001,p,.4,control),"NASA material reaction controller");
        near(control.capacity,caloric.capacity(T),"reaction controller NASA capacity");
    }
    // Keep source branch discontinuities instead of silently smoothing NASA7.
    p.species[0].nasa7[12]+=100;
    MaterialCaloric discontinuous;std::string discontinuousError;
    check(materialCaloric(solid,p,discontinuous,discontinuousError),"discontinuous NASA caloric");
    for(Real a:{750.,1000.})near(discontinuous.secant(a,1500),
        (discontinuous.energy(1500)-discontinuous.energy(a))/(1500-a),"NASA breakpoint energy jump retained");
    p.species[0].nasa7[12]-=100;
    // A formation-energy offset must not ruin the small-step heat-capacity secant.
    p.species[0].nasa7[5]=p.species[0].nasa7[12]=1e20;
    MaterialCaloric c;std::string error;check(materialCaloric(solid,p,c,error),"large offset caloric");
    near(c.secant(500,500+1e-8),c.capacity(500+5e-9),"offset-free material secant",1e-10,1e-7);
    // Legacy single-active-species states allow absent inactive tables.
    p=physics();p.species[1]=SpeciesThermo{};
    GasPrimitive w;w.rho=1;w.temperature=600;w.Y[0]=1;
    GasPrimitive recovered;check(recoverGas(conservativeGas(w,1,p),1,p,recovered),"inactive legacy thermo preserved");
    near(recovered.temperature,600,"legacy recovered temperature");
    w.temperature=2100;check(!chmt::finite(conservativeGas(w,1,p).energy),"out-of-range state rejects");
    auto failedGas=conservativeGas(recovered,1,p);failedGas.species[0]=-1;recovered.temperature=987;
    check(!recoverGas(failedGas,1,p,recovered)&&recovered.temperature==987,"failed gas recovery is transactional");
    solid.pore[0]=-1;Real unchanged=456;
    check(!recoverSolid(solid,p,unchanged)&&unchanged==456,"failed mixed recovery is transactional");
    // Borrowed addresses are regenerated after a value copy.
    p.species[0].hasEntropyReference=true;p.species[0].referenceTemperature=300;p.species[0].referenceEntropy=400;
    GasThermoAdapter<1> original(&p.species[0]);const auto copied=original;
    check(copied.view().species!=original.view().species,"adapter copy owns metadata");
    check(copied.view().coefficients!=original.view().coefficients,"adapter copy owns coefficients");
    near(ugkwp::speciesEntropyStandard(0,300.,copied.view()),400,"explicit entropy metadata preserved");
    // A gas energy intercept cannot substitute for NASA formation enthalpy.
    p.species[0]=nasa();Reaction reaction;reaction.gasNu[0]=1;
    check(!chmt::finite(reactionFormationEnergy(reaction,p)),"legacy formation diagnostic rejects NASA");
    std::cout<<"shared CHMT thermo: NASA gas/material recovery, common capacity, stable secants and legacy behavior passed\n";
}
