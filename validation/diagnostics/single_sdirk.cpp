
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
// The official result is one two-stage SDIRK step, not two half-steps.
int main()
{
    Model model;
    ClosedReactorInput<double,2> input;
    input.speciesMass[0]=.028;
    input.totalMass=.028;
    input.volume=.001;
    input.internalEnergy=mixtureEnergy(input.speciesMass,700.,model.t);
    ChemistryControls<double> controls;
    ClosedReactorResult<double,2> result;
    bool available=false;
    auto status=advanceClosedReactor(input,.25,model.t,model.m,controls,result,&available);
    assert(available);
    assert(status.attemptedSteps==1 && status.rejectedSteps==0);
    const double gamma=1-1/std::sqrt(2.);
    const double z=-.5;
    const double expected=.028*(1+(1-2*gamma)*z)/((1-gamma*z)*(1-gamma*z));
    assert(near(result.speciesMass[0],expected,1e-10));
    assert(near(result.speciesMass[0]+result.speciesMass[1],.028));

    // Legacy adaptive controls cannot change the official requested interval.
    controls.maximumSteps=0;
    controls.maximumRejectedSteps=0;
    controls.minimumStep=100;
    controls.initialStep=1e-20;
    ClosedReactorResult<double,2> second;
    status=advanceClosedReactor(input,.25,model.t,model.m,controls,second,&available);
    assert(available && status.attemptedSteps==1);
    assert(result.speciesMass[0]==second.speciesMass[0]);

    // Fatal input must not be reported as an evaluated numerical result.
    input.internalEnergy=std::numeric_limits<double>::quiet_NaN();
    status=advanceClosedReactor(input,.25,model.t,model.m,controls,result,&available);
    assert(!status && !available);
}
