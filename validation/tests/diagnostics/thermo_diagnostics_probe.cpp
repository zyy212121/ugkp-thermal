#include "gasTransport/MixtureThermo.H"
#include "gasTransport/SpeciesDiffusion.H"
#include <cassert>
#include <limits>
int main(){
 using namespace ugkwp;using namespace ugkp::diagnostics;
 DeviceContext ctx{};SpeciesThermoData<double> data[2];double coeff[6]={1000,0,0,1100,0,0};
 for(int k=0;k<2;++k){data[k].molarMass=.028;data[k].coefficientOffset=3*k;data[k].minTemperature=100;data[k].maxTemperature=3000;}
 SpeciesThermoView<double,2> view;view.species=data;view.coefficients=coeff;view.coefficientCount=6;view.diagnostics=&ctx;
 gasThermoDetail::SpeciesValues<double> values;
 assert((gasThermoDetail::evaluateSpecies(2,300.,view,values)==ThermoStatus::InvalidModel));assert(ctx.firstFatal.code==Code::InvalidIndex);
 ctx={};assert((gasThermoDetail::evaluateSpecies(0,-1.,view,values)==ThermoStatus::TemperatureOutOfRange));assert(ctx.firstFatal.code==Code::NegativeTemperature);
 ctx={};double y[2]={1.1,-.1},grad[2]={.1,-.1},D[2]={.01,.01},flux[2],h;
 assert(correctedSpeciesDiffusion(y,grad,1.,D,0.,1.,300.,view,flux,h));assert(!hasFatal(&ctx));assert(ctx.warnings[2].count==2);assert(flux[0]+flux[1]==0);
 ctx={};assert(!correctedSpeciesDiffusion(y,grad,1.,D,0.,0.,300.,view,flux,h));assert(ctx.firstFatal.code==Code::ZeroDenominator);
 ctx={};double partial[2]={.2,.7},fractions[2];assert((mixtureMassFractionsFromDensities<double,2>(partial,1.,1e-8,fractions,&ctx)));assert(fractions[0]==.2&&fractions[1]==.7);assert(ctx.warnings[0].count==1);
 ctx={};double mass[2]={.3,.7};double temperature=400;ThermoInversionControls<double> controls;
 double target=mixtureEnergy(mass,800.,view);assert(invertMixtureEnergy(mass,target,view,controls,temperature)==ThermoStatus::Success);assert(std::abs(temperature-800)<1e-6);
 ctx={};controls.maximumIterations=1;temperature=400;assert(invertMixtureEnergy(mass,target,view,controls,temperature)==ThermoStatus::Success);assert(temperature==400);assert(ctx.warnings[1].count==1);
 ctx={};temperature=400;bool evaluated=true;assert(invertMixtureEnergy(mass,std::numeric_limits<double>::quiet_NaN(),view,controls,temperature,&evaluated)==ThermoStatus::NonFiniteInput);assert(temperature==400&&!evaluated&&hasFatal(&ctx));
 struct Failed{ThermoStatus operator()(double,double&,double&)const{return ThermoStatus::InvalidModel;}};
 temperature=400;evaluated=true;assert(invertCaloricEnergy(Failed{},1.,100.,1000.,controls,temperature,&evaluated)==ThermoStatus::InvalidModel);assert(temperature==400&&!evaluated);
 ctx={};double huge[2]={std::numeric_limits<double>::max(),std::numeric_limits<double>::max()};
 assert((!mixtureMassFractionsFromDensities<double,2>(huge,1.,1e-8,fractions,&ctx)));assert(ctx.firstFatal.code==Code::NonFinite);assert(ctx.warnings[0].count==0);
 // Standalone status API reports failure without a diagnostic owner or abort.
 view.diagnostics=nullptr;assert((gasThermoDetail::evaluateSpecies(2,300.,view,values)==ThermoStatus::InvalidModel));
}
