// Native transient FV regression for unequal-D, three-species zero support.
// No flooring, limiter, rejected step or fallback is used in this fixture.
void zeroSupportCase(Time& runTime,fvMesh& mesh,scalar dt,scalar end,std::ofstream& out){
 Gas gas;using State=dry::MaterialState<1,3>;std::vector<State> q(mesh.nCells());
 std::array<dry::CondensedThermo,1> solid{};solid[0].density=1000;solid[0].cp0=1000;
 std::array<double,3> initial{};double initialEnergy=0,minimum=1e30,temperatureError=0,maxSum=0,firstFlux=0,oldCentered=0;
 forAll(mesh.C(),i){double V=mesh.V()[i];std::array<double,3> y=mesh.C()[i].x()<.5?std::array<double,3>{{0,.1,.9}}:std::array<double,3>{{.01,.9,.09}};q[i].condensedMass[0]=500*V;for(int s=0;s<3;++s){q[i].poreMass[s]=.5*V*y[s];initial[s]+=q[i].poreMass[s];}q[i].energy=dry::caloricState(q[i],solid,gas,300).energy;initialEnergy+=q[i].energy;}
 surfaceScalarField j0(IOobject("zeroJ0",runTime.timeName(),mesh),mesh,dimensionedScalar("zero",dimMass/dimTime,0));surfaceScalarField j1(IOobject("zeroJ1",runTime.timeName(),mesh),mesh,dimensionedScalar("zero",dimMass/dimTime,0));surfaceScalarField j2(IOobject("zeroJ2",runTime.timeName(),mesh),mesh,dimensionedScalar("zero",dimMass/dimTime,0));surfaceScalarField je(IOobject("zeroJE",runTime.timeName(),mesh),mesh,dimensionedScalar("zero",dimPower,0));
 auto state=[&](label i){dry::PoreFaceState<3> p;double m=0;for(double mass:q[i].poreMass)m+=mass;p.density=m/(.5*mesh.V()[i]);p.temperature=dry::recoverTemperature(q[i],solid,gas,300).temperature;for(int s=0;s<3;++s)p.massFraction[s]=q[i].poreMass[s]/m;return p;};
 runTime.setDeltaT(dt);label steps=label(std::lround(end/dt));
 for(label step=0;step<steps;++step){++runTime;forAll(mesh.neighbour(),face){label l=mesh.owner()[face],r=mesh.neighbour()[face];double d=1/mesh.deltaCoeffs()[face],area=mesh.magSf()[face];auto a=state(l),b=state(r);auto flux=dry::poreFaceFlux(a,b,d,0,{{.001,1,.001}},0,gas);j0[face]=area*flux.species[0];j1[face]=area*flux.species[1];j2[face]=area*flux.species[2];je[face]=area*flux.energy;maxSum=max(maxSum,mag(flux.species[0]+flux.species[1]+flux.species[2]));
  if(step==0&&a.massFraction[0]==0&&b.massFraction[0]>0){firstFlux=flux.species[0];std::array<double,3> mean{},gradient{};for(int s=0;s<3;++s){mean[s]=.5*(a.massFraction[s]+b.massFraction[s]);gradient[s]=(b.massFraction[s]-a.massFraction[s])/d;}oldCentered=dry::correctedDiffusion(1,mean,gradient,{{.001,1,.001}})[0];}
 }
 auto d0=fvc::div(j0),d1=fvc::div(j1),d2=fvc::div(j2),de=fvc::div(je);
 forAll(mesh.C(),i){double w=dt*mesh.V()[i];q[i].poreMass[0]-=w*d0()[i];q[i].poreMass[1]-=w*d1()[i];q[i].poreMass[2]-=w*d2()[i];q[i].energy-=w*de()[i];for(double mass:q[i].poreMass)minimum=min(minimum,mass);temperatureError=max(temperatureError,mag(dry::recoverTemperature(q[i],solid,gas,300).temperature-300));}
 }
 std::array<double,3> final{};double energy=0;for(auto& cell:q){for(int s=0;s<3;++s)final[s]+=cell.poreMass[s];energy+=cell.energy;}double speciesResidual=0;for(int s=0;s<3;++s)speciesResidual=max(speciesResidual,mag(final[s]-initial[s]));
 out<<"mode,N,dt,steps,initial_new_flux,old_centered_counterexample,min_inventory,species_residual,energy_relative,zero_sum_flux,temperature_error\nzeroSupport,"<<mesh.nCells()<<','<<dt<<','<<steps<<','<<firstFlux<<','<<oldCentered<<','<<minimum<<','<<speciesResidual<<','<<mag(energy-initialEnergy)/mag(initialEnergy)<<','<<maxSum<<','<<temperatureError<<'\n';
}
