void pyrolysisCase(Time& runTime,fvMesh& mesh,volScalarField& T,scalar dt,scalar end,std::ofstream& out,bool adiabatic) {
 Gas gas;std::array<dry::CondensedThermo,1> solid{};solid[0].density=1000;solid[0].cp0=1000;solid[0].formationEnergy=400000;
 using State=dry::MaterialState<1,2>;std::vector<State> q(mesh.nCells());
 std::vector<dry::Reaction<1,2>> reactions(1);reactions[0].preExponential=0.7;reactions[0].condensedOrder[0]=1;reactions[0].production.condensed[0]=-1;reactions[0].production.gas={{0.6,0.4}};
 double initialE=0,initialMass=0,reservoir=0;
 forAll(T,i){q[i].condensedMass[0]=500*mesh.V()[i];q[i].poreMass={{0.3*mesh.V()[i],0.2*mesh.V()[i]}};T[i]=300;q[i].energy=dry::caloricState(q[i],solid,gas,300).energy;initialE+=q[i].energy;initialMass+=500.5*mesh.V()[i];}
 runTime.setDeltaT(dt);label steps=label(std::lround(end/dt));
 for(label step=0;step<steps;++step){++runTime;forAll(T,i){auto source=dry::reactionSource(q[i],reactions,mesh.V()[i],solid,T[i]);dry::reactionEuler(q[i],source,dt);if(adiabatic)T[i]=dry::recoverTemperature(q[i],solid,gas,T[i]).temperature;else {double updated=dry::caloricState(q[i],solid,gas,300).energy;reservoir+=updated-q[i].energy;q[i].energy=updated;}}}
 double m=0,E=0,parent=0,element0=0,element1=0,tempErr=0;
 const double exactParent=500*std::exp(-.7*end),converted=500-exactParent;
 // Independent closed-form caloric solution, not recoverTemperature reused.
 const double exactT=(350185000-exactParent*400000-(0.5+converted)*160000)/(exactParent*1000+(0.5+converted)*700);
 forAll(T,i){m+=q[i].condensedMass[0]+q[i].poreMass[0]+q[i].poreMass[1];E+=q[i].energy;parent+=q[i].condensedMass[0];element0+=.6*q[i].condensedMass[0]+q[i].poreMass[0];element1+=.4*q[i].condensedMass[0]+q[i].poreMass[1];tempErr=max(tempErr,mag(T[i]-(adiabatic?exactT:300)));}
 out<<"mode,N,dt,parent_error,temperature_error,mass_residual,element_residual,energy_residual,temperature\n"<<(adiabatic?"adiabatic":"isothermal")<<','<<mesh.nCells()<<','<<dt<<','<<mag(parent-exactParent)<<','<<tempErr<<','<<m-initialMass<<','<<max(mag(element0-300.3),mag(element1-200.2))<<','<<E-initialE-reservoir<<','<<T[0]<<'\n';T.write();
}
struct SurfaceGas {
 double enthalpy(size_t s,double T)const{return (s==1?-1000000:200000)+(s==1?1100:1000)*T;}
};
void surfaceCase(Time& runTime,fvMesh& mesh,volScalarField& T,std::ofstream& out) {
 SurfaceGas gas;std::array<dry::CondensedThermo,1> solid{};solid[0].density=2000;solid[0].cp0=700;solid[0].formationEnergy=100000;
 std::array<dry::SurfaceMassAction<1,2>,1> rx{};rx[0].prefactor=100;rx[0].activationEnergy=6000*8.31446261815324;rx[0].gasOrder[0]=1;rx[0].stoichiometry.condensed={{-.012}};rx[0].stoichiometry.gas={{-.016,.028}};
 const label right=mesh.boundaryMesh().findPatchID("right");double lo=300,hi=1600,Tw=0,qsolid=0,residual=0;
 dry::DryInterfaceFlux<1,2> flux;dry::DryEnergyBalance balance;dry::PoreSurfaceFlux<2> pore;
 for(label iter=0;iter<60;++iter){Tw=.5*(lo+hi);T.boundaryFieldRef()[right]==Tw;solve(fvm::laplacian(dimensionedScalar("k",dimPower/dimLength/dimTemperature,20),T));
  qsolid=-20*fvc::snGrad(T)().boundaryField()[right][0];auto production=dry::surfaceProduction(rx,Tw,{{1500}},{{.2,0}});flux=dry::congruentDryInterfaceFlux(production,pore,solid,.25);
  dry::DryEnergyInput in;in.wallTemperature=Tw;in.gasPressure=100000;in.gasConductiveFlux=-100*(1500-Tw);in.solidConductiveFlux=qsolid;
  balance=dry::dryEnergyBalance(flux,pore,solid,gas,in);residual=balance.residual;if(residual>0)hi=Tw;else lo=Tw;
 }
 // Independent analytic thermal resistance + mass-action and ALE work.
 auto analytical=[&](double w){double r=20*std::exp(-6000/w);double m=.012*r;double recession=m/1500;double hgO=200000+1000*w,hgCO=-1000000+1100*w,es=100000+700*w;return r*(-.016*hgO+.028*hgCO)-100000*recession-100*(1500-w)-m*es+20*(w-300);};
 lo=300;hi=1600;for(label i=0;i<80;++i){double mid=.5*(lo+hi);if(analytical(mid)>0)hi=mid;else lo=mid;}double exactT=.5*(lo+hi),r=20*std::exp(-6000/exactT);
 // A 1-second constant-rate front: physical-area extensive ledgers independently integrated.
 double mSolid=flux.condensedConsumption[0],carbon=mSolid-flux.gasSpecies[1]*12/28,oxygen=flux.gasSpecies[0]+flux.gasSpecies[1]*16/28;
 out<<"mode,N,wall_T,wall_T_error,rate_error,recession,recession_error,mass_residual,element_residual,energy_residual,analytic_residual\n"<<"surface,"<<mesh.nCells()<<','<<Tw<<','<<mag(Tw-exactT)<<','<<mag(mSolid-.012*r)<<','<<flux.recessionSpeed<<','<<mag(flux.recessionSpeed-.012*r/1500)<<','<<flux.gasMass-mSolid<<','<<max(mag(carbon),mag(oxygen))<<','<<residual<<','<<analytical(Tw)<<'\n';T.write();
}
