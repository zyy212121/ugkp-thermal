// Standalone material FV validation only: no gas-flow solver or production adapter.
void coupledCase(Time& runTime,fvMesh& mesh,volScalarField& T,scalar dt,scalar end,std::ofstream& out) {
 using State=dry::MaterialState<1,2>;
 Gas gas;
 std::array<dry::CondensedThermo,1> solid{};
 solid[0].density=1000;solid[0].cp0=1000;solid[0].formationEnergy=400000;solid[0].conductivity=10000;
 std::vector<dry::Reaction<1,2>> network(1);
 network[0].preExponential=0.4*std::exp(2000.0/300);
 network[0].activationEnergy=2000*8.31446261815324;
 network[0].condensedOrder[0]=1;network[0].production.condensed[0]=-1;
 network[0].production.gas={{0.6,0.4}};
 const scalar K=1e-10,mu=1e-5,D=0.003;
 std::vector<State> q(mesh.nCells());
 std::vector<dry::MaterialPrimitive> prim(mesh.nCells());
 forAll(T,i){double V=mesh.V()[i],x=mesh.C()[i].x();T[i]=300;
  q[i].condensedMass[0]=500*V;q[i].poreMass={{0.5*V*(0.6+0.1*std::sin(2*constant::mathematical::pi*x)),0.5*V*(0.4-0.1*std::sin(2*constant::mathematical::pi*x))}};
  q[i].energy=dry::caloricState(q[i],solid,gas,T[i]).energy;
 }
 T.correctBoundaryConditions();
 auto totals=[&](){std::array<double,4> v{};for(auto& s:q){v[0]+=s.condensedMass[0]+s.poreMass[0]+s.poreMass[1];v[1]+=s.energy;v[2]+=0.6*s.condensedMass[0]+s.poreMass[0];v[3]+=0.4*s.condensedMass[0]+s.poreMass[1];}return v;};
 const auto initial=totals();std::array<double,4> boundary{};double throughMass=0,throughEnergy=0,maxDiffSum=0,minMass=1e30,maxMassResidual=0,maxEnergyResidual=0;
 dimensionSet capDim=dimEnergy/dimVolume/dimTemperature;
 volScalarField C(IOobject("C",runTime.timeName(),mesh),mesh,dimensionedScalar("C",capDim,1));
 volScalarField rhs(IOobject("rhs",runTime.timeName(),mesh),mesh,dimensionedScalar("rhs",dimEnergy/dimVolume/dimTime,0));
 surfaceScalarField heatK(IOobject("heatK",runTime.timeName(),mesh),mesh,dimensionedScalar("k",dimPower/dimLength/dimTemperature,5000));
 surfaceScalarField j0(IOobject("j0",runTime.timeName(),mesh),mesh,dimensionedScalar("j0",dimMass/dimTime,0));
 surfaceScalarField j1(IOobject("j1",runTime.timeName(),mesh),mesh,dimensionedScalar("j1",dimMass/dimTime,0));
 surfaceScalarField je(IOobject("je",runTime.timeName(),mesh),mesh,dimensionedScalar("je",dimPower,0));
 std::ofstream history((runTime.path()/"history.csv").c_str());history<<std::setprecision(17)<<"t,mass,energy,mass_residual,energy_residual,Tmin,Tmax,pmin,pmax\n";
 runTime.setDeltaT(dt);label steps=label(std::lround(end/dt));
 for(label step=0;step<steps;++step){
  ++runTime;
  // Conservative local chemistry at fixed extensive formation-inclusive E.
  forAll(T,i){auto source=dry::reactionSource(q[i],network,mesh.V()[i],solid,T[i]);dry::reactionEuler(q[i],source,dt);
   T[i]=dry::recoverTemperature(q[i],solid,gas,T[i]).temperature;
   prim[i]=dry::primitive(q[i],mesh.V()[i],T[i],solid,gas,0);
  }
  T.correctBoundaryConditions();
  auto faceState=[&](label i){dry::PoreFaceState<2> s;double m=q[i].poreMass[0]+q[i].poreMass[1];s.density=prim[i].poreDensity;s.temperature=T[i];s.massFraction={{q[i].poreMass[0]/m,q[i].poreMass[1]/m}};return s;};
  forAll(mesh.neighbour(),f){label l=mesh.owner()[f],r=mesh.neighbour()[f];double d=1/mesh.deltaCoeffs()[f],area=mesh.magSf()[f];
   auto flux=dry::poreFaceFlux(faceState(l),faceState(r),d,dry::darcyVelocity(K,mu,(prim[r].pressure-prim[l].pressure)/d,0),{{D,D}},0,gas);
   j0[f]=area*flux.species[0];j1[f]=area*flux.species[1];je[f]=area*flux.energy;heatK[f]=0.5*(prim[l].conductivity+prim[r].conductivity);
   maxDiffSum=max(maxDiffSum,mag(flux.diffusion[0]+flux.diffusion[1]));
  }
  forAll(mesh.boundary(),p)if(!isA<emptyFvPatch>(mesh.boundary()[p])){
   const labelUList& cells=mesh.boundary()[p].faceCells();
   forAll(cells,j){label c=cells[j];double d=1/mesh.deltaCoeffs().boundaryField()[p][j],area=mesh.magSf().boundaryField()[p][j],Tb=T.boundaryField()[p][j];
    dry::PoreFaceState<2> reservoir;reservoir.temperature=Tb;reservoir.density=90000/(300*Tb);reservoir.massFraction={{0.6,0.4}};
    auto flux=dry::poreFaceFlux(faceState(c),reservoir,d,dry::darcyVelocity(K,mu,(90000-prim[c].pressure)/d,0),{{D,D}},0,gas);
    j0.boundaryFieldRef()[p][j]=area*flux.species[0];j1.boundaryFieldRef()[p][j]=area*flux.species[1];je.boundaryFieldRef()[p][j]=area*flux.energy;heatK.boundaryFieldRef()[p][j]=prim[c].conductivity;
    double dm=dt*area*(flux.species[0]+flux.species[1]),de=dt*area*flux.energy;
    boundary[0]+=dm;boundary[1]+=de;boundary[2]+=dt*area*flux.species[0];boundary[3]+=dt*area*flux.species[1];throughMass+=mag(dm);throughEnergy+=mag(de);
   }
  }
  tmp<volScalarField> div0=fvc::div(j0),div1=fvc::div(j1),dive=fvc::div(je);
  forAll(T,i){double V=mesh.V()[i];q[i].poreMass[0]-=dt*V*div0()[i];q[i].poreMass[1]-=dt*V*div1()[i];q[i].energy-=dt*V*dive()[i];
   T[i]=dry::recoverTemperature(q[i],solid,gas,T[i]).temperature;
   C[i]=dry::caloricState(q[i],solid,gas,T[i]).capacity/V;rhs[i]=C[i]*T[i]/dt;
   minMass=min(minMass,min(q[i].poreMass[0],q[i].poreMass[1]));
  }
  scalarField before(T.primitiveField());
  // Linear cp: exact caloric storage and implicit native OF conduction.
  solve(fvm::Sp(C/dimensionedScalar("dt",dimTime,dt),T)-fvm::laplacian(heatK,T)==rhs);
  forAll(T,i)q[i].energy+=C[i]*mesh.V()[i]*(T[i]-before[i]);
  surfaceScalarField qh(-heatK*fvc::snGrad(T)*mesh.magSf());
  forAll(mesh.boundary(),p)if(!isA<emptyFvPatch>(mesh.boundary()[p])){
   double de=dt*gSum(qh.boundaryField()[p]);boundary[1]+=de;throughEnergy+=mag(de);
  }
  auto now=totals();double mr=now[0]-initial[0]+boundary[0],er=now[1]-initial[1]+boundary[1];maxMassResidual=max(maxMassResidual,mag(mr));maxEnergyResidual=max(maxEnergyResidual,mag(er));
  if(step%max(label(1),steps/20)==0 || step==steps-1){double pmin=1e30,pmax=0;forAll(T,i){auto p=dry::primitive(q[i],mesh.V()[i],T[i],solid,gas,0);pmin=min(pmin,p.pressure);pmax=max(pmax,p.pressure);}history<<runTime.value()<<','<<now[0]<<','<<now[1]<<','<<mr<<','<<er<<','<<gMin(T.primitiveField())<<','<<gMax(T.primitiveField())<<','<<pmin<<','<<pmax<<'\n';}
 }
 auto final=totals();double elementResidual=0;for(label e=2;e<4;++e)elementResidual=max(elementResidual,mag(final[e]-initial[e]+boundary[e]));
 out<<"mode,N,dt,mass_residual,energy_residual,mass_relative,energy_relative,element_residual,min_pore_mass,zero_sum_flux,Tmin,Tmax,solid_mass\n";
 double solidMass=0;for(auto& s:q)solidMass+=s.condensedMass[0];
 out<<"coupled,"<<mesh.nCells()<<','<<dt<<','<<maxMassResidual<<','<<maxEnergyResidual<<','<<maxMassResidual/(initial[0]+throughMass)<<','<<maxEnergyResidual/(mag(initial[1])+throughEnergy)<<','<<elementResidual<<','<<minMass<<','<<maxDiffSum<<','<<gMin(T.primitiveField())<<','<<gMax(T.primitiveField())<<','<<solidMass<<'\n';
 std::ofstream profile((runTime.path()/"profile.csv").c_str());profile<<std::setprecision(17)<<"x,T,p,phi,solidDensity,Y0\n";forAll(T,i){auto p=dry::primitive(q[i],mesh.V()[i],T[i],solid,gas,0);profile<<mesh.C()[i].x()<<','<<T[i]<<','<<p.pressure<<','<<p.porosity<<','<<q[i].condensedMass[0]/mesh.V()[i]<<','<<q[i].poreMass[0]/(q[i].poreMass[0]+q[i].poreMass[1])<<'\n';}
 T.write();
}
