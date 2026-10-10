#include "tests/TestSupport.H"
#include "gpu/GasWallMath.H"
#include <cstring>
using namespace chmt;
using chmt_test::check; using chmt_test::near;
int main(){
    auto p=chmt_test::physics();p.gasMode=ugkwp::GasMode::MixtureFrozen;
    p.gasViscosity=.02;p.gasConductivity=.8;
    for(int s=0;s<Ns;++s){p.species[s].cp1=0;p.gasDiffusivity[s]=.01;}
    p.wallModel.family=ugkwp::gaswall::WallFamily::BoundaryLayer;
    p.wallModel.model=ugkwp::gaswall::BoundaryLayerModel::ConstantTransport;
    std::string configError;check(validateWallModelConfig(p,configError),"auto workspace default accepts");
    auto capacityConfig=p;capacityConfig.wallModel.nodes=128;check(validateWallModelConfig(capacityConfig,configError),"refined node capacity accepts");
    capacityConfig.wallModel.nodes=129;check(!validateWallModelConfig(capacityConfig,configError),"unsupported capacity rejects");
    GasPrimitive bulk=chmt_test::gas(p);bulk.velocity={.5,10,0};
    ugkwp::gaswall::WallWorkspace<Real,Ns> workspace;
    GasWallClosureContext context;context.workspace=&workspace;context.ownerDistance=.2;context.matchingDistance=.8;
    context.matchingPressure=context.mechanicalPressure=bulk.pressure;context.matching.temperature=bulk.temperature;
    context.matching.velocity[0]=bulk.velocity.x;context.matching.velocity[1]=bulk.velocity.y;
    for(int s=0;s<Ns;++s)context.matching.massFraction[s]=bulk.Y[s];
    GasWallInput in;in.bulk=bulk;in.temperature=600;in.gasDistance=.2;in.area=in.gasArea=2;in.dt=.01;
    in.normal={1,0,0};in.velocity={0,2,0};in.normalSpeed=.5;in.sweptVolume=.01;
    in.wallContext=context;in.condensedRate[0]=.03;
    for(int s=0;s<Ns;++s){in.speciesRate[s]=.03*bulk.Y[s];in.poreRate[s]=.01*bulk.Y[s];in.poreSweepRate[s]=.004*bulk.Y[s];}
    SurfacePacketIdentity id;id.gasCell=0;id.solidCell=0;id.filmFace=0;
    GasWallResult gas;check(evaluateGasWall(in,id,p,gas),"shared analytic blowing wall accepts");
    near(gas.trace.temperature,600,"candidate Tw preserved");
    near(gas.trace.velocity.x,.5+.04/gas.trace.rho,"absolute blowing normal velocity");
    near(gas.primary.mass,.0006,"physical primary wall mass only");near(gas.pore.mass,.0002,"separate pore channel");
    const double peclet=.04*.8/.02;
    const double expectedShear=.02/.8*peclet/std::expm1(peclet)*8;
    near(gas.traction.y,expectedShear,"shared finite blowing shear, not lowRe gradient");
    double h=0;for(int s=0;s<Ns;++s)h+=bulk.Y[s]*speciesH(p.species[s],600);
    const double ke=.5*dot(gas.trace.velocity,gas.trace.velocity);
    near(gas.primary.advective,gas.primary.mass*(h+ke),"species formation enthalpy counted once");
    near(gas.primary.pressureWork,bulk.pressure*.01,"sweep pressure work once");
    near(gas.primary.viscousWork,-.02*dot(gas.traction,gas.trace.velocity),"absolute viscous work once");
    SurfacePhysicsInput surface;surface.bulk=bulk;surface.area=surface.gasArea=surface.solidArea=2;
    surface.normal=surface.gasNormal=surface.solidNormal={1,0,0};surface.gasDistance=.2;surface.solidDistance=.1;
    surface.hasSolid=true;surface.solidVolume=1;surface.dt=.01;surface.baseVelocity={0,2,0};
    surface.useAcceptedFlux=true;surface.useSweptGeometry=true;surface.gasSweptRate=1;surface.solidSweptRate=1;
    surface.wallContext=context;surface.acceptedCondensed[0]=.03;
    for(int s=0;s<Ns;++s){surface.acceptedGasSpecies[s]=in.speciesRate[s];surface.acceptedPoreSpecies[s]=in.poreRate[s];surface.acceptedPoreSweep[s]=in.poreSweepRate[s];}
    MaterialPrimitive material;material.temperature=500;material.conductivity=2;material.porosity=0;
    SurfacePhysicsResult candidate;check(evaluateSurfaceTemperature(surface,p,material,500,600,false,candidate),"CPU material candidate uses shared closure");
    near(candidate.gasState.velocity.x,gas.trace.velocity.x,"CPU/GPU candidate velocity agreement");
    near(candidate.gasConductive*.02,gas.primary.conductive,"CPU/GPU candidate heat agreement");
    near(candidate.gasViscousWork*.02,gas.primary.viscousWork,"CPU/GPU candidate work agreement");
    near(candidate.gasEnergy*.02,gas.primary.energy,"CPU/GPU candidate total energy agreement");
    auto melting=p;melting.material.enableMelting=true;melting.material.phaseCondensed=0;
    melting.material.phaseFilmY[0]=1;melting.meltTemperature=600;melting.liquid.e0=melting.condensed[0].e0+200000;
    auto birth=surface;birth.hasFilm=true;birth.useAcceptedFlux=false;birth.useSweptGeometry=false;
    birth.solid.condensed[0]=birth.solidBase.condensed[0]=1000;
    material.temperature=800;SurfacePhysicsResult born;
    check(evaluateSurfaceTemperature(birth,melting,material,600,600,true,born),"film birth uses same gas closure");
    const Real latent=liquidH(melting,600,bulk.pressure)-condensedE(melting.condensed[0],600)-bulk.pressure/melting.condensed[0].rho;
    const Real incoming=.8/.8*(bulk.temperature-600)+.02*64/(2*.8);
    near(born.phaseMass,(2/.1*(800-600)+incoming)/latent,"birth heat includes boundaryLayer heat and mechanical heating once");
    auto prepared=gas.layer;prepared.matchingSpeciesFlux[0]=123;prepared.reactionIntegral[0]=456;
    in.preparedLayer=&prepared;in.wallContext.workspace=nullptr;
    GasWallResult reused;check(evaluateGasWall(in,id,p,reused),"prepared device closure reused without duplicate solve");
    near(reused.primary.energy,gas.primary.energy,"auxiliary chemistry cannot enter exchange packet");
    prepared.traceTemperature+=1;check(!evaluateGasWall(in,id,p,reused),"stale material Tw in prepared output rejected");
    prepared.traceTemperature-=1;
    auto invalid=p;invalid.material.gasContactResistance=.1;
    GasWallResult sentinel=gas;check(!evaluateGasWall(in,id,invalid,sentinel),"temperature jump rejected for boundaryLayer");
    check(std::memcmp(&sentinel,&gas,sizeof(gas))==0,"failed closure does not publish packet");
    invalid=p;invalid.wallModel.family=ugkwp::gaswall::WallFamily::WallFunction;
    in.speciesRate[0]=.02;in.poreRate[0]=-.02;in.speciesRate[1]=in.poreRate[1]=0;
    check(!evaluateGasWall(in,id,invalid,sentinel),"ordinary wallFunction rejects canceling physical channels");
    auto reacting=p;reacting.gasMode=ugkwp::GasMode::MixtureChemistry;
    reacting.wallModel.model=ugkwp::gaswall::BoundaryLayerModel::FiniteRate;
    reacting.wallModel.nodes=40;reacting.wallModel.relativeTolerance=1e-9;
    reacting.gasViscosity=2e-5;reacting.gasConductivity=.04;
    for(int s=0;s<Ns;++s){reacting.species[s]=reacting.species[0];reacting.gasDiffusivity[s]=1e-4;}
    ugkwp::GasReactionData<Real> reaction;reaction.reactantCount=reaction.productCount=1;reaction.highRate.preExponential=2;
    ugkwp::GasStoichTerm<Real> reactant,product;reactant.species=0;reactant.coefficient=1;product.species=1;product.coefficient=1;
    Real basis[Ns]{};basis[0]=-1;basis[1]=1;
    GasWallInput chemical;chemical.bulk=bulk;chemical.bulk.temperature=500;chemical.bulk.velocity={};
    chemical.temperature=500;chemical.area=chemical.gasArea=1;chemical.dt=.001;chemical.gasDistance=.004;chemical.normal={1,0,0};
    chemical.wallContext=context;chemical.wallContext.ownerDistance=.004;chemical.wallContext.matchingDistance=.01;
    chemical.wallContext.matching.temperature=500;for(int d=0;d<3;++d)chemical.wallContext.matching.velocity[d]=0;
    auto& mechanism=chemical.wallContext.mechanism;mechanism.reactions=&reaction;mechanism.reactionCount=1;
    mechanism.reactants=&reactant;mechanism.reactantTermCount=1;mechanism.products=&product;mechanism.productTermCount=1;
    mechanism.referencePressure=101325;mechanism.speciesOrderHash=reacting.speciesFingerprint;
    mechanism.stoichiometricBasis=basis;mechanism.independentRank=1;
    GasWallResult reactive;check(evaluateGasWall(chemical,id,reacting,reactive),"CHMT finite-rate canonical mechanism and thermo agree");
    near(reactive.trace.Y[0],bulk.Y[0]/std::cosh(std::sqrt(2/1e-4)*.01),"independent finite-rate diffusion wall composition",.002);
    check(std::abs(reactive.layer.reactionIntegral[0])>1e-4,"wall auxiliary chemistry is resolvable");
    check(reactive.primary.mass==0&&reactive.primary.species[0]==0&&reactive.pore.mass==0,"auxiliary reaction is never material production");
    check(!reactive.layer.chemistryMismatchAvailable,"no coarse chemistry comparison is fabricated");
    reacting.gasMode=ugkwp::GasMode::MixtureFrozen;GasWallResult frozen;
    check(evaluateGasWall(chemical,id,reacting,frozen),"same finite-rate core frozen control");
    near(frozen.trace.Y[0],bulk.Y[0],"frozen composition contrast");

}
