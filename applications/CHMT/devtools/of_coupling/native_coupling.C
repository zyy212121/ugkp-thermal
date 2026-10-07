/* Native OF10 verification adapter, not the CHMT CUDA production backend.
 * The central-upwind flux expressions and conservative native FV equations
 * below are adapted from OpenFOAM Foundation 10 rhoCentralFoam.C,
 * Copyright (C) 2011-2022 OpenFOAM Foundation, GPL-3.0-or-later.
 * See $WM_PROJECT_DIR/applications/solvers/compressible/rhoCentralFoam.
 * Adaptations: equal-property ideal-gas constituents; CHMT interface fluxes;
 * small laminar test cases, direct prescribed CHMT mesh points; no turbulence.
 */
#include "fvCFD.H"
#include "directionInterpolate.H"
#include "zeroGradientFvPatchFields.H"
#include "pointIOField.H"
#include "materials/CpuMaterialDriver.H"
#include "materials/MaterialCaloric.H"
#include "ablation/CpuSurfaceInterface.H"
#include "gpu/GasWallMath.H"
#include "mesh/Geometry.H"
#include "mesh/SweepConstraints.H"
#include "mesh/TrajectorySurface.H"
#include "coupling/IntervalAudit.H"
#include "restart/Checkpoint.H"
#include "gpu/StateValidation.H"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <cstring>
using namespace Foam;
using chmt::Real;
using chmt::Vec3;
using chmt::HostState;
using chmt::HostMesh;
static void check(bool ok,const std::string& why){if(!ok)throw std::runtime_error(why);}
static Foam::vector asFoam(Vec3 p){return {p.x,p.y,p.z};}
static Vec3 cv(const Foam::vector& p){return {p.x(),p.y(),p.z()};}
static pointField points(const std::vector<Vec3>& x){pointField a(x.size());forAll(a,i)a[i]=asFoam(x[i]);return a;}
static HostMesh importMesh(const fvMesh& mesh){
    HostMesh m;for(const auto& p:mesh.points())m.points.push_back(cv(p));
    m.oldPoints=m.referencePoints=m.points;m.faceOffsets.push_back(0);
    m.neighbour.assign(mesh.nFaces(),-1);m.periodicPartner.assign(mesh.nFaces(),-1);
    m.boundaryKind.assign(mesh.nFaces(),chmt::BoundaryKind::Slip);
    m.boundaryPrimitive.resize(mesh.nFaces());m.boundarySst.resize(mesh.nFaces());
    forAll(mesh.faces(),f){m.owner.push_back(mesh.faceOwner()[f]);m.faceIds.push_back(f);
        if(f<mesh.nInternalFaces()){m.neighbour[f]=mesh.faceNeighbour()[f];m.boundaryKind[f]=chmt::BoundaryKind::Internal;}
        for(int p:mesh.faces()[f])m.facePoints.push_back(p);m.faceOffsets.push_back(m.facePoints.size());}
    std::string error;check(chmt::rebuildGeometry(m,error),error);m.oldVolumes=m.volumes;return m;
}
static autoPtr<fvMesh> cloneMesh(const fvMesh& source,const HostMesh& m){
    autoPtr<fvMesh> mesh(new fvMesh(IOobject(source.name(),source.time().timeName(),source.time(),IOobject::NO_READ,IOobject::NO_WRITE,false),
        points(m.points),faceList(source.faces()),labelList(source.faceOwner()),labelList(source.faceNeighbour())));
    List<polyPatch*> patches(source.boundaryMesh().size());
    forAll(patches,p)patches[p]=source.boundaryMesh()[p].clone(mesh().boundaryMesh()).ptr();
    mesh().addFvPatches(patches);return mesh;
}
static chmt::ModelConfig model(bool reaction){
    chmt::ModelConfig m;auto& p=m.physics;p.modelFingerprint=123;p.minDt=1e-12;p.maxDt=1e-3;p.cfl=.4;p.spatialOrder=1;p.nElements=1;
    p.gasConductivity=2;p.meshMotion.policy=reaction?chmt::MeshMotionPolicy::CoupledRecession:chmt::MeshMotionPolicy::Static;
    for(auto& s:p.species){s.R=287;s.cp0=1000;s.e0=200000;s.Tmin=100;s.Tmax=3000;s.element[0]=1;}
    for(auto& c:p.condensed){c.rho=10;c.cp0=1000;c.conductivity=20;c.Tmin=100;c.Tmax=3000;c.element[0]=1;}
    p.material.enableSurfaceReactions=reaction;p.material.nSurfaceReactions=reaction?1:0;
    auto& r=p.material.surfaceReactions[0];r.A=10;r.activationEnergy=15000;r.condensedNu[0]=-1;r.gasNu[1]=1;
    p.tolerances.absoluteGeometry=1e-18;p.tolerances.relativeGeometry=1e-11;
    p.tolerances.maxCouplingIterations=40;p.tolerances.maxRetries=12;
    return m;
}
static HostState initial(const fvMesh& gas,const fvMesh& solid,const chmt::PhysicsConfig& p,bool gcl){
    HostState h;h.gasMesh=importMesh(gas);h.gas.resize(gas.nCells());
    chmt::GasPrimitive w;w.temperature=700;w.pressure=100000;w.rho=w.pressure/(287*w.temperature);w.Y[0]=gcl?.8:1;w.Y[1]=gcl?.2:0;
    if(gcl)w.velocity={3,-2,1};
    forAll(gas.C(),c)h.gas[c]=chmt::conservativeGas(w,h.gasMesh.volumes[c],p);
    if(gcl)return h;
    h.solidMesh=importMesh(solid);h.solid.resize(solid.nCells());
    forAll(solid.C(),c){const Real fraction=1;h.solid[c].condensed[0]=fraction*p.condensed[0].rho*h.solidMesh.volumes[c];h.solid[c].condensed[1]=(1-fraction)*p.condensed[1].rho*h.solidMesh.volumes[c];h.solid[c].energy=(h.solid[c].condensed[0]+h.solid[c].condensed[1])*chmt::condensedE(p.condensed[0],500+40*solid.C()[c].y()/.01+20*solid.C()[c].z()/.01);}
    int gp=gas.boundaryMesh().findPatchID("left"),sp=solid.boundaryMesh().findPatchID("right");
    check(gas.boundary()[gp].size()==solid.boundary()[sp].size(),"conformal patch counts");
    forAll(gas.boundary()[gp],f){const int gf=gas.boundary()[gp].start()+f;int sf=-1;
        forAll(solid.boundary()[sp],j){const int candidate=solid.boundary()[sp].start()+j;if(Foam::mag(gas.Cf()[gf]-solid.Cf()[candidate])<1e-12){check(sf<0,"ambiguous face-centre match");sf=candidate;}}
        check(sf>=0,"explicit conformal face-centre match");
        h.surface.gasFace.push_back(gf);h.surface.solidFace.push_back(sf);h.surface.solidCell.push_back(h.solidMesh.owner[sf]);h.surface.persistentId.push_back(sf);
        h.gasMesh.boundaryKind[gf]=h.solidMesh.boundaryKind[sf]=chmt::BoundaryKind::Interface;}
    h.filmAux.resize(h.surface.gasFace.size());h.surface.baseVelocity.resize(h.filmAux.size());
    std::string error;check(chmt::rebuildGeometry(h.gasMesh,error),error);check(chmt::rebuildGeometry(h.solidMesh,error),error);check(chmt::rebuildTrajectorySurface(h,h.gasMesh,h.solidMesh,1,h.surface,error,p.tolerances),error);
    for(std::size_t f=0;f<h.filmAux.size();++f){h.filmAux[f].area=h.surface.area[f];h.filmAux[f].normal=h.surface.normal[f];}
    return h;
}
struct Metrics {
    Real maxGcl=0,maxNativeVolumeError=0,maxNativeSweepError=0,maxCfl=0,maxSpeciesSumError=0,maxFreestreamError=0;
    Real maxMassResidual=0,maxEnergyResidual=0,minGasTemperature=1e30,minDensity=1e30;
    Real pressureWork=0,conductive=0,advective=0,massTransferred=0,maxWallDifference=0,maxRecessionMismatch=0,maxThermalDiffusionNumber=0,maxMomentumResidual=0;
    Real maxRelativeGcl=0,maxNativeVolumeRelative=0,maxNativeSweepRelative=0,maxRelativeMassResidual=0,maxRelativeEnergyResidual=0,maxRelativeSpeciesSumError=0;
    Real externalPressureWork=0,maxOuterNormalSpeed=0,maxSweepRemainder=0,maxSweepRemainderRelative=0,maxPackingVolumeRelative=0;
    int gasSteps=0,acceptedWindows=0,replays=0,linearSolves=0,executedGasSteps=0,executedMaterialSolves=0,executedLinearSolves=0;bool rollbackChecked=false;Real restartError=-1;
};
static Real absoluteEnergy(const HostState& h){Real e=0;for(const auto& q:h.gas)e+=std::abs(q.energy);for(const auto& q:h.solid)e+=std::abs(q.energy);return e;}
static Real relativeGcl(const HostMesh& old,const HostMesh& next,const std::vector<Real>& sweep){
    Real worst=0;for(std::size_t c=0;c<old.volumes.size();++c){long double total=0;for(int i=old.cellFaceOffsets[c];i<old.cellFaceOffsets[c+1];++i)total+=static_cast<long double>(old.cellFaceSigns[i])*sweep[old.cellFaces[i]];const Real residual=static_cast<Real>(static_cast<long double>(next.volumes[c])-old.volumes[c]-total);worst=std::max(worst,std::abs(residual)/std::min(old.volumes[c],next.volumes[c]));}return worst;
}
static IOobject fieldIo(const word& n,const fvMesh& mesh){return IOobject(n,mesh.time().timeName(),mesh,IOobject::NO_READ,IOobject::NO_WRITE);}
static scalar nativeFace(const surfaceScalarField& q,const fvMesh& mesh,int f){if(f<mesh.nInternalFaces())return q[f];int p=mesh.boundaryMesh().whichPatch(f);return q.boundaryField()[p][f-mesh.boundary()[p].start()];}
static void addGasBudget(chmt::Budget& b,const chmt::ExchangePacket& packet){
    b.exchangeMass[chmt::GasParticipant]+=packet.mass;b.exchangeEnergy[chmt::GasParticipant]+=packet.energy;
    b.exchangeMomentum[chmt::GasParticipant]+=packet.momentum;
    for(int s=0;s<chmt::Ns;++s)b.exchangeSpecies[chmt::GasParticipant][s]+=packet.species[s];
}
static void fillStage(chmt::HostStageGeometry& stage,const HostMesh& old,const HostMesh& next,const std::vector<Real>& sweep,Real dt){
    stage.interval=dt;stage.geometryVersion=next.geometryVersion;stage.topologyHash=next.topologyHash;
    stage.oldVolume=old.volumes;stage.newVolume=next.volumes;stage.evaluationVolume=old.volumes;stage.sweptVolume=sweep;
    stage.areaVector=next.areaVectors;stage.cellCentre=next.cellCentres;stage.faceCentre=next.faceCentres;stage.oldPoints=old.points;stage.newPoints=next.points;
}
static HostState gasWindow(const fvMesh& source,Time& time,const HostState& base,const chmt::WallProgram& program,
    const chmt::PhysicsConfig& p,Real micro,bool gcl,chmt::IntervalHistory& history,Metrics& stats){
    using namespace chmt;
    std::string error;check(history.begin(program.interval,error),error);HostState h=base;
    time.setTime(base.time,static_cast<label>(base.acceptedSteps));
    autoPtr<fvMesh> native=cloneMesh(source,h.gasMesh);fvMesh& mesh=native();
    wordList bc(mesh.boundary().size(),zeroGradientFvPatchScalarField::typeName);
    volScalarField rho(fieldIo("rho",mesh),mesh,dimensionedScalar(dimDensity,1),bc);
    volVectorField rhoU(fieldIo("rhoU",mesh),mesh,dimensionedVector(dimDensity*dimVelocity,Zero),bc);
    volScalarField rhoE(fieldIo("rhoE",mesh),mesh,dimensionedScalar(dimPressure,1),bc);
    PtrList<volScalarField> rhoY(Ns);
    for(int s=0;s<Ns;++s)rhoY.set(s,new volScalarField(fieldIo("rhoY"+Foam::name(s),mesh),mesh,dimensionedScalar(dimDensity,0),bc));
    forAll(mesh.C(),c){rho[c]=h.gas[c].mass/mesh.V()[c];rhoU[c]=asFoam(h.gas[c].momentum)/mesh.V()[c];rhoE[c]=h.gas[c].energy/mesh.V()[c];for(int s=0;s<Ns;++s)rhoY[s][c]=h.gas[c].species[s]/mesh.V()[c];}
    rho.correctBoundaryConditions();rhoU.correctBoundaryConditions();rhoE.correctBoundaryConditions();
    rho.oldTime();rhoU.oldTime();rhoE.oldTime();for(int s=0;s<Ns;++s){rhoY[s].correctBoundaryConditions();rhoY[s].oldTime();}
    surfaceScalarField pos(fieldIo("pos",mesh),mesh,dimensionedScalar(dimless,1));
    surfaceScalarField neg(fieldIo("neg",mesh),mesh,dimensionedScalar(dimless,-1));
    const GasQ freeQ=conservativeGas([&](){GasPrimitive w;check(recoverGas(base.gas[0],base.gasMesh.volumes[0],p,w),"freestream recovery");return w;}(),1,p);
    int step=0;const int n=std::max(1,int(std::ceil((program.interval.end-program.interval.begin)/micro-1e-10)));
    for(int microIndex=0;microIndex<n;++microIndex){
        const Real end=microIndex+1==n?program.interval.end:program.interval.begin+(program.interval.end-program.interval.begin)*Real(microIndex+1)/n;
        const Real dt=end-h.time;WallKnot wall,endwall;
        check(intervalSampleWall(program,.5*(h.time+end),wall,error),error);check(intervalSampleWall(program,end,endwall,error),error);
        HostState next=h;std::vector<Real> sweep,solidSweep;
        check(makeStageGeometry(h.gasMesh,endwall.gasPoints,dt,next.gasMesh,sweep,error,p.tolerances),error);
        if(!h.solid.empty()){check(makeStageGeometry(h.solidMesh,endwall.solidPoints,dt,next.solidMesh,solidSweep,error,p.tolerances),error);check(rebuildTrajectorySurface(h,next.gasMesh,next.solidMesh,dt,next.surface,error,p.tolerances),error);}
        time.setDeltaT(dt);++time;mesh.movePoints(points(next.gasMesh.points));
        stats.maxGcl=std::max(stats.maxGcl,maximumGclResidual(h.gasMesh,next.gasMesh,sweep));stats.maxRelativeGcl=std::max(stats.maxRelativeGcl,relativeGcl(h.gasMesh,next.gasMesh,sweep));
        if(!h.solid.empty()){stats.maxGcl=std::max(stats.maxGcl,maximumGclResidual(h.solidMesh,next.solidMesh,solidSweep));stats.maxRelativeGcl=std::max(stats.maxRelativeGcl,relativeGcl(h.solidMesh,next.solidMesh,solidSweep));}
        forAll(mesh.V(),c){stats.maxNativeVolumeError=std::max(stats.maxNativeVolumeError,std::abs(mesh.V()[c]-next.gasMesh.volumes[c]));stats.maxNativeVolumeRelative=std::max(stats.maxNativeVolumeRelative,std::abs(mesh.V()[c]-next.gasMesh.volumes[c])/next.gasMesh.volumes[c]);}
        forAll(mesh.faces(),f){const Real residual=std::abs(dt*nativeFace(mesh.phi(),mesh,f)-sweep[f]);Real scale=next.gasMesh.volumes[next.gasMesh.owner[f]];if(next.gasMesh.neighbour[f]>=0)scale=std::min(scale,next.gasMesh.volumes[next.gasMesh.neighbour[f]]);stats.maxNativeSweepError=std::max(stats.maxNativeSweepError,residual);stats.maxNativeSweepRelative=std::max(stats.maxNativeSweepRelative,residual/scale);}
        volVectorField U(fieldIo("U",mesh),rhoU/rho,bc);
        volScalarField e(fieldIo("e",mesh),rhoE/rho-.5*magSqr(U),bc);
        volScalarField T(fieldIo("T",mesh),(e-dimensionedScalar(dimVelocity*dimVelocity,p.species[0].e0))/dimensionedScalar(dimVelocity*dimVelocity/dimTemperature,713),bc);
        volScalarField rPsi(fieldIo("rPsi",mesh),dimensionedScalar(dimVelocity*dimVelocity/dimTemperature,287)*T,bc);
        U.correctBoundaryConditions();e.correctBoundaryConditions();T.correctBoundaryConditions();rPsi.correctBoundaryConditions();
        // The following Kurganov flux is the OF10 rhoCentralFoam formulation.
        surfaceScalarField rho_pos(Foam::interpolate(rho,pos)),rho_neg(Foam::interpolate(rho,neg));
        surfaceVectorField rhoU_pos(Foam::interpolate(rhoU,pos,U.name())),rhoU_neg(Foam::interpolate(rhoU,neg,U.name()));
        surfaceScalarField rPsi_pos(Foam::interpolate(rPsi,pos,T.name())),rPsi_neg(Foam::interpolate(rPsi,neg,T.name()));
        surfaceScalarField e_pos(Foam::interpolate(e,pos,T.name())),e_neg(Foam::interpolate(e,neg,T.name()));
        surfaceVectorField U_pos("U_pos",rhoU_pos/rho_pos),U_neg("U_neg",rhoU_neg/rho_neg);
        surfaceScalarField p_pos("p_pos",rho_pos*rPsi_pos),p_neg("p_neg",rho_neg*rPsi_neg);
        surfaceScalarField phiv_pos("phiv_pos",U_pos&mesh.Sf()),phiv_neg("phiv_neg",U_neg&mesh.Sf());
        phiv_pos-=mesh.phi();phiv_neg-=mesh.phi();
        volScalarField sound("sound",sqrt((1000./713)*rPsi));
        surfaceScalarField cSf_pos("cSf_pos",Foam::interpolate(sound,pos,T.name())*mesh.magSf()),cSf_neg("cSf_neg",Foam::interpolate(sound,neg,T.name())*mesh.magSf());
        const dimensionedScalar zeroFlux(dimVolume/dimTime,0);
        surfaceScalarField ap("ap",max(max(phiv_pos+cSf_pos,phiv_neg+cSf_neg),zeroFlux));
        surfaceScalarField am("am",min(min(phiv_pos-cSf_pos,phiv_neg-cSf_neg),zeroFlux));
        surfaceScalarField a_pos("a_pos",ap/(ap-am)),a_neg("a_neg",1-a_pos),aSf("aSf",am*a_pos);
        phiv_pos*=a_pos;phiv_neg*=a_neg;
        surfaceScalarField aphiv_pos("aphiv_pos",phiv_pos-aSf),aphiv_neg("aphiv_neg",phiv_neg+aSf);
        surfaceScalarField phi("phi",aphiv_pos*rho_pos+aphiv_neg*rho_neg);
        surfaceVectorField phiUp("phiUp",aphiv_pos*rhoU_pos+aphiv_neg*rhoU_neg+(a_pos*p_pos+a_neg*p_neg)*mesh.Sf());
        surfaceScalarField phiEp("phiEp",aphiv_pos*(rho_pos*(e_pos+.5*magSqr(U_pos))+p_pos)+aphiv_neg*(rho_neg*(e_neg+.5*magSqr(U_neg))+p_neg)+aSf*p_pos-aSf*p_neg);
        // OF10 converts pressure work back to the absolute frame, once.
        phiEp+=mesh.phi()*(a_pos*p_pos+a_neg*p_neg);
        // Explicit physical heat diffusion, zero-gradient at noninterface walls.
        if(!gcl)phiEp-=dimensionedScalar(dimPower/dimLength/dimTemperature,p.gasConductivity)*fvc::snGrad(T)*mesh.magSf();
        PtrList<surfaceScalarField> phiY(Ns);
        for(int s=0;s<Ns;++s)phiY.set(s,new surfaceScalarField("phiY"+Foam::name(s),aphiv_pos*Foam::interpolate(rhoY[s],pos)+aphiv_neg*Foam::interpolate(rhoY[s],neg)));
        surfaceScalarField maxWave("maxWave",max(mag(ap),mag(am)));
        scalarField waveSum(fvc::surfaceSum(maxWave)().primitiveField());
        forAll(mesh.V(),c)stats.maxCfl=std::max(stats.maxCfl,.5*dt*waveSum[c]/mesh.V()[c]);
        check(stats.maxCfl<.45,"native acoustic CFL exceeded");
        if(!gcl){surfaceScalarField conductance("conductance",dimensionedScalar(dimPower/dimLength/dimTemperature,p.gasConductivity)*mesh.magSf()*mesh.deltaCoeffs());scalarField thermal(fvc::surfaceSum(conductance)().primitiveField());forAll(mesh.V(),c)stats.maxThermalDiffusionNumber=std::max(stats.maxThermalDiffusionNumber,dt*thermal[c]/(rho[c]*713*mesh.V()[c]));check(stats.maxThermalDiffusionNumber<.45,"explicit thermal diffusion bound exceeded");}
        GasIntervalRecord record;record.microSequence=step;record.begin=h.time;record.end=end;record.gasGeometry=h.gasMesh.geometryVersion;record.solidGeometry=h.solidMesh.geometryVersion;
        if(!gcl){
            forAll(mesh.boundary(),b)forAll(mesh.boundary()[b],f){
                phi.boundaryFieldRef()[b][f]=0;phiEp.boundaryFieldRef()[b][f]=p_pos.boundaryField()[b][f]*mesh.phi().boundaryField()[b][f];
                const int face=mesh.boundary()[b].start()+f;
                if(h.gasMesh.boundaryKind[face]!=BoundaryKind::Interface){const Real work=dt*phiEp.boundaryField()[b][f];next.budget.boundaryEnergy+=work;stats.externalPressureWork+=work;stats.maxOuterNormalSpeed=std::max(stats.maxOuterNormalSpeed,std::abs(mesh.phi().boundaryField()[b][f])/mesh.magSf().boundaryField()[b][f]);}
                phiUp.boundaryFieldRef()[b][f]=p_pos.boundaryField()[b][f]*mesh.Sf().boundaryField()[b][f];
                for(int s=0;s<Ns;++s)phiY[s].boundaryFieldRef()[b][f]=0;
                next.budget.boundaryMomentum+=cv(phiUp.boundaryField()[b][f])*dt;
            }
            for(std::size_t f=0;f<h.surface.area.size();++f){
                int gf=h.surface.gasFace[f],c=h.gasMesh.owner[gf];GasPrimitive bulk;
                check(recoverGas(h.gas[c],h.gasMesh.volumes[c],p,bulk),"actual native gas bulk recovery");
                GasWallInput input;input.bulk=bulk;input.temperature=wall.faces[f].temperature;input.gasDistance=h.surface.gasDistance[f];
                input.area=h.surface.area[f];input.gasArea=mag(h.gasMesh.areaVectors[gf]);input.dt=dt;
                input.normal=-h.gasMesh.areaVectors[gf]/input.gasArea;input.normalSpeed=-sweep[gf]/(input.gasArea*dt);input.sweptVolume=-sweep[gf];
                input.velocity=wall.faces[f].velocity;input.primaryKind=wall.faces[f].primaryKind;
                for(int s=0;s<Ns;++s){input.speciesRate[s]=wall.faces[f].speciesRate[s];input.poreRate[s]=wall.faces[f].poreRate[s];input.poreSweepRate[s]=wall.faces[f].poreSweepRate[s];}
                for(int k=0;k<Nc;++k)input.condensedRate[k]=wall.faces[f].condensedRate[k];
                SurfacePacketIdentity id;id.step=step;id.stage=1;id.geometry=record.gasGeometry;id.face=h.gasMesh.faceIds[gf];id.gasCell=c;id.solidCell=h.surface.solidCell[f];id.filmFace=f;
                GasWallResult wallResult;check(evaluateGasWall(input,id,p,wallResult),"production gas-wall packet rejected");
                record.gasTrace.push_back(bulk);record.gasTraction.push_back(wallResult.traction);
                const int b=mesh.boundaryMesh().whichPatch(gf),bf=gf-mesh.boundary()[b].start();
                next.budget.boundaryMomentum-=cv(phiUp.boundaryField()[b][bf])*dt;
                phi.boundaryFieldRef()[b][bf]=-(wallResult.primary.mass+wallResult.pore.mass)/dt;
                phiEp.boundaryFieldRef()[b][bf]=-(wallResult.primary.energy+wallResult.pore.energy)/dt;
                phiUp.boundaryFieldRef()[b][bf]=-asFoam(wallResult.primary.momentum+wallResult.pore.momentum)/dt;
                for(int s=0;s<Ns;++s)phiY[s].boundaryFieldRef()[b][bf]=-(wallResult.primary.species[s]+wallResult.pore.species[s])/dt;
                for(auto packet:{wallResult.primary,wallResult.pore}){packet.consumerMask=ConsumeGas;record.packets.push_back(packet);addGasBudget(next.budget,packet);}
                stats.pressureWork+=wallResult.primary.pressureWork;stats.conductive+=wallResult.primary.conductive;stats.advective+=wallResult.primary.advective;stats.massTransferred+=wallResult.primary.mass;
            }
        }
        // Actual native conservative transient equations, as in rhoCentralFoam.
        solve(fvm::ddt(rho)+fvc::div(phi));
        solve(fvm::ddt(rhoU)+fvc::div(phiUp));
        solve(fvm::ddt(rhoE)+fvc::div(phiEp));
        for(int s=0;s<Ns;++s)solve(fvm::ddt(rhoY[s])+fvc::div(phiY[s]));
        rho.correctBoundaryConditions();rhoU.correctBoundaryConditions();rhoE.correctBoundaryConditions();for(int s=0;s<Ns;++s)rhoY[s].correctBoundaryConditions();
        forAll(mesh.V(),c){auto& q=next.gas[c];q.mass=rho[c]*mesh.V()[c];q.momentum=cv(rhoU[c])*mesh.V()[c];q.energy=rhoE[c]*mesh.V()[c];Real sum=0;
            for(int s=0;s<Ns;++s){q.species[s]=rhoY[s][c]*mesh.V()[c];sum+=q.species[s];}
            stats.maxSpeciesSumError=std::max(stats.maxSpeciesSumError,std::abs(sum-q.mass));stats.maxRelativeSpeciesSumError=std::max(stats.maxRelativeSpeciesSumError,std::abs(sum-q.mass)/q.mass);GasPrimitive w;
            check(recoverGas(q,mesh.V()[c],p,w),"native gas endpoint not positive/admissible");stats.minGasTemperature=std::min(stats.minGasTemperature,w.temperature);stats.minDensity=std::min(stats.minDensity,w.rho);
            if(gcl){const GasQ a=q*(1/mesh.V()[c]);Real err=std::abs(a.mass-freeQ.mass)/freeQ.mass;
                err=std::max(err,std::abs(a.energy-freeQ.energy)/freeQ.energy);err=std::max(err,chmt::mag(a.momentum-freeQ.momentum)/std::max(1.,chmt::mag(freeQ.momentum)));
                for(int s=0;s<Ns;++s)err=std::max(err,std::abs(a.species[s]-freeQ.species[s])/freeQ.mass);stats.maxFreestreamError=std::max(stats.maxFreestreamError,err);}
        }
        check(history.appendAccepted(record,error),error);next.time=end;next.acceptedSteps=h.acceptedSteps+1;
        fillStage(next.gasStages[0],h.gasMesh,next.gasMesh,sweep,dt);next.gasStages[1]=next.gasStages[0];
        if(!h.solid.empty()){fillStage(next.solidStages[0],h.solidMesh,next.solidMesh,solidSweep,dt);next.solidStages[1]=next.solidStages[0];}
        next.lastAcceptedDt=dt;next.nextDt=micro;h=std::move(next);++step;++stats.gasSteps;
    }
    return h;
}
static void accumulate(Metrics& a,const Metrics& b){
#define MX(f) a.f=std::max(a.f,b.f)
    MX(maxGcl);MX(maxNativeVolumeError);MX(maxNativeSweepError);MX(maxCfl);MX(maxSpeciesSumError);MX(maxFreestreamError);MX(maxMassResidual);MX(maxEnergyResidual);MX(maxThermalDiffusionNumber);MX(maxMomentumResidual);MX(maxRelativeGcl);MX(maxNativeVolumeRelative);MX(maxNativeSweepRelative);MX(maxRelativeMassResidual);MX(maxRelativeEnergyResidual);MX(maxRelativeSpeciesSumError);MX(maxOuterNormalSpeed);MX(maxSweepRemainder);MX(maxSweepRemainderRelative);MX(maxPackingVolumeRelative);
#undef MX
    a.minGasTemperature=std::min(a.minGasTemperature,b.minGasTemperature);a.minDensity=std::min(a.minDensity,b.minDensity);
    a.gasSteps+=b.gasSteps;a.pressureWork+=b.pressureWork;a.conductive+=b.conductive;a.advective+=b.advective;a.massTransferred+=b.massTransferred;a.externalPressureWork+=b.externalPressureWork;
}
static HostState coupledWindow(const fvMesh& gas,fvMesh& solid,Time& time,const HostState& base,
    const chmt::ModelConfig& config,Real end,Real micro,Metrics& stats,bool testRollback){
    using namespace chmt;CpuMaterialDriver driver(config,&solid);CpuMaterialControls controls;controls.maxSubstep=end-base.time;
    CouplingInterval interval;interval.begin=base.time;interval.end=end;interval.identity.sequence=base.commitSequence+1;
    std::string error;WallProgram program;const bool predicted=driver.predictWall(base,interval,program,error);check(predicted,"predictWall: "+error);
    const pointField originalGas=gas.points(),originalSolid=solid.points();
    const Foam::label originalIndex=time.timeIndex();const Real originalTime=time.value();
    for(int iteration=0;iteration<30;++iteration){
        IntervalHistory history;Metrics attempted;HostState gasEnd=gasWindow(gas,time,base,program,config.physics,micro,false,history,attempted);
        HostState candidate;WallProgram corrected;CpuMaterialReport report;
        stats.executedGasSteps+=attempted.gasSteps;++stats.executedMaterialSolves;
        if(!driver.advanceCandidate(base,gasEnd,history,controls,candidate,corrected,report,error,&program)){std::ostringstream detail;detail<<"material candidate: "<<error<<" complete="<<history.complete()<<" baseTime="<<std::setprecision(17)<<base.time<<" begin="<<history.interval().begin<<" end="<<history.end()<<" desiredEnd="<<history.interval().end<<" gasTopo="<<base.gasMesh.topologyHash<<","<<gasEnd.gasMesh.topologyHash<<" solidTopo="<<base.solidMesh.topologyHash<<","<<gasEnd.solidMesh.topologyHash;throw std::runtime_error(detail.str());}
        stats.executedLinearSolves+=report.linearSolves;
        if(testRollback&&!stats.rollbackChecked){
            HostState sentinel;sentinel.time=-123;sentinel.solid.resize(1);sentinel.solid[0].energy=17;WallProgram sentinelWall;sentinelWall.interval.identity.sequence=987;
            CpuMaterialControls failed=controls;failed.nonlinearMaxIterations=1;CpuMaterialReport rejectedReport;
            check(!driver.advanceCandidate(base,gasEnd,history,failed,sentinel,sentinelWall,rejectedReport,error,&program),"deliberate native nonlinear rejection expected");
            check(error=="OpenFOAM material caloric nonlinear iteration failed","expected native nonlinear failure: "+error);
            check(sentinel.time==-123&&sentinel.solid.size()==1&&sentinel.solid[0].energy==17&&sentinelWall.interval.identity.sequence==987,"rejected native candidate mutated caller");
            check(gas.points()==originalGas&&solid.points()==originalSolid,"rejected window moved accepted native meshes");
            time.setTime(originalTime,originalIndex);check(time.value()==originalTime&&time.timeIndex()==originalIndex,"native time rollback");
            ++stats.executedMaterialSolves;stats.executedLinearSolves+=rejectedReport.linearSolves;stats.rollbackChecked=true;
        }
        Tolerances compareTol=config.physics.tolerances;compareTol.absoluteTemperature=1e-6;compareTol.relativeTemperature=1e-9;
        WallProgramComparison comparison;check(compareWallPrograms(program,corrected,compareTol,comparison,error),error);
        Real wallDifference=0;for(std::size_t f=0;f<program.knots.back().faces.size();++f)wallDifference=std::max(wallDifference,std::abs(program.knots.back().faces[f].temperature-corrected.knots.back().faces[f].temperature));
        if(!report.geometryCorrection&&comparison.regimeMatches&&comparison.thermalError<=1&&comparison.massPredictionError<=1&&comparison.geometryError<=1){
            check(validateMaterialDonorHistory(base,history,&report.donorPlan,error),"production donor history: "+error);
            WindowConservationAudit audit;check(auditCoupledCandidate(base,candidate,history,config.physics,audit,error),"production coupled audit: "+error);
            check(finalizeIntervalLedger(history,candidate,error),error);check(checkSynchronizedInterval(candidate,history,error),error);
            Vec3 momentum{};for(const auto& q:candidate.gas)momentum+=q.momentum;for(const auto& q:base.gas)momentum-=q.momentum;momentum+=(candidate.budget.boundaryMomentum-base.budget.boundaryMomentum)+(candidate.budget.supportImpulse-base.budget.supportImpulse);attempted.maxMomentumResidual=chmt::mag(momentum);
            attempted.maxMassResidual=std::abs(audit.massResidual);attempted.maxEnergyResidual=std::abs(audit.energyResidual);
            check(validateMaterialSweepRemainder(candidate,error),error);
            for(std::size_t f=0;f<candidate.solidSweepRemainder.size();++f){const int face=candidate.surface.solidFace[f];if(face<0)continue;const Real r=std::abs(candidate.solidSweepRemainder[f]);attempted.maxSweepRemainder=std::max(attempted.maxSweepRemainder,r);attempted.maxSweepRemainderRelative=std::max(attempted.maxSweepRemainderRelative,r/candidate.solidMesh.volumes[candidate.solidMesh.owner[face]]);}
            for(std::size_t c=0;c<candidate.solid.size();++c){Real occupied=0;for(int k=0;k<Nc;++k)occupied+=candidate.solid[c].condensed[k]/config.physics.condensed[k].rho;attempted.maxPackingVolumeRelative=std::max(attempted.maxPackingVolumeRelative,std::abs(occupied-candidate.solidMesh.volumes[c])/candidate.solidMesh.volumes[c]);}
            IntervalInventoryTotals initialTotals,finalTotals;check(intervalInventoryTotals(base,config.physics,initialTotals,error),error);check(intervalInventoryTotals(candidate,config.physics,finalTotals,error),error);
            attempted.maxRelativeMassResidual=attempted.maxMassResidual/std::max(double(initialTotals.mass),double(finalTotals.mass));attempted.maxRelativeEnergyResidual=attempted.maxEnergyResidual/std::max(absoluteEnergy(base),absoluteEnergy(candidate));
            autoPtr<fvMesh> nativeSolid=cloneMesh(solid,candidate.solidMesh);forAll(nativeSolid().V(),c){const Real delta=std::abs(nativeSolid().V()[c]-candidate.solidMesh.volumes[c]);attempted.maxNativeVolumeError=std::max(attempted.maxNativeVolumeError,delta);attempted.maxNativeVolumeRelative=std::max(attempted.maxNativeVolumeRelative,delta/candidate.solidMesh.volumes[c]);}
            accumulate(stats,attempted);stats.linearSolves+=report.linearSolves;++stats.acceptedWindows;stats.maxWallDifference=std::max(stats.maxWallDifference,wallDifference);
            candidate.acceptedSteps=gasEnd.acceptedSteps;candidate.commitSequence=base.commitSequence+1;candidate.lastAcceptedDt=end-base.time;candidate.nextDt=micro;
            // The accepted stage authority is the last actually executed gas microstep.
            candidate.gasStages=gasEnd.gasStages;candidate.solidStages=gasEnd.solidStages;
            check(gas.points()==originalGas&&solid.points()==originalSolid,"candidate unexpectedly mutated accepted native meshes");
            time.setTime(candidate.time,candidate.acceptedSteps);return candidate;
        }
        ++stats.replays;time.setTime(originalTime,originalIndex);program=std::move(corrected);program.interval.identity.epoch=iteration+1;program.donorPlan.interval.identity=program.interval.identity;
        if(iteration==29){std::ostringstream message;message<<"coupling failed to converge: geometryCorrection="<<report.geometryCorrection<<" thermal="<<comparison.thermalError<<" mass="<<comparison.massPredictionError<<" geometry="<<comparison.geometryError;throw std::runtime_error(message.str());}
    }
    throw std::runtime_error("unreachable coupling exit");
}
static std::vector<Real> normalizedState(const HostState& h,const chmt::PhysicsConfig& p){
    std::vector<Real> values;for(std::size_t c=0;c<h.gas.size();++c){chmt::GasPrimitive w;check(chmt::recoverGas(h.gas[c],h.gasMesh.volumes[c],p,w),"full-state recovery");values.insert(values.end(),{w.temperature/700,w.pressure/100000,w.Y[1],w.velocity.x,w.velocity.y,w.velocity.z});}
    for(std::size_t c=0;c<h.solid.size();++c){chmt::MaterialPrimitive w;check(chmt::recoverMaterial(h.solid[c],h.solidMesh.volumes[c],p,w),"full material recovery");values.push_back(w.temperature/500);}
    for(const auto& x:h.surface.centre)values.push_back(x.x/.004);return values;
}
static Real difference(const HostState& a,const HostState& b){
    Real x=0;check(a.solidSweepRemainder.size()==b.solidSweepRemainder.size(),"restart remainder layout");
    if(!a.solidSweepRemainder.empty()&&std::memcmp(a.solidSweepRemainder.data(),b.solidSweepRemainder.data(),a.solidSweepRemainder.size()*sizeof(Real))!=0)x=std::numeric_limits<Real>::min();
    for(std::size_t f=0;f<a.solidSweepRemainder.size();++f)x=std::max(x,std::abs(a.solidSweepRemainder[f]-b.solidSweepRemainder[f]));
    check(a.gas.size()==b.gas.size()&&a.solid.size()==b.solid.size(),"restart inventory size");
    for(std::size_t c=0;c<a.gas.size();++c){x=std::max(x,std::abs(a.gas[c].mass-b.gas[c].mass));x=std::max(x,std::abs(a.gas[c].energy-b.gas[c].energy));x=std::max(x,chmt::mag(a.gas[c].momentum-b.gas[c].momentum));for(int s=0;s<chmt::Ns;++s)x=std::max(x,std::abs(a.gas[c].species[s]-b.gas[c].species[s]));}
    for(std::size_t c=0;c<a.solid.size();++c){x=std::max(x,std::abs(a.solid[c].energy-b.solid[c].energy));for(int k=0;k<chmt::Nc;++k)x=std::max(x,std::abs(a.solid[c].condensed[k]-b.solid[c].condensed[k]));}
    for(std::size_t j=0;j<a.gasMesh.points.size();++j)x=std::max(x,chmt::mag(a.gasMesh.points[j]-b.gasMesh.points[j]));
    for(std::size_t j=0;j<a.solidMesh.points.size();++j)x=std::max(x,chmt::mag(a.solidMesh.points[j]-b.solidMesh.points[j]));return x;
}
int main(int argc,char** argv){try{
    argList::noParallel();argList::addOption("mode","word","gcl, fixed, moving");argList::addOption("micro","scalar","maximum native gas step");argList::addOption("window","scalar","material coupling interval");
    argList args(argc,argv);if(!args.checkRootCase())return 2;
    const std::string mode=args.optionLookupOrDefault<word>("mode","moving");
    const Real micro=args.optionLookupOrDefault<scalar>("micro",2e-7),window=args.optionLookupOrDefault<scalar>("window",2e-6),finish=1e-4;
    Time time(Time::controlDictName,args);fvMesh gas(IOobject(polyMesh::defaultRegion,time.timeName(),time,IOobject::MUST_READ));
    fvMesh solid(IOobject("solid",time.timeName(),time,IOobject::MUST_READ));
    const auto config=model(mode=="moving"||mode=="incompatible");HostState h=initial(gas,solid,config.physics,mode=="gcl"),begin=h;
    if(mode!="gcl"){std::string initialError;const bool valid=chmt::validateRuntimeState(config,h,initialError);check(valid,"native initial state: "+initialError);begin=h;}
    if(mode=="incompatible"){chmt::CpuMaterialDriver driver(config,&solid);chmt::CouplingInterval interval;interval.end=window;chmt::WallProgram sentinel;sentinel.interval.identity.sequence=987;std::string error;const bool accepted=driver.predictWall(h,interval,sentinel,error);check(!accepted&&error.find("incompatible swept-volume/planarity constraints")!=std::string::npos,"expected incompatible quad trajectory, got: "+error);check(sentinel.interval.identity.sequence==987&&sentinel.knots.empty()&&difference(h,begin)==0&&time.value()==0,"incompatible prediction changed accepted state");std::ofstream result((time.path()/"negative.json").c_str());result<<"{\"passed\":true,\"case\":\"incompatible_two_axis_quad_recession\",\"diagnostic\":\""<<error<<"\",\"statePreserved\":true}\n";std::cout<<"NATIVE_OF10 expected negative: "<<error<<"; accepted state preserved\n";return 0;}
    Metrics stats;auto started=std::chrono::steady_clock::now();
    const int windows=int(std::round(finish/window));check(windows>0,"positive windows");
    std::ofstream stateTrajectory((time.path()/"state_trajectory.csv").c_str());stateTrajectory<<std::setprecision(17);
    std::ofstream trajectory((time.path()/"trajectory.csv").c_str());trajectory<<std::setprecision(17)<<"time,gas_wall_temperature,gas_wall_pressure,gas_wall_y1,solid_surface_temperature,recession,total_mass,total_energy\n";
    for(int k=0;k<windows;++k){const Real end=finish*Real(k+1)/windows;
        if(mode=="gcl"){
            chmt::WallProgram program;program.interval.begin=h.time;program.interval.end=end;program.interval.identity.sequence=k+1;
            chmt::WallKnot a,b;a.time=h.time;a.gasPoints=h.gasMesh.points;b=a;b.time=end;
            for(std::size_t j=0;j<b.gasPoints.size();++j){const Vec3 ref=h.gasMesh.referencePoints[j];b.gasPoints[j].x=ref.x+5e-4*std::sin(3.141592653589793*ref.x/.02)*std::sin(2*3.141592653589793*end/finish);}
            program.knots={a,b};chmt::IntervalHistory history;Metrics trial;h=gasWindow(gas,time,h,program,config.physics,micro,true,history,trial);accumulate(stats,trial);++stats.acceptedWindows;
        }else{
            const HostState previous=h;h=coupledWindow(gas,solid,time,previous,config,end,micro,stats,true);
            if(k==windows/2){
                std::string error;const std::string path=(time.path()/"checkpoint").c_str();const bool saved=chmt::writeCheckpoint(path,config,h,error);check(saved,"native coupled checkpoint write: "+error);
                HostState restored;const bool loaded=chmt::readCheckpoint(path,config,restored,error);check(loaded,"native coupled checkpoint read: "+error);check(difference(h,restored)==0,"checkpoint inventory/points roundtrip");
                const Real nextTime=finish*Real(k+2)/windows;Metrics one,two;
                HostState continued=coupledWindow(gas,solid,time,h,config,nextTime,micro,one,false);
                HostState restarted=coupledWindow(gas,solid,time,restored,config,nextTime,micro,two,false);
                stats.restartError=difference(continued,restarted);check(stats.restartError==0,"restart native progression differs");time.setTime(h.time,h.acceptedSteps);
            }
        }
        chmt::GasPrimitive w;std::string error;check(chmt::recoverGas(h.gas[0],h.gasMesh.volumes[0],config.physics,w),"report gas recovery");
        chmt::IntervalInventoryTotals total;check(chmt::intervalInventoryTotals(h,config.physics,total,error),error);Real solidT=0,recession=0;
        if(mode!="gcl"){chmt::MaterialPrimitive material;check(chmt::recoverMaterial(h.solid[h.surface.solidCell[0]],h.solidMesh.volumes[h.surface.solidCell[0]],config.physics,material),"report solid recovery");solidT=material.temperature;recession=-h.surface.centre[0].x;}
        stateTrajectory<<h.time;for(Real value:normalizedState(h,config.physics))stateTrajectory<<','<<value;stateTrajectory<<'\n';
        trajectory<<h.time<<','<<w.temperature<<','<<w.pressure<<','<<w.Y[1]<<','<<solidT<<','<<recession<<','<<double(total.mass)<<','<<double(total.energy)<<'\n';
    }
    chmt::GasPrimitive w;check(chmt::recoverGas(h.gas[0],h.gasMesh.volumes[0],config.physics,w),"final recovery");
    const Real recession=mode=="gcl"?0:-h.surface.centre[0].x;
    pointIOField finalGasPoints(IOobject("points",time.timeName(),"polyMesh",time,IOobject::NO_READ,IOobject::NO_WRITE,false),points(h.gasMesh.points));check(finalGasPoints.write(),"write final native gas points");
    if(!h.solid.empty()){pointIOField finalSolidPoints(IOobject("points",time.timeName(),"solid/polyMesh",time,IOobject::NO_READ,IOobject::NO_WRITE,false),points(h.solidMesh.points));check(finalSolidPoints.write(),"write final native solid points");}
    const Real seconds=std::chrono::duration<Real>(std::chrono::steady_clock::now()-started).count();
    std::ofstream out((time.path()/"summary.json").c_str());out<<std::setprecision(17)<<"{\n\"completed\": true, \"mode\": \""<<mode<<"\", \"microDt\": "<<micro<<", \"windowDt\": "<<window<<", \"physicalTime\": "<<finish<<", \"wallSeconds\": "<<seconds<<", \"gasCells\": "<<gas.nCells()<<", \"solidCells\": "<<solid.nCells()<<",\n";
#define EMIT(f) out<<"\"" #f "\": "<<stats.f<<",\n"
    EMIT(maxGcl);EMIT(maxNativeVolumeError);EMIT(maxNativeSweepError);EMIT(maxCfl);EMIT(maxSpeciesSumError);EMIT(maxFreestreamError);EMIT(maxMassResidual);EMIT(maxEnergyResidual);EMIT(minGasTemperature);EMIT(minDensity);EMIT(pressureWork);EMIT(conductive);EMIT(advective);EMIT(massTransferred);EMIT(gasSteps);EMIT(acceptedWindows);EMIT(replays);EMIT(linearSolves);EMIT(restartError);EMIT(maxWallDifference);EMIT(maxThermalDiffusionNumber);EMIT(maxMomentumResidual);EMIT(executedGasSteps);EMIT(executedMaterialSolves);EMIT(executedLinearSolves);EMIT(maxRelativeGcl);EMIT(maxNativeVolumeRelative);EMIT(maxNativeSweepRelative);EMIT(maxRelativeMassResidual);EMIT(maxRelativeEnergyResidual);EMIT(maxRelativeSpeciesSumError);EMIT(externalPressureWork);EMIT(maxOuterNormalSpeed);EMIT(maxSweepRemainder);EMIT(maxSweepRemainderRelative);EMIT(maxPackingVolumeRelative);
#undef EMIT
    std::ofstream fields((time.path()/"endpoint.csv").c_str());fields<<std::setprecision(17)<<"component,value,scale\n";for(std::size_t c=0;c<h.gas.size();++c){chmt::GasPrimitive a;check(chmt::recoverGas(h.gas[c],h.gasMesh.volumes[c],config.physics,a),"endpoint field recovery");fields<<"Tg,"<<a.temperature<<",700\npg,"<<a.pressure<<",100000\nY1,"<<a.Y[1]<<",1\nUx,"<<a.velocity.x<<",1\nUy,"<<a.velocity.y<<",1\nUz,"<<a.velocity.z<<",1\n";}for(std::size_t c=0;c<h.solid.size();++c){chmt::MaterialPrimitive a;check(chmt::recoverMaterial(h.solid[c],h.solidMesh.volumes[c],config.physics,a),"endpoint material recovery");fields<<"Ts,"<<a.temperature<<",500\n";}for(const auto& c:h.surface.centre)fields<<"front,"<<c.x<<",.004\n";
    chmt::IntervalInventoryTotals initialTotal,finalTotal;std::string totalError;check(chmt::intervalInventoryTotals(begin,config.physics,initialTotal,totalError),totalError);check(chmt::intervalInventoryTotals(h,config.physics,finalTotal,totalError),totalError);
    const Real cumulativeMass=double(finalTotal.mass-initialTotal.mass)+h.budget.boundaryMass-begin.budget.boundaryMass;
    const Real cumulativeEnergy=double(finalTotal.energy-initialTotal.energy)+h.budget.boundaryEnergy-begin.budget.boundaryEnergy-(h.budget.bodyWork-begin.budget.bodyWork)-(h.budget.supportWork-begin.budget.supportWork)-(h.budget.radiation-begin.budget.radiation)-(h.budget.filmKineticDefect-begin.budget.filmKineticDefect);
    out<<"\"cumulativeMassResidual\": "<<cumulativeMass<<",\n\"cumulativeEnergyResidual\": "<<cumulativeEnergy<<",\n\"cumulativeRelativeMassResidual\": "<<std::abs(cumulativeMass)/std::max(double(initialTotal.mass),double(finalTotal.mass))<<",\n\"cumulativeRelativeEnergyResidual\": "<<std::abs(cumulativeEnergy)/std::max(absoluteEnergy(begin),absoluteEnergy(h))<<",\n\"boundaryMass\": "<<h.budget.boundaryMass<<",\n\"boundaryEnergy\": "<<h.budget.boundaryEnergy<<",\n";
    out<<"\"rollbackChecked\": "<<(stats.rollbackChecked?"true":"false")<<",\n\"recession\": "<<recession<<",\n\"endpoint\": ["<<w.temperature<<','<<w.pressure<<','<<w.Y[1]<<','<<recession<<"]\n}\n";
    std::cout<<"NATIVE_OF10_COUPLING mode="<<mode<<" physical_time="<<finish<<" gas_steps="<<stats.gasSteps<<" accepted_windows="<<stats.acceptedWindows<<" recession="<<recession<<" max_mass_residual="<<stats.maxMassResidual<<" max_energy_residual="<<stats.maxEnergyResidual<<" elapsed="<<seconds<<'\n';return 0;
}catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}}
