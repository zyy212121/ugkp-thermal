#ifndef CHMT_TEST_NATIVE_MATERIAL_PROBE_H
#define CHMT_TEST_NATIVE_MATERIAL_PROBE_H
// Observe the real sparse solves and material/film calls only in this test
// translation unit. Preload every dependency before the scoped macros so
// neither native declarations nor separately compiled production code change.
#include "materials/CpuMaterialDriver.H"
#include "core/ThermalBoundary.H"
#include "materials/MaterialTransport.H"
#include "materials/MaterialDrive.H"
#include "materials/SurfacePacketAddressing.H"
#include "ablation/CpuSurfaceInterface.H"
#include "ablation/WallClosureHost.H"
#include "film/CpuFilmDriver.H"
#include "mesh/Geometry.H"
#include "mesh/SweepConstraints.H"
#include "mesh/TrajectorySurface.H"
#include "coupling/IntervalAudit.H"
#include "fvCFD.H"
#include "fixedValueFvPatchFields.H"
#include "zeroGradientFvPatchFields.H"
#include "fvmLaplacian.H"
#include "fvmSup.H"
#include <algorithm>
#include <stdexcept>
namespace chmt_test {
static unsigned long long materialSolveCalls=0,materialRhsCalls=0,filmCandidateCalls=0;
static chmt::Real lastMaterialInterval=0;
inline void resetMaterialObservations(){materialSolveCalls=materialRhsCalls=filmCandidateCalls=0;lastMaterialInterval=0;}
inline bool observedMaterialTransport(const chmt::HostMesh& mesh,const std::vector<chmt::SolidQ>& state,
    const chmt::PhysicsConfig& physics,const std::vector<chmt::Real>& sweep,chmt::Real interval,chmt::Real cfl,
    chmt::MaterialTransportResult& output,std::string& error,bool* numerical=nullptr){
    ++materialRhsCalls;lastMaterialInterval=interval;
    return chmt::evaluateMaterialTransport(mesh,state,physics,sweep,interval,cfl,output,error,numerical);
}
inline bool observedFilmCandidate(const chmt::HostState& state,const chmt::PhysicsConfig& physics,chmt::Real interval,
    const std::vector<chmt::CpuFilmForcing>& forcing,const std::vector<chmt::CpuFilmDrive>& drive,
    chmt::CpuFilmCandidate& output,std::string& error,bool* numerical=nullptr){
    ++filmCandidateCalls;return chmt::advanceCpuFilmCandidate(state,physics,interval,forcing,drive,output,error,numerical);
}
inline const Foam::dictionary& countedMaterialSolver(const Foam::dictionary& solver){
    ++materialSolveCalls;return solver;
}
}
#define solve(solver) solve(chmt_test::countedMaterialSolver(solver))
#define evaluateMaterialTransport(...) chmt_test::observedMaterialTransport(__VA_ARGS__)
#define advanceCpuFilmCandidate(...) chmt_test::observedFilmCandidate(__VA_ARGS__)
#include "materials/CpuMaterialDriver.C"
#undef solve
#undef evaluateMaterialTransport
#undef advanceCpuFilmCandidate
// Hexadecimal FP64 output is a comparison artifact, separate from production
// reports. It covers inventories, geometric state, packet payloads and clocks.
#include <cassert>
#include <iomanip>
#include <iostream>
namespace chmt_test {
// Independent finite-volume boundary integral for these orthogonal fixtures.
// Recovered endpoint temperatures and the specified boundary values supply the
// oracle; no production budget or report participates in this calculation.
inline long double boundaryHeat(const chmt::HostState& state,const chmt::PhysicsConfig& p,chmt::Real dt){
    const auto& mesh=state.solidMesh;long double heat=0;
    for(std::size_t f=0;f<mesh.owner.size();++f){
        if(mesh.neighbour[f]>=0||mesh.boundaryKind[f]==chmt::BoundaryKind::Interface
            ||mesh.thermalBoundary.empty()||mesh.thermalBoundary[f]!=chmt::ThermalBoundaryKind::FixedValue)continue;
        const auto c=mesh.owner[f];const auto& q=state.solid[c];chmt::MaterialPrimitive material;
        assert(chmt::recoverMaterial<true>(q,mesh.volumes[c],p,material));
        long double conductivity=0,occupied=0,pore=0;
        for(int j=0;j<chmt::Nc;++j){const long double volume=q.condensed[j]/p.condensed[j].rho;occupied+=volume;conductivity+=volume*p.condensed[j].conductivity/mesh.volumes[c];}
        for(auto mass:q.pore)pore+=mass;
        if(pore>0)conductivity+=(1-occupied/mesh.volumes[c])*p.gasConductivity;
        const chmt::Real area=chmt::mag(mesh.areaVectors[f]);
        const chmt::Real distance=chmt::dot(mesh.faceCentres[f]-mesh.cellCentres[c],mesh.areaVectors[f]/area);
        heat+=dt*conductivity*area/distance*(material.temperature-mesh.boundaryPrimitive[f].temperature);
    }
    return heat;
}
template<class T>inline void physicalValue(T value){std::cout<<value<<' ';}
inline void physicalValue(chmt::Vec3 value){physicalValue(value.x);physicalValue(value.y);physicalValue(value.z);}
template<class T>inline void physicalValues(const T& values){physicalValue(values.size());for(const auto& value:values)physicalValue(value);}
inline void physicalSolid(const chmt::SolidQ& q){
    for(auto value:q.condensed)physicalValue(value);for(auto value:q.pore)physicalValue(value);
    for(auto value:q.progress)physicalValue(value);physicalValue(q.energy);physicalValue(q.porosity);
}
inline void physicalFilm(const chmt::FilmQ& q){physicalValue(q.mass);physicalValue(q.enthalpy);for(auto value:q.species)physicalValue(value);}
inline void physicalMesh(const chmt::HostMesh& mesh){
    physicalValues(mesh.points);physicalValues(mesh.oldPoints);physicalValues(mesh.referencePoints);
    physicalValues(mesh.volumes);physicalValues(mesh.oldVolumes);physicalValues(mesh.cellCentres);
    physicalValues(mesh.faceCentres);physicalValues(mesh.areaVectors);physicalValues(mesh.meshVelocity);
    physicalValue(mesh.geometryVersion);physicalValue(mesh.topologyHash);
}
inline void physicalSnapshot(const char* label,const chmt::HostState& state,const chmt::WallProgram& program){
    std::cout<<std::hexfloat<<"CHMT_PHYSICAL "<<label<<' ';physicalValue(materialSolveCalls);physicalValue(materialRhsCalls);physicalValue(filmCandidateCalls);
    physicalValue(state.time);physicalValue(state.nextDt);physicalValue(state.lastAcceptedDt);
    physicalValue(state.acceptedSteps);physicalValue(state.rejectedSteps);physicalValue(state.commitSequence);
    physicalValue(state.solid.size());for(const auto& q:state.solid)physicalSolid(q);
    physicalValue(state.gas.size());for(const auto& q:state.gas){physicalValue(q.mass);physicalValue(q.momentum);physicalValue(q.energy);for(auto value:q.species)physicalValue(value);}
    physicalValue(state.film.size());for(const auto& q:state.film)physicalFilm(q);
    physicalValue(state.filmAux.size());for(const auto& a:state.filmAux){
        physicalValue(a.pressure);physicalValue(a.thickness);physicalValue(a.temperature);
        physicalValue(a.baseVelocity);physicalValue(a.meanVelocity);physicalValue(a.topVelocity);physicalValue(a.normal);
        physicalValue(a.area);physicalValue(a.kineticEnergy);physicalValue(a.normalVelocity);physicalValue(a.solidNormalVelocity);physicalValue(a.solidFront);physicalValue(a.gasFront);
    }
    physicalMesh(state.solidMesh);physicalMesh(state.gasMesh);physicalValues(state.solidSweepRemainder);
    physicalValues(state.gasVoidFraction);physicalValues(state.normalEnthalpy);physicalValues(state.normalLiquidFraction);physicalValues(state.normalAcceptedWallHeat);
    physicalValue(state.ledger.size());for(const auto& q:state.ledger){
        physicalValue(q.step);physicalValue(q.geometry);physicalValue(q.face);physicalValue(q.stage);physicalValue(int(q.kind));
        physicalValue(q.gasCell);physicalValue(q.solidCell);physicalValue(q.filmFace);physicalValue(q.particleIndex);
        physicalValue(q.mass);for(auto value:q.species)physicalValue(value);for(auto value:q.condensed)physicalValue(value);
        for(auto value:q.pore)physicalValue(value);for(auto value:q.poreSweep)physicalValue(value);
        // SolidFilm momentum was an unused support-force diagnostic; all other
        // packet momentum channels still feed real gas/particle recipients.
        if(q.kind!=chmt::ExchangeKind::SolidFilm)physicalValue(q.momentum);
        physicalValue(q.energy);physicalValue(q.conductive);physicalValue(q.advective);physicalValue(q.pressureWork);physicalValue(q.viscousWork);physicalValue(q.radiation);physicalValue(q.liquidKineticAdvection);physicalValue(q.consumerMask);
    }
    physicalValue(state.filmStorage.size());for(const auto& q:state.filmStorage){
        physicalValue(q.step);physicalValue(q.geometry);physicalValue(q.face);physicalValue(q.stage);physicalValue(q.filmFace);physicalValue(q.oldPV);physicalValue(q.newPV);physicalValue(q.consumed);
    }
    physicalValue(program.interval.identity.sequence);physicalValue(program.interval.identity.epoch);physicalValue(program.interval.begin);physicalValue(program.interval.end);
    physicalValue(program.knots.size());for(const auto& k:program.knots){
        physicalValue(k.time);physicalValue(k.localTime.sum);physicalValue(k.localTime.correction);physicalValues(k.gasPoints);physicalValues(k.solidPoints);
        physicalValue(k.faces.size());for(const auto& f:k.faces){
            physicalValue(f.temperature);physicalValue(f.velocity);physicalValue(int(f.primaryKind));for(auto value:f.speciesRate)physicalValue(value);
            for(auto value:f.condensedRate)physicalValue(value);for(auto value:f.poreRate)physicalValue(value);for(auto value:f.poreSweepRate)physicalValue(value);
            physicalValue(f.radiationFlux);physicalValue(f.solidNormalVelocity);physicalValue(f.normalVelocity);
        }
    }
    physicalValue(program.donorPlan.knots.size());for(const auto& k:program.donorPlan.knots){
        physicalValue(k.time);physicalValue(k.localTime.sum);physicalValue(k.localTime.correction);
        physicalValue(k.cumulativeSolid.size());for(const auto& q:k.cumulativeSolid)physicalSolid(q);
        physicalValue(k.cumulativeFilm.size());for(const auto& q:k.cumulativeFilm)physicalFilm(q);
    }
    std::cout<<'\n';
}
}
#endif
