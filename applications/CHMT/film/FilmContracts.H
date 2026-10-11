#ifndef CHMT_FILM_FILMCONTRACTS_H
#define CHMT_FILM_FILMCONTRACTS_H
#include "film/FilmMath.H"
#include <string>
namespace chmt {
constexpr int FilmContractMaxSamples=16;
struct FilmProfileRequest {
    Real delta=0,mu=0,rho=0,h=0;
    Vec3 baseVelocity{},topShear{},pressureBodyGradient{};
    int nSamples=0;
    Real z[FilmContractMaxSamples]{};
};
struct FilmProfileResult {
    Vec3 meanVelocity{},topVelocity{},bottomShear{};
    Vec3 volumeFlux{},massFlux{},enthalpyFlux{};
    Real dissipation=0,boundaryWork=0,kineticEnergyPerArea=0;
    int nSamples=0;
    Vec3 velocity[FilmContractMaxSamples]{};
};
struct FilmPressureRequest {
    FilmQ state{};
    Real area=0,oldPressure=0,newPressure=0;
    PhysicsConfig physics{};
};
struct FilmPressureResult {
    FilmQ state{};
    FilmAux aux{};
    Real internalEnergy=0,pressureVolumeDelta=0;
};
struct FilmPhaseRequest {
    Real massFlux=0,liquidEnthalpy=0,gasEnthalpy=0;
    Real pressure=0,normalSpeed=0;
    Vec3 normal{0,0,1},liquidVelocity{},gasVelocity{};
    Real liquidViscousWork=0,gasViscousWork=0,gasConductiveFlux=0;
};
struct FilmPhaseResult {
    Real gasEnergyFlux=0,liquidHeatInput=0,enthalpyRhs=0;
    Real liquidDensity=0,gasDensity=0;
};
bool evaluateFilmProfileContract(const FilmProfileRequest&,FilmProfileResult&,std::string& error);
bool evaluateFilmPressureContract(const FilmPressureRequest&,FilmPressureResult&,std::string& error);
bool evaluateFilmPhaseContract(const FilmPhaseRequest&,FilmPhaseResult&,std::string& error);
CHMT_HD inline bool evaluateFilmContractMath(const FilmProfileRequest& in, FilmProfileResult& out) {
    if (!finite(in.rho) || in.rho<=0 || !finite(in.h) || in.nSamples<0
        || in.nSamples>FilmContractMaxSamples) return false;
    const FilmProfile p=filmProfile(in.delta,in.mu,in.baseVelocity,in.topShear,in.pressureBodyGradient);
    if (!finite(p.dissipation) || !finite(p.kineticPerArea)) return false;
    FilmProfileResult next;
    next.meanVelocity=p.meanVelocity; next.topVelocity=p.topVelocity; next.bottomShear=p.bottomShear;
    next.volumeFlux=p.meanVelocity*in.delta;
    next.massFlux=next.volumeFlux*in.rho;
    next.enthalpyFlux=next.massFlux*in.h;
    next.dissipation=p.dissipation;
    // Boundary traction work; pressure/body-gradient work is a separate term.
    next.boundaryWork=dot(in.topShear,p.topVelocity)-dot(p.bottomShear,in.baseVelocity);
    next.kineticEnergyPerArea=in.rho*p.kineticPerArea;
    next.nSamples=in.nSamples;
    for (int i=0; i<in.nSamples; ++i) {
        if (!finite(in.z[i]) || in.z[i]<0 || in.z[i]>in.delta) return false;
        next.velocity[i]=profileVelocity(p,in.z[i]);
    }
    out=next; return true;
}
CHMT_HD inline bool evaluateFilmContractMath(const FilmPressureRequest& in, FilmPressureResult& out) {
    FilmAux oldAux;
    if (!recoverFilm(in.state,in.area,in.oldPressure,in.physics,oldAux)) return false;
    FilmPressureVolumeUpdate storage;
    storage.oldPV=in.oldPressure*in.state.mass/in.physics.liquid.rho;
    storage.newPV=in.newPressure*in.state.mass/in.physics.liquid.rho;
    FilmPressureResult next;
    next.state=in.state;
    next.pressureVolumeDelta=filmPressureVolumeDelta(storage);
    next.state.enthalpy+=next.pressureVolumeDelta;
    if (!recoverFilm(next.state,in.area,in.newPressure,in.physics,next.aux)) return false;
    next.internalEnergy=filmInternalEnergy(next.state,in.newPressure,in.physics);
    if (!finite(next.internalEnergy)) return false;
    out=next; return true;
}
CHMT_HD inline bool evaluateFilmContractMath(const FilmPhaseRequest& in, FilmPhaseResult& out) {
    const Real values[]={in.massFlux,in.liquidEnthalpy,in.gasEnthalpy,in.pressure,
        in.normalSpeed,in.liquidViscousWork,in.gasViscousWork,in.gasConductiveFlux};
    for (Real value:values) if (!finite(value)) return false;
    if (!finite(in.normal) || !finite(in.liquidVelocity) || !finite(in.gasVelocity)
        || !closeEnough(mag(in.normal),1,1e-13,1e-13)) return false;
    const Real ul=dot(in.liquidVelocity,in.normal)-in.normalSpeed;
    const Real ug=dot(in.gasVelocity,in.normal)-in.normalSpeed;
    if (ul==0 || ug==0 || in.massFlux/ul<=0 || in.massFlux/ug<=0) return false;
    FilmPhaseResult next;
    next.gasEnergyFlux=physicalInterfaceEnergy(in.massFlux,in.massFlux*in.gasEnthalpy,
        in.gasVelocity,in.pressure,in.normalSpeed,in.gasViscousWork,in.gasConductiveFlux);
    next.liquidHeatInput=liquidInterfaceHeat(in.massFlux,in.liquidEnthalpy,in.liquidVelocity,
        in.pressure,in.normalSpeed,in.liquidViscousWork,next.gasEnergyFlux);
    next.enthalpyRhs=next.liquidHeatInput-in.massFlux*in.liquidEnthalpy;
    next.liquidDensity=in.massFlux/ul; next.gasDensity=in.massFlux/ug;
    if (!finite(next.gasEnergyFlux) || !finite(next.enthalpyRhs)) return false;
    out=next; return true;
}
}
#endif
