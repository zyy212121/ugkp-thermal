#ifndef CHMT_FILM_CPUFILMDRIVER_H
#define CHMT_FILM_CPUFILMDRIVER_H
#include "core/HostState.H"
#include <string>
namespace chmt {
// Extensive accepted interval transfers, positive INTO the film. Energy excludes
// radiation, liquid kinetic advection, and the endpoint pV storage conversion.
// Gross withdrawal fields must be populated when opposite transfers are combined.
struct CpuFilmForcing {
    Real mass=0,species[Ns]{},energy=0;
    Real interfaceKineticOutflow=0,radiationEnergy=0;
    // SolidFilm is kept separate and applied last for exact remainder events.
    Real phaseMass=0,phaseSpecies[Ns]{},phaseEnergy=0,phaseKineticOutflow=0;
    Real withdrawnMass=0,withdrawnSpecies[Ns]{};
    bool topWorkIncluded=false,bottomWorkIncluded=false;
    bool exactPhaseEvent=false; // Phase OR gas depletion, accepted only after an exact-empty candidate proof.
};
struct CpuFilmDrive {
    Real pressure=undefinedValue(),endpointPressure=undefinedValue();
    Vec3 shear{};
    bool hasCoupledNormalTrace=false;
    Real bottomNormalVelocity=0,topNormalVelocity=0;
    Real interfaceNormalVelocity=0,solidNormalVelocity=0;
};
struct CpuFilmReport {
    Budget budgetDelta{};
    Real internalEnergyBefore=0,internalEnergyAfter=0;
    Real reducedEnergyResidual=0,kineticDefect=0,numericalMassRoundoff=0;
    bool phaseEvent=false;
    std::vector<FilmQ> edgeFlux{}; // Rates, owner-outward, evaluated once per edge.
    std::vector<FilmRateBudget> rateBudget{};
};
struct CpuFilmCandidate {
    std::vector<FilmQ> film{};
    std::vector<FilmAux> filmAux{};
    CpuFilmReport report{};
};
// Finalize the CPU-owned SolidFilm packet at an exactly empty film event,
// including depletion caused entirely by accepted gas packets. The remaining
// physical energy goes to/from the existing solid owner; gas packets are never
// changed. No near-empty tolerance or mass clipping is used. primary contains
// the accepted non-phase forcing. Outputs remain unchanged on failure.
bool finalizeCpuFilmPhasePacket(const FilmQ& accepted,Real acceptedPressure,
    const FilmQ& localRate,const CpuFilmForcing& primary,Real dt,Vec3 bottomVelocity,
    const PhysicsConfig&,ExchangePacket& phase,bool& exactEvent,std::string& error);
// One first-order conservative CPU substep. Inputs and output remain untouched
// on failure. The caller owns adaptive subcycling, event alignment and publication.
// Coupled wet faces require real pressure, traction and explicit normal traces.
bool advanceCpuFilmCandidate(const HostState&,const PhysicsConfig&,Real dt,
    const std::vector<CpuFilmForcing>&,const std::vector<CpuFilmDrive>&,
    CpuFilmCandidate&,std::string& error);
// Interface-owner RHS: suppresses top and bottom interface work and excludes all
// integrated transfers; retains edge transport, body work and standalone radiation.
// Rates/budgets are published together only after a successful evaluation.
bool evaluateCpuFilmTransport(const HostState&,const PhysicsConfig&,Real dt,
    const std::vector<CpuFilmDrive>&,std::vector<FilmQ>& rates,
    std::vector<FilmRateBudget>& budgets,std::string& error,
    std::vector<FilmAux>* refreshedAux=nullptr,
    std::vector<Vec3>* pressureGradient=nullptr,
    std::vector<Real>* sideMeshVolumeRate=nullptr);
// Local material restrictions only. forcing is integrated over forcingInterval;
// dt is also restricted to that supplied interval and maxDt. No gas CFL is used.
bool estimateCpuFilmStep(const HostState&,const PhysicsConfig&,
    const std::vector<CpuFilmForcing>&,const std::vector<CpuFilmDrive>&,
    Real forcingInterval,Real maxDt,Real& dt,std::string& error);
}
#endif
