#ifndef CHMT_FILM_CPUFILMDRIVER_H
#define CHMT_FILM_CPUFILMDRIVER_H
#include "core/HostState.H"
#include <string>
namespace chmt {



struct CpuFilmForcing {
    Real mass=0,species[Ns]{},energy=0;
    Real interfaceKineticOutflow=0,radiationEnergy=0;

    Real phaseMass=0,phaseSpecies[Ns]{},phaseEnergy=0,phaseKineticOutflow=0;
    Real withdrawnMass=0,withdrawnSpecies[Ns]{};
    bool topWorkIncluded=false,bottomWorkIncluded=false;
    bool exactPhaseEvent=false;
};
struct CpuFilmDrive {
    Real pressure=undefinedValue(),endpointPressure=undefinedValue();
    Vec3 shear{};
    bool hasCoupledNormalTrace=false;
    Real bottomNormalVelocity=0,topNormalVelocity=0;
    Real interfaceNormalVelocity=0,solidNormalVelocity=0;
};
struct CpuFilmCandidate {
    std::vector<FilmQ> film{};
    std::vector<FilmAux> filmAux{};
};





bool finalizeCpuFilmPhasePacket(const FilmQ& accepted,Real acceptedPressure,
    const FilmQ& localRate,const CpuFilmForcing& primary,Real dt,Vec3 bottomVelocity,
    const PhysicsConfig&,ExchangePacket& phase,bool& exactEvent,std::string& error,bool* numericalFailure=nullptr);



bool advanceCpuFilmCandidate(const HostState&,const PhysicsConfig&,Real dt,
    const std::vector<CpuFilmForcing>&,const std::vector<CpuFilmDrive>&,
    CpuFilmCandidate&,std::string& error,bool* numericalFailure=nullptr);



bool evaluateCpuFilmTransport(const HostState&,const PhysicsConfig&,Real dt,
    const std::vector<CpuFilmDrive>&,std::vector<FilmQ>& rates,
    std::string& error,
    std::vector<FilmAux>* refreshedAux=nullptr,
    std::vector<Vec3>* pressureGradient=nullptr,
    std::vector<Real>* sideMeshVolumeRate=nullptr,bool* numericalFailure=nullptr);


bool estimateCpuFilmStep(const HostState&,const PhysicsConfig&,
    const std::vector<CpuFilmForcing>&,const std::vector<CpuFilmDrive>&,
    Real forcingInterval,Real maxDt,Real& dt,std::string& error,bool* numericalFailure=nullptr);

}
#endif
