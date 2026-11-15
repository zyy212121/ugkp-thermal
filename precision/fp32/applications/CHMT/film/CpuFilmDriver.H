#ifndef CHMT_FILM_CPUFILMDRIVER_H
#define CHMT_FILM_CPUFILMDRIVER_H
#include "core/HostState.H"
#include <string>
namespace chmt {



struct CpuFilmForcing {
    PhysicalReal mass=0,species[Ns]{};EnergyReal energy=0;
    EnergyReal interfaceKineticOutflow=0,radiationEnergy=0;

    PhysicalReal phaseMass=0,phaseSpecies[Ns]{};EnergyReal phaseEnergy=0,phaseKineticOutflow=0;
    PhysicalReal withdrawnMass=0,withdrawnSpecies[Ns]{};
    bool topWorkIncluded=false,bottomWorkIncluded=false;
    bool exactPhaseEvent=false;
};
struct CpuFilmDrive {
    PhysicalReal pressure=undefinedValue(),endpointPressure=undefinedValue();
    PhysicalVec3 shear{};
    bool hasCoupledNormalTrace=false;
    PhysicalReal bottomNormalVelocity=0,topNormalVelocity=0;
    PhysicalReal interfaceNormalVelocity=0,solidNormalVelocity=0;
};
struct CpuFilmCandidate {
    std::vector<FilmQ> film{};
    std::vector<FilmAux> filmAux{};
};





bool finalizeCpuFilmPhasePacket(const FilmQ& accepted,PhysicalReal acceptedPressure,
    const FilmQ& localRate,const CpuFilmForcing& primary,Real dt,PhysicalVec3 bottomVelocity,
    const PhysicsConfig&,ExchangePacket& phase,bool& exactEvent,std::string& error,bool* numericalFailure=nullptr);



bool advanceCpuFilmCandidate(const HostState&,const PhysicsConfig&,Real dt,
    const std::vector<CpuFilmForcing>&,const std::vector<CpuFilmDrive>&,
    CpuFilmCandidate&,std::string& error,bool* numericalFailure=nullptr);



bool evaluateCpuFilmTransport(const HostState&,const PhysicsConfig&,Real dt,
    const std::vector<CpuFilmDrive>&,std::vector<FilmQ>& rates,
    std::string& error,
    std::vector<FilmAux>* refreshedAux=nullptr,
    std::vector<PhysicalVec3>* pressureGradient=nullptr,
    std::vector<Real>* sideMeshVolumeRate=nullptr,bool* numericalFailure=nullptr);


bool estimateCpuFilmStep(const HostState&,const PhysicsConfig&,
    const std::vector<CpuFilmForcing>&,const std::vector<CpuFilmDrive>&,
    Real forcingInterval,Real maxDt,Real& dt,std::string& error,bool* numericalFailure=nullptr);

}
#endif
