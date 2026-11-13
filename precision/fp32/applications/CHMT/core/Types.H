#ifndef CHMT_CORE_TYPES_H
#define CHMT_CORE_TYPES_H

#include <cmath>
#include "../../../common/GpuPrecisionTypes.H"
#include "../../../common/gasBoundary/Definition.H"
#include "../../../common/thermal/ParticleState.H"
#include <cstdint>
#include <limits>
#include <type_traits>
#include "../../../common/gasTransport/GasModel.H"

#include "../../../common/gasTransport/GasBuildConfig.H"

#ifdef __CUDACC__
#define CHMT_HD __host__ __device__
#else
#define CHMT_HD
#endif

namespace chmt {
using Real = double;
using EnergyReal = GpuEnergy;
using PhysicalReal = GpuReal;
constexpr int Ns=ugkwp::compiledGasSpecies, Nc=2, Nr=8, Ne=8;
constexpr PhysicalReal universalGasConstant=PhysicalReal(8.31446261815324); // J/(mol K)
// Default labels exist only for synthetic component fixtures. The production
// species identity is the immutable runtime model name/order hash.
constexpr std::uint64_t defaultSpeciesNameHash() {
    std::uint64_t h=14695981039346656037ULL;
    for(int s=0;s<Ns;++s){
        h^=static_cast<unsigned char>('S');h*=1099511628211ULL;
        int divisor=1;while(s/divisor>=10)divisor*=10;
        for(;divisor;divisor/=10){h^=static_cast<unsigned char>('0'+(s/divisor)%10);h*=1099511628211ULL;}
        h*=1099511628211ULL;
    }
    return h;
}
constexpr std::uint64_t defaultSpeciesHash=defaultSpeciesNameHash();
// Schema 10 appends complete common particle thermal/contact history.
// Schemas 6--9 remain readable; absent history is explicitly uninitialized.
// Borrowed shared-thermo views are rebuilt and never serialized.
constexpr std::uint64_t checkpointSchema=14;

CHMT_HD inline bool finite(Real x) { return x==x && x<=std::numeric_limits<Real>::max() && x>=-std::numeric_limits<Real>::max(); }
CHMT_HD inline Real absValue(Real x) { return x<0 ? -x : x; }
CHMT_HD inline Real minValue(Real a,Real b) { return a<b?a:b; }
CHMT_HD inline Real maxValue(Real a,Real b) { return a>b?a:b; }
template<class T=PhysicalReal> CHMT_HD inline T undefinedValue() { return std::numeric_limits<T>::quiet_NaN(); }
CHMT_HD inline bool finite(PhysicalReal x){return std::isfinite(x);}
CHMT_HD inline PhysicalReal absValue(PhysicalReal x){return x<PhysicalReal(0)?-x:x;}
CHMT_HD inline PhysicalReal minValue(PhysicalReal a,PhysicalReal b){return a<b?a:b;}
CHMT_HD inline PhysicalReal maxValue(PhysicalReal a,PhysicalReal b){return a>b?a:b;}
struct Vec3 { Real x=0,y=0,z=0; };
CHMT_HD inline Vec3 operator+(Vec3 a,Vec3 b) { return {a.x+b.x,a.y+b.y,a.z+b.z}; }
CHMT_HD inline Vec3 operator-(Vec3 a,Vec3 b) { return {a.x-b.x,a.y-b.y,a.z-b.z}; }
CHMT_HD inline Vec3 operator-(Vec3 a) { return {-a.x,-a.y,-a.z}; }
CHMT_HD inline Vec3 operator*(Vec3 a,Real b) { return {a.x*b,a.y*b,a.z*b}; }
CHMT_HD inline Vec3 operator*(Real b,Vec3 a) { return a*b; }
CHMT_HD inline Vec3 operator/(Vec3 a,Real b) { return a*(1/b); }
CHMT_HD inline Vec3& operator+=(Vec3& a,Vec3 b) { a=a+b;return a; }
CHMT_HD inline Vec3& operator-=(Vec3& a,Vec3 b) { a=a-b;return a; }
CHMT_HD inline Real dot(Vec3 a,Vec3 b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
CHMT_HD inline Vec3 cross(Vec3 a,Vec3 b) { return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x}; }
CHMT_HD inline Real mag(Vec3 a) { return ::sqrt(dot(a,a)); }
CHMT_HD inline bool finite(Vec3 a) { return finite(a.x)&&finite(a.y)&&finite(a.z); }
CHMT_HD inline Vec3 normalized(Vec3 a) { const Real m=mag(a);return m>0&&finite(m)?a/m:Vec3{}; }

struct PhysicalVec3 { PhysicalReal x=0,y=0,z=0; };
CHMT_HD inline PhysicalVec3 toPhysical(Vec3 v){return {PhysicalReal(v.x),PhysicalReal(v.y),PhysicalReal(v.z)};}
CHMT_HD inline Vec3 toGeometry(PhysicalVec3 v){return {Real(v.x),Real(v.y),Real(v.z)};}
CHMT_HD inline PhysicalVec3 operator+(PhysicalVec3 a,PhysicalVec3 b){return {a.x+b.x,a.y+b.y,a.z+b.z};}
CHMT_HD inline PhysicalVec3 operator-(PhysicalVec3 a,PhysicalVec3 b){return {a.x-b.x,a.y-b.y,a.z-b.z};}
CHMT_HD inline PhysicalVec3 operator-(PhysicalVec3 a){return {-a.x,-a.y,-a.z};}
CHMT_HD inline PhysicalVec3 operator*(PhysicalVec3 a,PhysicalReal b){return {a.x*b,a.y*b,a.z*b};}
CHMT_HD inline PhysicalVec3 operator*(PhysicalReal b,PhysicalVec3 a){return a*b;}
CHMT_HD inline PhysicalVec3 operator/(PhysicalVec3 a,PhysicalReal b){return {a.x/b,a.y/b,a.z/b};}
CHMT_HD inline PhysicalVec3& operator+=(PhysicalVec3& a,PhysicalVec3 b){a=a+b;return a;}
CHMT_HD inline PhysicalVec3& operator-=(PhysicalVec3& a,PhysicalVec3 b){a=a-b;return a;}
CHMT_HD inline PhysicalVec3& operator*=(PhysicalVec3& a,PhysicalReal b){a=a*b;return a;}
CHMT_HD inline PhysicalVec3& operator/=(PhysicalVec3& a,PhysicalReal b){a=a/b;return a;}
CHMT_HD inline PhysicalReal dot(PhysicalVec3 a,PhysicalVec3 b){return a.x*b.x+a.y*b.y+a.z*b.z;}
CHMT_HD inline PhysicalVec3 cross(PhysicalVec3 a,PhysicalVec3 b){return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}
CHMT_HD inline PhysicalReal mag(PhysicalVec3 a){return std::sqrt(dot(a,a));}
CHMT_HD inline bool finite(PhysicalVec3 a){return finite(a.x)&&finite(a.y)&&finite(a.z);}
CHMT_HD inline PhysicalVec3 normalized(PhysicalVec3 a){const PhysicalReal m=mag(a);return m>0&&finite(m)?a/m:PhysicalVec3{};}

struct GasQ {
    PhysicalReal mass=0; PhysicalVec3 momentum{}; EnergyReal energy=0; PhysicalReal species[Ns]{};
    // kg, kg m/s, J, kg; energy = thermal + resolved mean kinetic only.
    // SST rho*k is separate and is never added/subtracted by gas EOS recovery.
};
CHMT_HD inline GasQ operator+(const GasQ& a,const GasQ& b) {
    GasQ q{};q.mass=a.mass+b.mass;q.momentum=a.momentum+b.momentum;q.energy=a.energy+b.energy;
    for(int s=0;s<Ns;++s)q.species[s]=a.species[s]+b.species[s];
    return q;
}
CHMT_HD inline GasQ operator-(const GasQ& a,const GasQ& b) {
    GasQ q{};q.mass=a.mass-b.mass;q.momentum=a.momentum-b.momentum;q.energy=a.energy-b.energy;
    for(int s=0;s<Ns;++s)q.species[s]=a.species[s]-b.species[s];
    return q;
}
CHMT_HD inline GasQ operator*(const GasQ& a,PhysicalReal b) {
    GasQ q{};q.mass=a.mass*b;q.momentum=a.momentum*b;q.energy=a.energy*EnergyReal(b);
    for(int s=0;s<Ns;++s)q.species[s]=a.species[s]*b;
    return q;
}
CHMT_HD inline GasQ operator*(PhysicalReal b,const GasQ& a) { return a*b; }
CHMT_HD inline GasQ& operator+=(GasQ& a,const GasQ& b) { a=a+b;return a; }
struct GasPrimitive { PhysicalReal rho=0;PhysicalVec3 velocity{};EnergyReal temperature=0,pressure=0,soundSpeed=0;PhysicalReal Y[Ns]{}; };
struct GasGradient { PhysicalVec3 rho{},velocity[3]{},temperature{},pressure{},Y[Ns]{}; };
struct SolidQ { PhysicalReal condensed[Nc]{},progress[Nr]{},pore[Ns]{},porosity=0; EnergyReal energy=0; };
struct FilmQ { PhysicalReal mass=0,species[Ns]{}; EnergyReal enthalpy=0; }; // true H [J], not h_ref inventory.
struct FilmAux {
    PhysicalReal pressure=0; Real thickness=0; EnergyReal temperature=0;
    PhysicalVec3 baseVelocity{},meanVelocity{},topVelocity{},normal{};
    Real area=0,solidFront=0,gasFront=0;
    PhysicalReal normalVelocity=0,solidNormalVelocity=0;
    // normalVelocity/solidNormalVelocity: physical outward m/s. Fronts: accumulated normal displacement m.
};
using ParticleQ=ugkp::thermal::ParticleState<Vec3,PhysicalVec3>;

struct SstQ { EnergyReal rhoK=0; PhysicalReal rhoOmega=0; }; // cell integrals [J] and [kg/s].
struct SstPrimitive {
    PhysicalReal k=0,omega=0,eddyViscosity=0,blend1=0,blend2=0;
    // k [m^2/s^2], omega [1/s], eddyViscosity=nut [m^2/s], blends dimensionless.
    // Mean stress uses mu_t=rho*nut; modeled k does not enter GasQ.energy.
};
struct SpeciesThermo {
    // Historical LinearCp fields retain their order and default meaning.
    PhysicalReal R=0,cp0=0,cp1=0,e0=0,Tmin=0,Tmax=0,element[Ne]{};
    ugkwp::SpeciesThermoModel model=ugkwp::SpeciesThermoModel::LinearCp;
    PhysicalReal Tmid=0,nasa7[14]{};
    PhysicalReal referencePressure=101325,referenceTemperature=0,referenceEntropy=0;
    bool hasEntropyReference=false;
};
struct CondensedThermo { PhysicalReal rho=0,cp0=0,cp1=0,e0=0,conductivity=0,Tmin=0,Tmax=0,element[Ne]{}; };
struct Reaction {
    PhysicalReal A=0,temperaturePower=0,activationEnergy=0,condensedNu[Nc]{},gasNu[Ns]{},order[Nc]{};
    // nu kg/mol extent; order acts on condensed kg/m^3, fixing A's units.
};
using BoundaryKind = ugkp::boundary::Kind;
enum class ThermalBoundaryKind { ZeroGradient,FixedValue };
CHMT_HD inline bool fixedTemperature(const ThermalBoundaryKind* policy,int face) {
    return policy && policy[face]==ThermalBoundaryKind::FixedValue;
}
enum class ExchangeKind { GasSolid,GasFilm,SolidFilm,PoreGas,ParticleGas,ParticleWall };
enum class FilmThermalMode { ThicknessAveraged,ResolvedNormal };
enum class ReconstructionMode { LimitedLinear,SmoothVerification };
enum class ErrorLocation { None,Cell,Face,Reaction,Configuration,Layer,Particle };
enum class ErrorCode { Success=0,InvalidInput=1,Composition=2,PropertyRange=3,Inventory=4,Duplicate=5,Packet=6,Unsupported=7,Nonfinite=8 };
struct DeviceStatus { int code=0,index=-1;Real value=0;ErrorLocation location=ErrorLocation::None; };
CHMT_HD inline bool failStatus(DeviceStatus* out,ErrorCode code,int index,Real value,ErrorLocation location) {
    if(out){out->code=static_cast<int>(code);out->index=index;out->value=value;out->location=location;}return false;
}
constexpr unsigned ConsumeGas=1u,ConsumeSolid=2u,ConsumeFilm=4u,ConsumeParticle=8u;
constexpr int GasParticipant=0,SolidParticipant=1,FilmParticipant=2,ParticleParticipant=3,NParticipants=4;
struct ExchangePacket {
    std::uint64_t step=0,geometry=0,face=0;int stage=0;ExchangeKind kind=ExchangeKind::GasSolid;
    int gasCell=-1,solidCell=-1,filmFace=-1,particleIndex=-1;
    PhysicalReal mass=0,species[Ns]{},condensed[Nc]{},pore[Ns]{},poreSweep[Ns]{};PhysicalVec3 momentum{};
    EnergyReal energy=0,conductive=0,advective=0,pressureWork=0,viscousWork=0,radiation=0,liquidKineticAdvection=0;
    unsigned consumerMask=0;
    // poreSweep is the PoreGas geometric mass component [kg], NOT another inventory.
    // Physical Darcy amount is pore-poreSweep; counterflow may exceed the net.
    // Positive mass/species/channels leave condensed/pore/particle donor.
    // Negative individual reaction channels are allowed; net mass must match.
    // energy alone updates gas; decomposition is audit-only. Liquid k advection
    // is signed OUTWARD from film, including a negative solid->film entry.
};
struct FilmPressureVolumeUpdate {
    std::uint64_t step=0,geometry=0,face=0;int stage=0,filmFace=-1;
    EnergyReal oldPV=0,newPV=0;bool consumed=false;
    // Storage conversion [J], once per face/stage AFTER all physical packets.
};
struct Tolerances {
    ugkp::diagnostics::DeviceContext* diagnostics=nullptr;
    PhysicalReal absoluteMass=PhysicalReal(1e-14),relativeMass=PhysicalReal(1e-10),absoluteEnergy=PhysicalReal(1e-10),relativeEnergy=PhysicalReal(1e-10);
    PhysicalReal absoluteTemperature=PhysicalReal(1e-10),relativeTemperature=PhysicalReal(1e-12);
    Real absoluteGeometry=1e-14,relativeGeometry=1e-10;
    int maxCouplingIterations=30,maxRetries=12;
};
struct Budget {
    PhysicalReal boundaryMass=0,boundarySpecies[Ns]{},boundaryElements[Ne]{};
    EnergyReal boundaryEnergy=0;
    PhysicalVec3 boundaryMomentum{};
    PhysicalVec3 supportImpulse{};
    EnergyReal supportWork=0,bodyWork=0,radiation=0; Real gclResidual=0;
    PhysicalReal exchangeMass[NParticipants]{},exchangeSpecies[NParticipants][Ns]{}; EnergyReal exchangeEnergy[NParticipants]{};
    PhysicalVec3 exchangeMomentum[NParticipants]{};
    EnergyReal filmPressureVolume=0,filmKineticAdvection=0,filmKineticStorage=0,filmReducedResidual=0,filmKineticDefect=0;
    Real thinLayerGeometryDefect=0,thinLayerStressDefect=0; EnergyReal numericalEnergyResidual=0;
    PhysicalReal turbulenceOmegaConstraint=0;
    EnergyReal turbulenceInventory=0,turbulenceProduction=0,turbulenceDissipation=0,turbulenceBoundaryFlux=0;
    std::uint64_t consumedPackets=0;
};
} // namespace chmt
#endif
