#ifndef CHMT_CORE_TYPES_H
#define CHMT_CORE_TYPES_H

#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

// A generated header must define CHMT_NSPECIES, CHMT_SPECIES_NAMES (an
// initializer of string literals), and CHMT_SPECIES_HASH (uint64 FNV-1a).
#ifdef CHMT_SPECIES_HEADER
#include CHMT_SPECIES_HEADER
#endif
#ifndef CHMT_NSPECIES
#define CHMT_NSPECIES 2
#endif
#ifndef CHMT_SPECIES_NAMES
#if CHMT_NSPECIES != 2
#error "Non-default species count requires a generated CHMT_SPECIES_HEADER"
#endif
#define CHMT_SPECIES_NAMES {"S0", "S1"}
#endif
#ifdef __CUDACC__
#define CHMT_HD __host__ __device__
#else
#define CHMT_HD
#endif

namespace chmt {
using Real = double;
constexpr int Ns=CHMT_NSPECIES, Nc=2, Nr=8, Ne=8;
static_assert(Ns>0, "At least one compiled gas species is required");
constexpr Real universalGasConstant=8.31446261815324; // J/(mol K)
constexpr const char* compiledSpeciesNames[Ns]=CHMT_SPECIES_NAMES;
constexpr std::uint64_t speciesNameHash() {
    std::uint64_t h=14695981039346656037ULL;
    for (int s=0;s<Ns;++s) {
        for (int i=0;compiledSpeciesNames[s][i];++i) {
            h^=static_cast<unsigned char>(compiledSpeciesNames[s][i]); h*=1099511628211ULL;
        }
        h^=0; h*=1099511628211ULL; // NUL delimits names, including order.
    }
    return h;
}
#ifdef CHMT_SPECIES_HASH
constexpr std::uint64_t compiledSpeciesHash=CHMT_SPECIES_HASH;
static_assert(compiledSpeciesHash==speciesNameHash(),"Generated species hash/name order mismatch");
#else
#ifdef CHMT_SPECIES_HEADER
#error "Generated CHMT_SPECIES_HEADER must supply an explicit CHMT_SPECIES_HASH"
#endif
constexpr std::uint64_t compiledSpeciesHash=speciesNameHash();
#endif
// Schema 3 persists thermal policy independently of EOS-valid boundary values.
constexpr std::uint64_t checkpointSchema=3;

CHMT_HD inline bool finite(Real x) { return x==x && x<=std::numeric_limits<Real>::max() && x>=-std::numeric_limits<Real>::max(); }
CHMT_HD inline Real absValue(Real x) { return x<0 ? -x : x; }
CHMT_HD inline Real minValue(Real a,Real b) { return a<b?a:b; }
CHMT_HD inline Real maxValue(Real a,Real b) { return a>b?a:b; }
CHMT_HD inline Real undefinedValue() { return std::numeric_limits<Real>::quiet_NaN(); }
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

struct GasQ {
    Real mass=0; Vec3 momentum{}; Real energy=0; Real species[Ns]{};
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
CHMT_HD inline GasQ operator*(const GasQ& a,Real b) {
    GasQ q{};q.mass=a.mass*b;q.momentum=a.momentum*b;q.energy=a.energy*b;
    for(int s=0;s<Ns;++s)q.species[s]=a.species[s]*b;
    return q;
}
CHMT_HD inline GasQ operator*(Real b,const GasQ& a) { return a*b; }
CHMT_HD inline GasQ& operator+=(GasQ& a,const GasQ& b) { a=a+b;return a; }
struct GasPrimitive { Real rho=0;Vec3 velocity{};Real temperature=0,pressure=0,soundSpeed=0;Real Y[Ns]{}; };
struct GasGradient { Vec3 rho{},velocity[3]{},temperature{},pressure{},Y[Ns]{}; };
struct SolidQ { Real condensed[Nc]{},progress[Nr]{},pore[Ns]{},energy=0,porosity=0; };
struct FilmRateBudget {
    Real bodyPower=0,supportPower=0,prescribedTopPower=0,radiationPower=0;
    Real edgeEnergyOutflow=0,edgeKineticOutflow=0,interfaceKineticOutflow=0,dissipationPower=0;
}; // W: external powers input-positive, edge/interface fluxes outward-positive.
struct FilmQ { Real mass=0,enthalpy=0,species[Ns]{}; }; // true H [J], not h_ref inventory.
struct FilmAux {
    Real pressure=0,thickness=0,temperature=0;
    Vec3 baseVelocity{},meanVelocity{},topVelocity{},normal{};
    Real area=0,kineticEnergy=0,normalVelocity=0,solidNormalVelocity=0,solidFront=0,gasFront=0;
    // normalVelocity/solidNormalVelocity: physical outward m/s. Fronts: accumulated normal displacement m.
};
struct ParticleQ {
    Vec3 contactRelativeVelocity{}; // captured wall-relative incoming velocity, m/s.
    Real contactMechanicalEnergy=0; // ideal elastic dwell reservoir J, separate from thermal energy.
    std::uint64_t id=0;int cell=-1,face=-1,state=0;Vec3 position{},velocity{};
    Real mass=0,energy=0,diameter=0,contactAge=0,contactDuration=0,contactArea=0;
    std::uint64_t rng=0;
};
struct SstQ { Real rhoK=0,rhoOmega=0; }; // cell integrals [J] and [kg/s].
struct SstPrimitive {
    Real k=0,omega=0,eddyViscosity=0,blend1=0,blend2=0;
    // k [m^2/s^2], omega [1/s], eddyViscosity=nut [m^2/s], blends dimensionless.
    // Mean stress uses mu_t=rho*nut; modeled k does not enter GasQ.energy.
};
struct SpeciesThermo { Real R=0,cp0=0,cp1=0,e0=0,Tmin=0,Tmax=0,element[Ne]{}; };
struct CondensedThermo { Real rho=0,cp0=0,cp1=0,e0=0,conductivity=0,Tmin=0,Tmax=0,element[Ne]{}; };
struct Reaction {
    Real A=0,temperaturePower=0,activationEnergy=0,condensedNu[Nc]{},gasNu[Ns]{},order[Nc]{};
    // nu kg/mol extent; order acts on condensed kg/m^3, fixing A's units.
};
enum class BoundaryKind { Internal,Periodic,Slip,NoSlip,Inlet,Outlet,Interface,Empty };
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
    Real mass=0,species[Ns]{},condensed[Nc]{},pore[Ns]{},poreSweep[Ns]{};Vec3 momentum{};
    Real energy=0,conductive=0,advective=0,pressureWork=0,viscousWork=0,radiation=0,liquidKineticAdvection=0;
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
    Real oldPV=0,newPV=0;bool consumed=false;
    // Storage conversion [J], once per face/stage AFTER all physical packets.
};
struct Tolerances {
    Real absoluteMass=1e-14,relativeMass=1e-10,absoluteEnergy=1e-10,relativeEnergy=1e-10;
    Real absoluteTemperature=1e-10,relativeTemperature=1e-12,absoluteGeometry=1e-14,relativeGeometry=1e-10;
    int maxCouplingIterations=30,maxRetries=12;
};
struct Budget {
    Real boundaryMass=0,boundarySpecies[Ns]{},boundaryElements[Ne]{},boundaryEnergy=0;
    Vec3 boundaryMomentum{},supportImpulse{};
    Real supportWork=0,bodyWork=0,radiation=0,gclResidual=0;
    Real exchangeMass[NParticipants]{},exchangeSpecies[NParticipants][Ns]{},exchangeEnergy[NParticipants]{};
    Vec3 exchangeMomentum[NParticipants]{};
    Real filmPressureVolume=0,filmKineticAdvection=0,filmKineticStorage=0,filmReducedResidual=0,filmKineticDefect=0;
    Real thinLayerGeometryDefect=0,thinLayerStressDefect=0,numericalEnergyResidual=0;
    Real turbulenceOmegaConstraint=0; // accepted omega-only constraint inventory changes [kg/s].
    Real turbulenceInventory=0,turbulenceProduction=0,turbulenceDissipation=0,turbulenceBoundaryFlux=0;
    std::uint64_t consumedPackets=0;
};
static_assert(std::is_trivially_copyable<GasQ>::value,"GasQ device ABI");
static_assert(std::is_trivially_copyable<ExchangePacket>::value,"ExchangePacket device ABI");
static_assert(std::is_trivially_copyable<Budget>::value,"Budget device ABI");
} // namespace chmt
#endif
