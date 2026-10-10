#pragma once
#include "../GpuPrecisionTypes.H"
#include <type_traits>
#include "../gasTransport/MixtureThermo.H"
#include "../chemistry/GasRates.H"
#include "../gasNumerics/GpuSstAlgebra.cuh"

namespace ugkwp { namespace gaswall {
using WallReal=double;
static_assert(sizeof(WallReal)==8,"wall operator arithmetic requires binary64");
enum class WallFamily : int { LowRe=0, WallFunction=1, BoundaryLayer=2 };
enum class BoundaryLayerModel : int { ConstantTransport=0, ReactingSst=1, FiniteRate=ReactingSst };
enum class WallCode : int { Success=0, InvalidInput, InvalidThermodynamics, Unsupported,
    InvalidGeometry, ChemistryFailure, NonConvergence, SingularJacobian, InadmissibleState };
struct WallStatus { WallCode code=WallCode::Success; int node=-1,iteration=0; double residual=std::numeric_limits<double>::quiet_NaN(); };
template<class R> struct WallModelConfig {
    WallFamily family=WallFamily::BoundaryLayer;
    BoundaryLayerModel model=BoundaryLayerModel::ReactingSst;
    int nodes=24,maxIterations=60;
    R relativeTolerance=R(1e-8),absoluteTolerance=R(1e-10),stretch=R(2);
    R turbulentPrandtl=R(.9),turbulentSchmidt=R(.7);
    bool enableSst=false;
};
template<class R,int N> struct WallMatchingState {
    R temperature=0,velocity[3]{},massFraction[N]{},k=0,omega=0;
};
template<class R,int N,class TableReal=R> struct WallGasModel {
    SpeciesThermoView<TableReal,N> thermo;
    GasMechanismView<TableReal,N> mechanism;
    GasMode mode=GasMode::MixtureFrozen;
    R viscosity=0,conductivity=0,diffusivity[N]{},molecularPrandtl=0;
    // molecularPrandtl>0 selects mu*Cp(local T,Y)/Pr; zero uses conductivity.
    SstCoefficientsT<R> sstCoefficients=defaultSstCoefficientsT<R>();
};
template<class R> struct WallOwnerQuadrature {
    const R* distance=nullptr;
    const R* volumeWeight=nullptr;
    int count=0;
    R volume=0,firstMoment=0; // m^3 and integral y dV, from actual owner geometry.
};
template<class R,int N,class TableReal=R> struct WallInput {
    R pressure=0,temperature=0,massFlux[N]{},velocity[3]{},normal[3]{};
    R matchingDistance=0,ownerDistance=0;
    WallMatchingState<R,N> matching;
    WallGasModel<R,N,TableReal> model;
    WallOwnerQuadrature<TableReal> quadrature;
    // Optional coarse chemistry integrated per wall area over the same domain.
    // Absence sets chemistryMismatchAvailable=false and mismatch values to NaN.
    const TableReal* coarseReactionIntegral=nullptr;
};
template<class R,int N> struct WallOutput {
    R traceTemperature=0,traceDensity=0,traceMassFraction[N]{},traceVelocity[3]{};
    R conductiveHeatFlux=0,traction[3]{},wallSpeciesFlux[N]{},matchingSpeciesFlux[N]{};
    R reactionIntegral[N]{},chemistryMismatch[N]{},speciesBalanceResidual[N]{};
    R integratedKSource=0,ownerOmega=0,wallKFlux=0;
    bool chemistryMismatchAvailable=false;
    int iterations=0;
    R residual=0;
};
// Caller-owned FP64 scratch, independent of outer/table storage precision.
// No inventory, recursion, allocation or persistent gas state.
// Dense blocks form a block-tridiagonal Jacobian: storage O(MaxNodes*(N+4)^2),
// solve O(MaxNodes*(N+4)^3), independent wall faces can execute in parallel.
template<int N,int MaxNodes> struct WallWorkspaceStorage {
    using R=WallReal;
    static constexpr int maxNodes=MaxNodes,blockSize=N+4;
    // ReactingSst state: wall-relative tangential u[2], T-Tw, Ns-1 Y, sqrt(k), 1/sqrt(omega).
    R y[MaxNodes]{},state[MaxNodes][blockSize]{},trial[MaxNodes][blockSize]{};
    R residual[MaxNodes][blockSize]{},delta[MaxNodes][blockSize]{};
    R seed[MaxNodes][8]{}; // Reusable positive Picard initializer scratch.
    R lower[MaxNodes][blockSize][blockSize]{},diagonal[MaxNodes][blockSize][blockSize]{},upper[MaxNodes][blockSize][blockSize]{};
};
template<class StateReal,int N,int MaxNodes=48> using WallWorkspace=WallWorkspaceStorage<N,MaxNodes>;
namespace detail {
template<class R> UGKWP_GAS_HD inline R abs(R x){return x<0?-x:x;}
template<class R> UGKWP_GAS_HD inline R max(R x,R y){return x>y?x:y;}
template<class R> UGKWP_GAS_HD inline R min(R x,R y){return x<y?x:y;}
UGKWP_GAS_HD inline bool fail(WallStatus& s,WallCode c,int node=-1){s.code=c;s.node=node;return false;}
template<class R> UGKWP_GAS_HD inline R dot(const R* a,const R* b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
template<class R> UGKWP_GAS_HD inline R expRatio(R x){return abs(x)<R(1e-5)?R(1)-x/R(2)+x*x/R(12)-x*x*x*x/R(720):x/::expm1(x);}
template<class R,int N,class TableReal> UGKWP_GAS_HD inline bool validateInput(const WallInput<R,N,TableReal>& in,const WallModelConfig<R>& c,WallStatus& s){
    s=WallStatus{};
    if(c.family!=WallFamily::BoundaryLayer||!gasFinite(in.pressure)||!(in.pressure>0)
        ||!gasFinite(in.temperature)||!(in.temperature>0)||!gasFinite(in.matching.temperature)||!(in.matching.temperature>0)
        ||!gasFinite(in.matchingDistance)||!(in.matchingDistance>0)||!gasFinite(in.ownerDistance)
        ||!(in.ownerDistance>0&&in.ownerDistance<in.matchingDistance)
        ||!gasFinite(in.model.molecularPrandtl)||in.model.molecularPrandtl<0
        ||!gasFinite(in.model.viscosity)||!(in.model.viscosity>0)||!gasFinite(in.model.conductivity)||!(in.model.conductivity>0))return fail(s,WallCode::InvalidInput);
    R normal2=0,sum=0,mass=0;
    for(int d=0;d<3;++d){if(!gasFinite(in.normal[d])||!gasFinite(in.velocity[d])||!gasFinite(in.matching.velocity[d]))return fail(s,WallCode::InvalidInput);normal2+=in.normal[d]*in.normal[d];}
    const R tol=R(128*N)*std::numeric_limits<TableReal>::epsilon();
    if(abs(normal2-R(1))>tol)return fail(s,WallCode::InvalidInput);
    for(int j=0;j<N;++j){if(!gasFinite(in.massFlux[j])||!gasFinite(in.matching.massFraction[j])||in.matching.massFraction[j]<0
        ||!gasFinite(in.model.diffusivity[j])||in.model.diffusivity[j]<0)return fail(s,WallCode::InvalidInput);
        if(in.coarseReactionIntegral&&!gasFinite(in.coarseReactionIntegral[j]))return fail(s,WallCode::InvalidInput);
        sum+=in.matching.massFraction[j];mass+=in.massFlux[j];}
    if(abs(sum-R(1))>tol||mass<0)return fail(s,WallCode::InvalidInput);
    if(validateSpeciesThermoView(in.model.thermo)!=ThermoStatus::Success)return fail(s,WallCode::InvalidThermodynamics);
    for(int j=0;j<N;++j)if(in.temperature<in.model.thermo.species[j].minTemperature||in.temperature>in.model.thermo.species[j].maxTemperature||in.matching.temperature<in.model.thermo.species[j].minTemperature||in.matching.temperature>in.model.thermo.species[j].maxTemperature||!gasFinite(speciesH(j,in.temperature,in.model.thermo))||!gasFinite(speciesH(j,in.matching.temperature,in.model.thermo)))return fail(s,WallCode::InvalidThermodynamics);
    if(!gasFinite(c.relativeTolerance)||!(c.relativeTolerance>0)||!gasFinite(c.absoluteTolerance)||c.absoluteTolerance<0
        ||!gasFinite(c.stretch)||c.stretch<1||!gasFinite(c.turbulentPrandtl)||!(c.turbulentPrandtl>0)
        ||!gasFinite(c.turbulentSchmidt)||!(c.turbulentSchmidt>0))return fail(s,WallCode::InvalidInput);
    return true;
}
template<class R,int N,class TableReal> UGKWP_GAS_HD inline bool trace(const WallInput<R,N,TableReal>& in,const R* Y,WallOutput<R,N>& out){
    R gasR=0,m=0;for(int j=0;j<N;++j){gasR+=Y[j]*universalGasConstant<R>()/in.model.thermo.species[j].molarMass;m+=in.massFlux[j];out.traceMassFraction[j]=Y[j];out.wallSpeciesFlux[j]=in.massFlux[j];}
    out.traceTemperature=in.temperature;out.traceDensity=in.pressure/(gasR*in.temperature);
    for(int d=0;d<3;++d)out.traceVelocity[d]=in.velocity[d]+in.normal[d]*m/out.traceDensity;
    return gasFinite(out.traceDensity)&&out.traceDensity>0;
}
} // detail
}} // namespace
