#pragma once
#include "../GpuPrecisionTypes.H"
#include "WallResolutionDefaults.H"
#include "../Diagnostics.H"
#include <type_traits>
#include "../gasTransport/MixtureThermo.H"
#include "../gasTransport/MolecularViscosity.H"
#include "../chemistry/GasRates.H"
#include "../gasNumerics/GpuSstAlgebra.cuh"

namespace ugkwp { namespace gaswall {
using WallReal=double;
enum class WallFamily : int { LowRe=0, WallFunction=1, BoundaryLayer=2 };
enum class BoundaryLayerModel : int { ConstantTransport=0, ReactingSst=1, FiniteRate=ReactingSst, EquilibriumOde=2 };
enum class WallCode : int { Success=0, InvalidInput, InvalidThermodynamics, Unsupported,
    InvalidGeometry, ChemistryFailure, NonConvergence, SingularJacobian, InadmissibleState };
struct WallStatus { WallCode code=WallCode::Success; int node=-1,iteration=0; double residual=std::numeric_limits<double>::quiet_NaN(); };
template<class R> struct WallModelConfig {
    WallFamily family=WallFamily::BoundaryLayer;
    BoundaryLayerModel model=BoundaryLayerModel::ReactingSst;
    int nodes=defaultWallNodes,maxIterations=60;
    R relativeTolerance=R(1e-8),absoluteTolerance=R(1e-10),stretch=R(defaultWallStretch);
    R turbulentPrandtl=R(.9),turbulentSchmidt=R(.7);
    R equilibriumKappa=R(.41),equilibriumAPlus=R(17);
    bool localStressCorrection=true; // blowing/suction local stress; wall damping retained
    bool enableSst=false;
    // Internal route selection, set only on a coherent solver/profile config copy.
    // Legacy, analytic and unrelated template routes retain their original math.
    bool coherentDiscretization=false;
};
template<class R,int N> struct WallMatchingState {
    R temperature=0,velocity[3]{},massFraction[N]{},k=0,omega=0;
};
template<class R,int N,class TableReal=R> struct WallGasModel {
    MolecularViscosityView<TableReal> molecularViscosity;
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
    ugkp::diagnostics::DeviceContext* diagnostics=nullptr;
    std::int64_t diagnosticEntity=-1;
    std::uint32_t diagnosticNode=0;
    R pressure=0,temperature=0,massFlux[N]{},velocity[3]{},normal[3]{};
    // False: prescribed wall temperature. True: solve wall temperature from
    // zero conductive heat flux; temperature then supplies the coordinate origin.
    bool adiabatic=false;
    R matchingDistance=0,ownerDistance=0;
    WallMatchingState<R,N> matching;
    WallGasModel<R,N,TableReal> model;
    WallOwnerQuadrature<TableReal> quadrature;
    // Optional quadrature publication; volumeSstClosure promotes its conservative
    // projection into the gas owner operator. Standalone diagnostics stay optional.
    bool integrateOwnerKSource=false;
    bool volumeSstClosure=false; // actual owner volume only, not the matching extension
    // Frozen resolved tangential derivatives in the orthonormal (t0,t1,n)
    // frame. Columns 0/1 are tangential directions; the BVP supplies column 2.
    R sstTangentialGradient[3][2]{};
    // Optional coarse chemistry integrated per wall area over the same domain.
    // Absence sets chemistryMismatchAvailable=false and mismatch values to NaN.
    const TableReal* coarseReactionIntegral=nullptr;
};
template<class R,int N> struct WallExchangeOutput {
    R traceTemperature=0,traceDensity=0,traceMassFraction[N]{},traceVelocity[3]{};
    R conductiveHeatFlux=0,traction[3]{},wallSpeciesFlux[N]{},matchingSpeciesFlux[N]{};
    R reactionIntegral[N]{},chemistryMismatch[N]{},speciesBalanceResidual[N]{};
    R integratedKSource=0,ownerOmega=0,wallKFlux=0;
    R ownerKContent=0,ownerKDestruction=0,ownerKBalanceResidual=0; // integral rho*k and betaStar*rho*k*omega
    // Persistent total normal k transport and omega diffusion; workspace is pooled.
    // Faces interpolate these shared fluxes, preserving owner/neighbour antisymmetry.
    int kFluxCount=0;
    R heatConduction=0,heatViscous=0,heatTransport=0,heatBalanceResidual=0;
    bool chemistryMismatchAvailable=false;
    int iterations=0;
    R residual=0;
};
// Full results remain useful at standalone entrypoints and in a bounded worker
// pool. Resident faces keep only exchange scalars plus model-sized flux arrays.
template<class R,int N> struct WallOutput : WallExchangeOutput<R,N> {
    R kFluxHeight[129]{},kTransportFlux[129]{},omegaDiffusiveFlux[129]{};
};
template<class R,int N> struct WallResidentOutput : WallExchangeOutput<R,N> {
    R* kFluxHeight=nullptr;
    R* kTransportFlux=nullptr;
    R* omegaDiffusiveFlux=nullptr;
};
template<class R,int N> UGKWP_GAS_HD inline void storeWallResidentOutput(
 const WallOutput<R,N>& source,WallResidentOutput<R,N>& target,R* flux,int capacity){
    static_cast<WallExchangeOutput<R,N>&>(target)=source;
    target.kFluxHeight=flux;
    target.kTransportFlux=flux?flux+capacity:nullptr;
    target.omegaDiffusiveFlux=flux?flux+2*capacity:nullptr;
    for(int i=0;i<source.kFluxCount;++i){
        target.kFluxHeight[i]=source.kFluxHeight[i];
        target.kTransportFlux[i]=source.kTransportFlux[i];
        target.omegaDiffusiveFlux[i]=source.omegaDiffusiveFlux[i];
    }
}
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
template<class R,int N,class T> UGKWP_GAS_HD inline R wallMolecularViscosity(
 const WallInput<R,N,T>& in,R temperature,const R* Y){
 return ugkwp::molecularViscosity<N>(in.model.molecularViscosity,in.model.viscosity,temperature,Y);
}
template<class R,int N,class T> UGKWP_GAS_HD inline R wallMolecularPrandtl(const WallInput<R,N,T>& in,const R* Y){
    // An explicitly supplied conductivity retains the existing precedence.
    return in.model.molecularPrandtl>0?ugkwp::molecularPrandtl<N>(in.model.molecularViscosity,in.model.molecularPrandtl,Y):R(0);
}
template<class R,int N,class T> UGKWP_GAS_HD inline R wallMolecularDiffusivity(const WallInput<R,N,T>& in,int j,R temperature,R pressure){
    return ugkwp::molecularDiffusivity(in.model.molecularViscosity,j,in.model.diffusivity[j],temperature,pressure);
}
namespace detail {
template<class R> UGKWP_GAS_HD inline R abs(R x){return x<0?-x:x;}
template<class R> UGKWP_GAS_HD inline R max(R x,R y){return x>y?x:y;}
template<class R> UGKWP_GAS_HD inline R min(R x,R y){return x<y?x:y;}
UGKWP_GAS_HD inline bool fail(WallStatus& s,WallCode c,int node=-1){s.code=c;s.node=node;return false;}
template<class R> UGKWP_GAS_HD inline R dot(const R* a,const R* b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
template<class R> UGKWP_GAS_HD inline bool wallTangentBasis(const R* normal,R tangent[2][3]){
    int axis=0;if(abs(normal[1])<abs(normal[axis]))axis=1;if(abs(normal[2])<abs(normal[axis]))axis=2;
    R norm=0;for(int d=0;d<3;++d){tangent[0][d]=(d==axis?R(1):R(0))-normal[axis]*normal[d];norm+=tangent[0][d]*tangent[0][d];}
    if(!gasFinite(norm)||norm==0)return false;
    for(int d=0;d<3;++d)tangent[0][d]/=::sqrt(norm);
    tangent[1][0]=normal[1]*tangent[0][2]-normal[2]*tangent[0][1];
    tangent[1][1]=normal[2]*tangent[0][0]-normal[0]*tangent[0][2];
    tangent[1][2]=normal[0]*tangent[0][1]-normal[1]*tangent[0][0];return true;
}
template<class R> UGKWP_GAS_HD inline R expRatio(R x){return abs(x)<R(1e-5)?R(1)-x/R(2)+x*x/R(12)-x*x*x*x/R(720):(x>R(50)?x*::exp(-x)/(-::expm1(-x)):x/::expm1(x));}


template<class R,int N,class T> UGKWP_GAS_HD inline bool wallFatal(const WallInput<R,N,T>& in,
 ugkp::diagnostics::Code code,unsigned site,double value=0,unsigned quantity=0,int component=-1){
    if(in.diagnostics)ugkp::diagnostics::recordFatal(in.diagnostics,ugkp::diagnostics::makeRecord(code,site,23,in.diagnosticNode,
        ugkp::diagnostics::EntityKind::Face,in.diagnosticEntity,component,value,0,0,quantity));
    return false;
}
template<class R,int N,class T> UGKWP_GAS_HD inline void wallWarning(const WallInput<R,N,T>& in,
 ugkp::diagnostics::Code code,unsigned site,double value,double bound,double corrected=0,int component=-1){
    ugkp::diagnostics::recordWarning(in.diagnostics,ugkp::diagnostics::makeRecord(code,site,23,in.diagnosticNode,
        ugkp::diagnostics::EntityKind::Face,in.diagnosticEntity,component,value,bound,corrected,
        code==ugkp::diagnostics::Code::FiniteYClip?ugkp::diagnostics::Quantity::MassFraction:(code==ugkp::diagnostics::Code::ConservationResidual?ugkp::diagnostics::Quantity::Conservation:ugkp::diagnostics::Quantity::Accuracy)));
}
template<class R,int N,class T> UGKWP_GAS_HD inline bool wallFraction(const WallInput<R,N,T>& in,R value,R& clipped,int species){
    if(!gasFinite(value))return wallFatal(in,ugkp::diagnostics::Code::NonFinite,2401,value,ugkp::diagnostics::Quantity::MassFraction,species);
    clipped=min(R(1),max(R(0),value));
    if(clipped!=value)wallWarning(in,ugkp::diagnostics::Code::FiniteYClip,2402,value,value<0?0:1,clipped,species);
    return true;
}
template<bool Enthalpy,class R,int N,class TableReal> UGKWP_GAS_HD inline R wallProperty(
 int species,R temperature,const SpeciesThermoView<TableReal,N>& thermo,bool* evaluated=nullptr){
    const R value=Enthalpy?speciesH(species,temperature,thermo):speciesCp(species,temperature,thermo);
    if(evaluated)*evaluated=gasFinite(value);
    return value;
}
template<class R,int N,class TableReal> UGKWP_GAS_HD inline bool validateInput(const WallInput<R,N,TableReal>& in,const WallModelConfig<R>& c,WallStatus& s){
    s=WallStatus{};
    if(ugkp::diagnostics::hasFatal(in.diagnostics))return false;

    if(!in.model.thermo.species||!in.model.thermo.coefficients){fail(s,WallCode::InvalidThermodynamics);return wallFatal(in,ugkp::diagnostics::Code::MissingOperator,10026);}
    const R scalarValues[]={in.pressure,in.temperature,in.matching.temperature,in.matchingDistance,in.ownerDistance,
        in.model.molecularPrandtl,in.model.viscosity,in.model.conductivity,c.relativeTolerance,c.absoluteTolerance,c.stretch,c.turbulentPrandtl,c.turbulentSchmidt};
    for(int i=0;i<13;++i)if(!gasFinite(scalarValues[i])){
        fail(s,WallCode::InvalidInput);
        return wallFatal(in,ugkp::diagnostics::Code::NonFinite,2403,scalarValues[i],i==0?ugkp::diagnostics::Quantity::Pressure:(i<3?ugkp::diagnostics::Quantity::Temperature:0),i);
    }
    for(int d=0;d<3;++d){const R values[]={in.normal[d],in.velocity[d],in.matching.velocity[d]};
        for(int j=0;j<3;++j)if(!gasFinite(values[j])){fail(s,WallCode::InvalidInput);return wallFatal(in,ugkp::diagnostics::Code::NonFinite,10027,values[j],ugkp::diagnostics::Quantity::Velocity,3*j+d);}}
    for(int j=0;j<N;++j){const R values[]={in.massFlux[j],in.matching.massFraction[j],in.model.diffusivity[j]};
        for(int k=0;k<3;++k)if(!gasFinite(values[k])){fail(s,WallCode::InvalidInput);return wallFatal(in,ugkp::diagnostics::Code::NonFinite,10028,values[k],k==1?ugkp::diagnostics::Quantity::MassFraction:0,j);}}
    for(int j=0;j<N;++j){const R molarMass=in.model.thermo.species[j].molarMass;
        if(!gasFinite(molarMass)||molarMass==0){fail(s,WallCode::InvalidThermodynamics);return wallFatal(in,gasFinite(molarMass)?ugkp::diagnostics::Code::ZeroDenominator:ugkp::diagnostics::Code::NonFinite,2405,molarMass,ugkp::diagnostics::Quantity::Denominator,j);}}
    if(in.temperature<=0||in.matching.temperature<=0){fail(s,WallCode::InvalidInput);return wallFatal(in,ugkp::diagnostics::Code::NegativeTemperature,2404,in.temperature<=0?in.temperature:in.matching.temperature,ugkp::diagnostics::Quantity::Temperature);}
    if(in.matchingDistance==0||in.pressure==0){fail(s,WallCode::InvalidInput);return wallFatal(in,ugkp::diagnostics::Code::ZeroDenominator,10029,in.matchingDistance==0?in.matchingDistance:in.pressure);}
    return true;
}
// The legacy terminalProfile argument remains source-compatible, but never
// bypasses physical validity or selects an alternate thermodynamic profile.
template<class R,int N,class TableReal> UGKWP_GAS_HD inline bool trace(const WallInput<R,N,TableReal>& in,const R* Y,WallOutput<R,N>& out,const bool terminalProfile=false){
    if(!in.model.thermo.species)return false;
    R gasR=0,m=0;for(int j=0;j<N;++j){gasR+=Y[j]*universalGasConstant<R>()/in.model.thermo.species[j].molarMass;m+=in.massFlux[j];out.traceMassFraction[j]=Y[j];out.wallSpeciesFlux[j]=in.massFlux[j];}
    out.traceTemperature=in.temperature;out.traceDensity=in.pressure/(gasR*in.temperature);
    for(int d=0;d<3;++d)out.traceVelocity[d]=in.velocity[d]+in.normal[d]*m/out.traceDensity;
    if(!gasFinite(out.traceDensity))return wallFatal(in,ugkp::diagnostics::Code::NonFinite,2406,out.traceDensity,ugkp::diagnostics::Quantity::Density);
    for(int d=0;d<3;++d)if(!gasFinite(out.traceVelocity[d]))return wallFatal(in,ugkp::diagnostics::Code::NonFinite,10030,out.traceVelocity[d],ugkp::diagnostics::Quantity::Velocity,d);
    (void)terminalProfile;
    return true;
}
} // detail
}} // namespace
