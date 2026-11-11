// Checkpoint 2A: production-header mixed coupling energy regression.
// Numerical execution belongs to the parent checkpoint batch.
// Runs unchanged production device functions serially on a host. Not CUDA execution.
#include <cmath>
using std::isfinite;
using std::sqrt;
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include "GpuPrecisionTypes.H"
#include "GpuOperatorContract.cuh"
#include "CharacteristicMuscl.cuh"
#include "OpenFoamLimitedLinear.cuh"
#include "OpenFoamViscousFlux.cuh"
#include "OpenFoamWallFunctions.cuh"
#include "RiemannBoundaryState.cuh"
#include "RiemannGasFlux.cuh"
#include "GpuSstAlgebra.cuh"
#include "GpuLesAlgebra.cuh"
using Real=GpuReal;
using Time=double;
#define GPU_OPERATOR_REAL Real
#define GPU_OPERATOR_TIME Time
#define GPU_OPERATOR_R GPU_R
#define GPU_OPERATOR_TINY(x) GPU_R(1e-30)
#define __host__
#define __global__
#define __device__
#define __forceinline__ inline
// PTX traps are represented by process failure in this serial host emulator.
#define asm(...) std::abort()
struct ThreadIndex { int x=0; } blockIdx, threadIdx, blockDim;
constexpr Real OfSmall=GPU_R(2.22044604925031308085e-16);
constexpr Real OfVSmall=GPU_R(1e-30);
constexpr Real OfGreat=GPU_R(1.0)/OfSmall;
struct GasPrimDevice { Real rho,ux,uy,uz,p,T; };
#include "gasTransport/GasStateView.H"
#include "operators/clampMin.cuh"
#include "operators/clampRange.cuh"
#include "operators/linearScheduledValueDevice.cuh"
#include "operators/makeGasPrimDevice.cuh"
#include "operators/riemannFacePrimitiveForGradient.cuh"
#include "operators/computeGasPrimitiveGradientsKernel.cuh"
#include "operators/computeSstGradientsKernel.cuh"
#include "operators/sstVelocityInvariants.cuh"
#include "operators/computeGasHllcAdcSensorKernel.cuh"
#include "operators/updateBarthLimiter.cuh"
#include "operators/computeGasGradientLimiterKernel.cuh"
#include "operators/computeGasEddyViscosityKernel.cuh"
#include "operators/updateWaveTransmissivePressureBoundaryKernel.cuh"
#include "operators/updateLegacyGasBoundaryMirrorKernel.cuh"
#include "operators/gasFaceSubgridTransportProperties.cuh"
#include "operators/computeRiemannGasFaceFluxDevice.cuh"
#include "operators/computeGasInternalFaceFluxKernel.cuh"
#include "operators/computeGasFluxPositivityScaleKernel.cuh"
#include "operators/computeSstFaceFluxKernel.cuh"


#include <iostream>
#include <iomanip>
#include <limits>
#include <algorithm>

struct State : ugkwp::GasStateView<GpuReal,GpuTime,ugkwp::SstCoefficients,
                                 GpuEnergy,GpuGeometry,GpuWallEnergy> {
    ugkwp::GasSpeciesState<GpuReal,2,double> gasSpecies;
};
static_assert(std::is_same<decltype(State{}.rho),float*>::value,"bulk storage must be float");
static_assert(std::is_same<decltype(State{}.p),float*>::value,"primitive storage must be float");
static_assert(std::is_same<decltype(State{}.rhoE),double*>::value,"authoritative energy stays double");
static_assert(std::is_same<decltype(State{}.rhoENext),double*>::value,"RK energy stays double");
static_assert(std::is_same<decltype(State{}.V),double*>::value,"geometry stays double");
static_assert(std::is_same<decltype(State{}.pressureScheduleTimes),double*>::value,"time stays double");
static_assert(std::is_same<decltype(State{}.gasSpecies.thermo.coefficients),const double*>::value,"immutable thermo tables stay double");


#include "ParticleCaloricHeatResponse.H"
#include "GpuParticleThermalCarry.H"
ThreadIndex gridDim{1};
struct CoupledState : State {
    int count=1,particleCapacity=1,*particleCountDevice=&count;
    int cellParticleOffset[2]={0,1},sortedParticleIndex[1]={0};
    int pStatus[1]={1},pCellId[1]={0};unsigned char pStuck[1]={0};
    GpuReal pm[1]={.2f},pd[1]={.001f},pux[1]{},puy[1]{},puz[1]{};
    GpuReal puxOld[1]{},puyOld[1]{},puzOld[1]{},pTheta[1]{},pT[1]={500};
    GpuReal epsGPrev[1]={.75f},rhoSolid=2500,particleThermalRho=2500;
    GpuReal particleCp=1000,particleThermalCapacity=2500000;
    GpuReal dragParameter0=0,dragParameter1=0,dragParameter2=0,dragParameter3=0;
    GpuReal TpMin=100,TpMax=2000,particleDiameterFallback=.001f,epsSMin=0,gasPrOneThird=1;
    int solveParticleTemperature=1,particleGasHeatTransferModelId=1;
    double pThermalEnthalpyResidual[1]{},mixtureHeatInitialTemperature[1]{},mixtureHeatFraction[1]{};
    GpuReal couplingRhoOld[1]{},couplingUxOld[1]{},couplingUyOld[1]{},couplingUzOld[1]{},couplingTgasOld[1]{},thetaDragAlpha[1]{1};
    GpuReal momRhoP[1]{1},momRhoUPx[1]{},momRhoUPy[1]{},momRhoUPz[1]{},momRhoPD[1]{.001f};
    double momRhoEP[1]{};
    ugkwp::ParticleCaloric law;
};

void require(bool value,const char* text) {
    if(!value){std::fprintf(stderr,"mixed EOS/RK: %s\n",text);std::abort();}
}
template<class T> void allocateField(T&) {}
template<class T> void allocateField(T*& field) {field=new T[8]{};}
template<class T> void releaseField(T&) {}
template<class T> void releaseField(T*& field) {delete[] field;}
struct Fixture {
    CoupledState s;
    ugkp::diagnostics::DeviceContext diagnostics{};
    int cellStatus[1]{};
    Fixture() {
#define UGKWP_GAS_STATE_FIELD(Type,Name) allocateField(s.Name);
#include "gasTransport/GasStateFields.inc"
#undef UGKWP_GAS_STATE_FIELD
        s.nCells=1;s.nFaces=s.nInternalFaces=0;s.diagnostics=&diagnostics;
        s.gasSpecies.cellStatus=cellStatus;s.gasSpecies.mode=ugkwp::GasMode::SingleLegacy;
        s.gammaGas=GpuReal(1.4);s.Rgas=GpuReal(287);s.gasCp=GpuReal(1004.5);
        s.rhoMin=GpuReal(1e-12);s.TgasMin=GpuReal(1e-6);s.V[0]=1.;s.cellLength[0]=1.;
        s.rho[0]=GpuReal(2);s.rhoE[0]=1e9;s.Tgas[0]=300;
        blockDim.x=1;threadIdx.x=blockIdx.x=0;
    }
    ~Fixture() {
#define UGKWP_GAS_STATE_FIELD(Type,Name) releaseField(s.Name);
#include "gasTransport/GasStateFields.inc"
#undef UGKWP_GAS_STATE_FIELD
    }
};
bool close(double a,double b,double tolerance) {
    return std::abs(a-b)<=tolerance*std::max({1.,std::abs(a),std::abs(b)});
}

#define DeviceState CoupledState
#define GPU_OPERATOR_THERMAL 0
#include "operators/solidEpsFromMomentDevice.cuh"
#include "gasTransport/MixtureParticleCoupling.cuh"
#include "GpuCellDragRate.cuh"
template<class T>T sqr3(T x,T y,T z){return x*x+y*y+z*z;}
struct UnitDrag {
    static GpuReal inverseResponseTime(const ugkwpGpuDrag::DragInput&){return GpuReal(1);}
};
template<class DragModel>void cellDrag(CoupledState*sp,double dt) {
    const DragModel dragModel{};
#include "GpuCellDragUpdate.inl"
}
ugkwp::ParticleCaloric particleCaloricDevice(const CoupledState&s,int){return s.law;}
double particleEffectiveTemperatureDevice(const CoupledState&s,int i) {
    return ugkwp::effectiveParticleTemperature(
        ugkwp::ParticleThermalCarry<GpuReal>{s.pT[i],s.pThermalEnthalpyResidual[i]},s.law);
}
#include "GpuParticleGasHeatRelaxation.cuh"
#include "GpuMixtureThermalParticleHeat.cuh"
#undef DeviceState

struct CouplingFixture:Fixture {
    ugkwp::SpeciesThermoData<double> table[2];
    double coefficients[6]={1000,0,0,1000,0,0};
    float partial[2]={.5f,.5f},sound[1]{},cp[1]{},gasR[1]{};
    CouplingFixture(double velocity) {
        s.rho[0]=1;s.rhoUx[0]=s.Ux[0]=GpuReal(velocity);
        s.rhoUy[0]=s.rhoUz[0]=s.Uy[0]=s.Uz[0]=0;
        s.pux[0]=s.Ux[0];s.V[0]=.7000000000000001;
        s.gasMu=1.8e-5f;s.gasPrClamped=.71f;s.gasPrOneThird=std::cbrt(.71f);
        s.gasThermalConductivity=.06f;s.Tgas[0]=300;
        s.law.phaseChange=true;s.law.solidCp=1000;
        auto& species=s.gasSpecies;species.mode=ugkwp::GasMode::MixtureFrozen;
        species.rho=partial;species.soundSpeed=sound;species.heatCapacity=cp;species.gasConstant=gasR;
        for(int k=0;k<2;++k){table[k].coefficientOffset=3*k;table[k].molarMass=.028;
            table[k].minTemperature=100;table[k].maxTemperature=2000;table[k].referencePressure=101325;}
        species.thermo.species=table;species.thermo.coefficients=coefficients;species.thermo.coefficientCount=6;
        species.thermoControls.relativeEnergyTolerance=1e-12;
        species.thermoControls.relativeTemperatureTolerance=1e-12;
        const double masses[2]={partial[0],partial[1]};
        const double momentum=s.rhoUx[0],kinetic=.5*momentum*momentum/s.rho[0];
        s.rhoE[0]=kinetic+ugkwp::mixtureEnergy(masses,300.,species.thermo);
        recoverGasPrimitiveCell(s,0);
        require(!diagnostics.fatalClaim,"mixed initial EOS failed");
    }
};
void testCellDragUnresolvedEnergy() {
    double previous=0;
    for(double internal:{.125,.25,.5}) {
        Fixture f;auto&s=f.s;s.rho[0]=1;s.rhoSolid=4;s.gammaGas=1.4f;s.Rgas=1;s.TgasMin=.1f;
        const GpuReal velocity=65536;
        s.Ux[0]=s.rhoUx[0]=s.momRhoUPx[0]=velocity;
        const double kinetic=.5*double(velocity)*velocity;
        s.rhoE[0]=kinetic+500;s.momRhoEP[0]=kinetic+internal;
        const double before=s.rhoE[0];
        cellDrag<UnitDrag>(&s,.25);
        const double expected=internal*(1.-double(s.thetaDragAlpha[0]))/.75;
        const double received=s.rhoE[0]-before;
        const double tolerance=16*std::numeric_limits<double>::epsilon()*std::abs(before);
        require(std::isfinite(received)&&received>previous,"unresolved drag heat disappeared or lost trend");
        require(std::abs(received-expected)<=tolerance,"double unresolved drag exchange is inconsistent");
        require(s.rhoUx[0]==velocity,"zero-slip drag changed momentum");
        previous=received;
        std::cout<<"drag internal="<<internal<<" received="<<received<<" expected="<<expected
                 <<" residual="<<received-expected<<'\n';
    }
}
void testMixtureHeat(bool caloric) {
    double referenceGas=0,referenceParticle=0;
    for(double boost:{0.,10000000.}) {
        CouplingFixture f(boost);auto&s=f.s;
        const double volume=double(s.epsGPrev[0])*s.V[0],oldGas=s.rhoE[0];
        const double oldParticle=caloric?s.law.enthalpy(s.pT[0])+s.pThermalEnthalpyResidual[0]
            :double(s.pT[0])*s.particleCp;
        if(caloric) {
            prepareMixtureThermalParticleHeatKernel(&s,.1);
            mixtureThermalParticleHeatCell(s,0,.1,true,false,mixtureParticles::SerialCouplingTeam{});
        } else {
            mixtureParticleCouplingCell<UnitDrag,true>(s,0,.1,false,true,false,mixtureParticles::SerialCouplingTeam{});
        }
        require(!f.diagnostics.fatalClaim&&s.gasSpecies.cellStatus[0]==0,"mixed heat source failed");
        const double newParticle=caloric?s.law.enthalpy(s.pT[0])+s.pThermalEnthalpyResidual[0]
            :double(s.pT[0])*s.particleCp;
        const double residual=volume*(s.rhoE[0]-oldGas)+double(s.pm[0])*(newParticle-oldParticle);
        const double budgetTolerance=16*std::numeric_limits<double>::epsilon()*std::abs(oldGas)*volume
            +64*std::numeric_limits<float>::epsilon()*std::abs(double(s.pm[0])*(newParticle-oldParticle))+1e-8;
        require(std::isfinite(s.Tgas[0])&&s.Tgas[0]>300&&s.pT[0]<500&&s.pT[0]>300,"heat trend is invalid");
        require(std::abs(residual)<=budgetTolerance,"actual accepted particle/gas heat budget is inconsistent");
        if(boost==0){referenceGas=s.Tgas[0];referenceParticle=s.pT[0];}
        else {
            require(std::abs(s.Tgas[0]-referenceGas)<.2,"high kinetic energy changed finite-bath gas root");
            require(std::abs(s.pT[0]-referenceParticle)<.2,"high kinetic energy changed particle heat exchange");
        }
        std::cout<<"caloric="<<caloric<<" boost="<<boost<<" Tg="<<s.Tgas[0]<<" Tp="<<s.pT[0]
                 <<" budget residual="<<residual<<" bound="<<budgetTolerance<<'\n';
    }
}
int main(){
    std::cout<<std::setprecision(17);
    testCellDragUnresolvedEnergy();testMixtureHeat(false);testMixtureHeat(true);
    std::cout<<"PASS mixed coupling authoritative energy\n";
}
