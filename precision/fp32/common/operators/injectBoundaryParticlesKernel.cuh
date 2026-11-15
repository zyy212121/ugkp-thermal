#include "gasBoundary/Definition.H"
#include "ParticleInjectionVelocity.H"
#pragma once
#include "../wall/GpuColdWall1DHandles.cuh"
#include "BoundaryInjectionSequence.H"
#include "ParticleResidentInterval.H"
#include "GpuParticleThermalCarryStorage.cuh"
template<class State,class Real,class Time>
__device__ bool fieldParticleInlet(const State& s,int face,int cell,Time time,
    Real& ux,Real& uy,Real& uz,Real& rate,std::false_type) { return false; }
template<class State,class Real,class Time>
__device__ bool fieldParticleInlet(const State& s,int face,int cell,Time time,
    Real& ux,Real& uy,Real& uz,Real& rate,std::true_type) {
    const auto& view=s.gasBoundaryProgram;
    if(!view.faces||face>=view.count)return false;
    const auto& f=view.faces[face];if(f.solidMode<0)return false;
    using namespace ugkp::boundary;
    const Real eps=f.solidMode==0?s.epsS[cell]:
        Real(evaluate(view,f.function[SolidFraction],time,f.value[SolidFraction]));
    ux=f.uMode==0?s.gasBoundaryUx[face]:f.uMode==1?Real(evaluate(view,f.function[Ux],time,f.value[Ux])):s.gasBoundaryUx[face];
    uy=f.uMode==0?s.gasBoundaryUy[face]:f.uMode==1?Real(evaluate(view,f.function[Uy],time,f.value[Uy])):s.gasBoundaryUy[face];
    uz=f.uMode==0?s.gasBoundaryUz[face]:f.uMode==1?Real(evaluate(view,f.function[Uz],time,f.value[Uz])):s.gasBoundaryUz[face];
    ux=ugkwp::particleInjectionVelocityComponent(s.particleInjectionVelocitySpecified,s.particleInjectionVelocity[0],ux);
    uy=ugkwp::particleInjectionVelocityComponent(s.particleInjectionVelocitySpecified,s.particleInjectionVelocity[1],uy);
    uz=ugkwp::particleInjectionVelocityComponent(s.particleInjectionVelocitySpecified,s.particleInjectionVelocity[2],uz);
    const Real incoming=-(ux*s.Sfx[face]+uy*s.Sfy[face]+uz*s.Sfz[face]);
    rate=(eps>Real(0)?eps:Real(0))*s.rhoSolid*(incoming>Real(0)?incoming:Real(0));
    return true;
}
__global__ void injectBoundaryParticlesKernel
(
    DeviceState* sp,
    const GPU_OPERATOR_TIME requestedDt,
    const GPU_OPERATOR_TIME requestedSimulationTime,
    const unsigned long long positionSequence=0,
    const bool positionSequenceEnabled=false
)
{
    const auto dt=ugkwp::particleResidentStepDt(*sp,requestedDt);
    const auto simulationTime=ugkwp::particleResidentEndTime(*sp,requestedSimulationTime);
    DeviceState& s = *sp;
    const int j = blockIdx.x*blockDim.x + threadIdx.x;
    if (j >= s.nBoundarySources)
    {
        return;
    }
#if defined(UGKP_DEVELOPMENT_PROBES) && !GPU_OPERATOR_THERMAL
    if (s.sourceInjectedCount != nullptr)
    {
        s.sourceInjectedCount[j] = 0;
    }
#endif
    if (s.particleCapacity <= 0 || s.injectionParcelMass <= GPU_OPERATOR_R(0.0))
    {
        return;
    }

    const int sourceFace = s.sourceFace[j];
    if (sourceFace < 0 || sourceFace >= s.nFaces)
    {
        return;
    }

    const int sourceCell = s.sourceCell[j];
    if (sourceCell < 0 || sourceCell >= s.nCells)
    {
        return;
    }

    GPU_OPERATOR_REAL sourceUx=ugkwp::particleInjectionVelocityComponent(
        s.particleInjectionVelocitySpecified,s.particleInjectionVelocity[0],finiteOr(s.sourceUx[j],GPU_OPERATOR_R(0.0)));
    GPU_OPERATOR_REAL sourceUy=ugkwp::particleInjectionVelocityComponent(
        s.particleInjectionVelocitySpecified,s.particleInjectionVelocity[1],finiteOr(s.sourceUy[j],GPU_OPERATOR_R(0.0)));
    GPU_OPERATOR_REAL sourceUz=ugkwp::particleInjectionVelocityComponent(
        s.particleInjectionVelocitySpecified,s.particleInjectionVelocity[2],finiteOr(s.sourceUz[j],GPU_OPERATOR_R(0.0)));



    const GPU_OPERATOR_REAL sourceTheta=clampMin(finiteOr(
        s.sourceTheta[j]==GPU_OPERATOR_R(-1.0)?s.theta[sourceCell]:s.sourceTheta[j],
        GPU_OPERATOR_R(0.0)),GPU_OPERATOR_R(0.0));
    GPU_OPERATOR_REAL rate=finiteOr(s.sourceMassRate[j],GPU_OPERATOR_R(0.0));
    fieldParticleInlet(s,sourceFace,sourceCell,simulationTime,
        sourceUx,sourceUy,sourceUz,rate,ugkp::boundary::Capability<DeviceState>{});
    if (rate <= GPU_OPERATOR_R(0.0))
    {
        return;
    }

    const GPU_OPERATOR_REAL injectionParcelMass = s.injectionParcelMass;
    GPU_OPERATOR_REAL available =
        clampMin(finiteOr(s.sourceResidualMass[j], GPU_OPERATOR_R(0.0)), GPU_OPERATOR_R(0.0)) + rate*GPU_OPERATOR_REAL(dt);
    if (available < injectionParcelMass)
    {
        s.sourceResidualMass[j] = available;
        return;
    }

    const int nNew = static_cast<int>(floorf(available/injectionParcelMass));
    GPU_OPERATOR_REAL consumed = GPU_OPERATOR_R(0.0);
    int created = 0;

    for (int k = 0; k < nNew; ++k)
    {
        const int slot = atomicAdd(s.particleCountDevice, 1);
        if (slot >= s.particleCapacity)
        {
            atomicSub(s.particleCountDevice, 1);
            break;
        }

        Foam::gpuWall::allocateColdWall1DHandle(s,slot);

        unsigned long long rng =
            mixSeed
            (
                s.rngSeed
              ^ static_cast<unsigned long long>
                (
                    0xd1b54a32d192ed03ULL
                  + (static_cast<unsigned long long>(j) << 32)
                  + static_cast<unsigned long long>(slot)
                )
            );
        s.px[slot] = finiteOr(s.sourcePx[j], s.Cx[sourceCell]);
        s.py[slot] = finiteOr(s.sourcePy[j], s.Cy[sourceCell]);
        s.pz[slot] = finiteOr(s.sourcePz[j], s.Cz[sourceCell]);
        const auto& geometry = s.boundaryInjection;
        if (geometry.offsets) {
            double x = s.px[slot], y = s.py[slot], z = s.pz[slot];
            const int begin = geometry.offsets[j];
            ugkwp::sampleBoundaryInjection(geometry.triangles,
                begin, begin + geometry.counts[j],
                ugkwp::particleInjectionPositionSeed(s,s.rngSeed,sourceFace,
                    double(simulationTime),k,positionSequence,positionSequenceEnabled),x,y,z);
            s.px[slot] = x; s.py[slot] = y; s.pz[slot] = z;
        }
        s.pux[slot] = sourceUx;
        s.puy[slot] = sourceUy;
        s.puz[slot] = sourceUz;
        s.pT[slot] =
            clampRange(finiteOr(s.sourceT[j], s.TpMin), s.TpMin, s.TpMax);
        resetParticleThermalEnthalpyResidual(s, slot, 0);
        s.pTheta[slot] = sourceTheta;
        resetSeparateContactAge(s, slot, 0);
        s.pd[slot] = sampleDiameterAroundDevice(s, finiteOr(s.sourceD[j], s.particleDiameterFallback), rng);
        s.pm[slot] = injectionParcelMass;
        s.pCellId[slot] = sourceCell;
        s.pStatus[slot] = 1;
#if GPU_OPERATOR_THERMAL
        s.pStuck[slot] = 0;
        s.pStuckFaceId[slot] = -1;
        s.pDepositionArea[slot] = 0.0f;
        s.pContactDuration[slot] = 0.0f;
        s.pContactMaximumArea[slot] = 0.0f;
        s.pContactPeakFraction[slot] = 0.0f;
#ifdef GPU_PAIRED_PARTICLE_HEAT

        if(s.particleContactHeatEpoch)s.particleContactHeatEpoch[slot]=0;
#endif
        clearColdWallParticleState(s, slot);
        clearColdWall2DParticleState(s, slot);
#endif
        s.pRng[slot] = rng;
        s.pOrigId[slot] =
            (static_cast<unsigned long long>(j) << 40)
          ^ static_cast<unsigned long long>(slot);

        consumed += GPU_OPERATOR_REAL(s.pm[slot]);
        ++created;
    }

    s.sourceResidualMass[j] = available - consumed;
#if defined(UGKP_DEVELOPMENT_PROBES) && !GPU_OPERATOR_THERMAL
    if (s.sourceInjectedCount != nullptr)
    {
        s.sourceInjectedCount[j] = created;
    }
#endif
}
