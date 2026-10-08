
#pragma once
template<bool FullMoments>
struct ConstrainedPressureOperation
{
    static constexpr bool closeInactiveCells=true;
    struct Shared {};
    struct Accumulator {PressureReal moments[FullMoments?7:4];int count;};
    template<PressureDirectory Directory>
    __device__ static __forceinline__ void prepare
    (DeviceState& s,int c,PressureTime dt,Shared& w,PressureParameters& q)
    {
        // Retain the established block parameter layout and inactive-cell
        // loads while sharing the mathematical preparation and traversal.
        __shared__ PressureReal scaledDelta[4];
        __shared__ PressureReal pressureParameters[13];
        __shared__ int pressureParameterActive;
        __shared__ int pressureParameterResolved;
        if (threadIdx.x == 0)
        {
            if constexpr (Directory == PressureDirectory::Base || Directory == PressureDirectory::Injection)
            {
                scaledDelta[0] = finiteOr(s.pressureDeltaMomX[c], PressureReal(0));
                scaledDelta[1] = finiteOr(s.pressureDeltaMomY[c], PressureReal(0));
                scaledDelta[2] = finiteOr(s.pressureDeltaMomZ[c], PressureReal(0));
                scaledDelta[3] = finiteOr(s.pressureDeltaEnergy[c], PressureReal(0));
            }
            else pressureDeltaFromLimitedFaces(s, c, dt, scaledDelta);
            // Bind the common formula directly to the established shared
            // parameter storage, without an intermediate parameter object.
            struct ParameterStorage
            {
                PressureReal &px1,&py1,&pz1,&e1,&ux0,&uy0,&uz0,&ux1,&uy1,&uz1,&theta1,&thermalScale,&thetaScale;
                int &active,&resolved;
            } output{pressureParameters[0],pressureParameters[1],pressureParameters[2],pressureParameters[3],
                     pressureParameters[4],pressureParameters[5],pressureParameters[6],pressureParameters[7],
                     pressureParameters[8],pressureParameters[9],pressureParameters[10],pressureParameters[11],
                     pressureParameters[12],pressureParameterActive,pressureParameterResolved};
            makePressureParameters(s,c,scaledDelta,output);
        }
        __syncthreads();
        q.active = pressureParameterActive != 0;
        q.ux0 = pressureParameterActive ? pressureParameters[4] : PressureReal(0);
        q.uy0 = pressureParameterActive ? pressureParameters[5] : PressureReal(0);
        q.uz0 = pressureParameterActive ? pressureParameters[6] : PressureReal(0);
        q.ux1 = pressureParameterActive ? pressureParameters[7] : PressureReal(0);
        q.uy1 = pressureParameterActive ? pressureParameters[8] : PressureReal(0);
        q.uz1 = pressureParameterActive ? pressureParameters[9] : PressureReal(0);
        q.theta1 = pressureParameterActive ? pressureParameters[10] : PressureReal(0);
        q.thermalScale = pressureParameterActive ? pressureParameters[11] : PressureReal(0);
        q.thetaScale = pressureParameterActive ? pressureParameters[12] : PressureReal(0);
        q.resolved = pressureParameterActive && pressureParameterResolved != 0;
    }
    template<bool CheckCell,bool Compact>
    __device__ static __forceinline__ void visit
    (DeviceState& s,int c,int i,bool active,
     PressureReal ux0,PressureReal uy0,PressureReal uz0,
     PressureReal ux1,PressureReal uy1,PressureReal uz1,
     PressureReal theta1,PressureReal thermalScale,PressureReal thetaScale,
     bool resolved,Accumulator& a)
    {
        if(i<0 || i>=s.particleCapacity || (Compact?s.compactPStatus[i]:s.pStatus[i])==0)return;
        if constexpr(CheckCell) {if((Compact?s.compactPCellId[i]:s.pCellId[i])!=c)return;}
        if(active)
            applyPressureParticleStateDevice<Compact>
            (s,i,ux0,uy0,uz0,ux1,uy1,uz1,theta1,thermalScale,thetaScale,resolved);
        accumulatePressureParticleMomentsDevice<FullMoments,Compact>(s,c,i,a.moments,a.count);
    }
    template<PressureDirectory Directory>
    __device__ static __forceinline__ void finish
    (DeviceState& s,int c,const PressureParameters&,Accumulator& a)
    {
        extern __shared__ PressureReal pressureWarpPartials[];
        reducePressureParticleMomentsDevice<FullMoments>(a.moments,a.count,pressureWarpPartials);
        if(threadIdx.x==0)
        {
            if constexpr(Directory==PressureDirectory::Base || Directory==PressureDirectory::Injection)
                storePressureParticleMomentsDevice<FullMoments>(s,c,a.moments,a.count,Directory==PressureDirectory::Injection);
            else publishPressureParticleMomentsDevice<FullMoments>(s,c,a.moments,a.count);
        }
    }
};
