
#pragma once
struct AnalyticPressureOperation
{
    static constexpr bool closeInactiveCells=false;
    struct Shared {PressureReal delta[4];};
    struct Accumulator {};
    template<PressureDirectory Directory>
    __device__ static __forceinline__ void prepare
    (DeviceState& s,int c,PressureTime dt,Shared& w,PressureParameters& q)
    {
        if(threadIdx.x==0)readValidatedPressureDelta(s,c,w.delta);
        __syncthreads();
        makePressureParameters(s,c,w.delta,q);
    }
    template<bool CheckCell,bool Compact>
    __device__ static __forceinline__ void visit
    (DeviceState& s,int c,int i,bool,
     PressureReal ux0,PressureReal uy0,PressureReal uz0,
     PressureReal ux1,PressureReal uy1,PressureReal uz1,
     PressureReal theta1,PressureReal thermalScale,PressureReal thetaScale,
     bool resolved,Accumulator&)
    {
        applyCollisionalPressureProjectionOneParticle<Compact>
        (s,i,c,ux0,uy0,uz0,ux1,uy1,uz1,theta1,resolved,thermalScale,thetaScale);
    }
    template<PressureDirectory Directory>
    __device__ static __forceinline__ void finish
    (DeviceState& s,int c,const PressureParameters& q,Accumulator&)
    {
        if(threadIdx.x==0)
        {
            s.momRhoUPx[c]=q.px1;s.momRhoUPy[c]=q.py1;s.momRhoUPz[c]=q.pz1;s.momRhoEP[c]=q.e1;
            s.rhoUsx[c]=q.px1;s.rhoUsy[c]=q.py1;s.rhoUsz[c]=q.pz1;s.rhoEs[c]=q.e1;
            s.Usx[c]=q.ux1;s.Usy[c]=q.uy1;s.Usz[c]=q.uz1;s.theta[c]=q.theta1;
        }
    }
};
