#ifndef CHMT_GPU_GASSTABILITY_H
#define CHMT_GPU_GASSTABILITY_H
#include "gpu/DeviceViews.H"
#include "core/Configuration.H"
namespace chmt {
// Gas-owned acoustic, transport and molecular/turbulent diffusion bound. CPU
// reaction, material diffusion, Darcy and film clocks do not enter this function.
CHMT_HD inline Real gasOnlyStableStepLimit(GasView gas,GeometryView geometry,const PhysicsConfig& p,Real interval){
    if(!finite(interval)||interval<=0)return 0;
    Real limit=p.maxDt;
    for(int c=0;c<gas.nCells;++c){Real sum=0,diffusion=0;const auto w=gas.primitive[c];
        for(int k=geometry.cellFaceOffsets[c];k<geometry.cellFaceOffsets[c+1];++k){const int f=geometry.cellFaces[k];
            if(geometry.boundaryKind[f]==BoundaryKind::Empty)continue;
            const Real area=mag(geometry.areaVector[f]),distance=mag(geometry.faceCentre[f]-geometry.cellCentre[c]);
            if(!(area>0&&distance>0)){return 0;}
            const Vec3 n=geometry.areaVector[f]/area;const Real wn=geometry.sweptVolume[f]/(interval*area);
            sum+=area*(absValue(dot(w.velocity,n)-wn)+w.soundSpeed);
            Real D=0,cv=0,R=0;for(int s=0;s<Ns;++s){D=maxValue(D,p.gasDiffusivity[s]);cv+=w.Y[s]*(p.species[s].cp0-p.species[s].R+p.species[s].cp1*w.temperature);R+=w.Y[s]*p.species[s].R;}
            const Real nut=p.enableSst?gas.sst.primitive[c].eddyViscosity:0;
            if(p.enableSst)D+=nut/p.sst.turbulentSchmidt;
            const Real thermal=p.gasConductivity/(w.rho*cv)+(p.enableSst?nut*(cv+R)/(cv*p.sst.turbulentPrandtl):0);
            diffusion+=area*maxValue(maxValue(p.gasViscosity/w.rho+nut,D),thermal)/distance;
        }
        if(sum+2*diffusion>0)limit=minValue(limit,p.cfl*geometry.evaluationVolume[c]/(sum+2*diffusion));
    }
    return limit;
}
}
#endif
