#pragma once
#include "ablation/InterfaceMath.H"
#include "coupling/IntervalContracts.H"
namespace chmt {
// The material closure consumes a gas inventory based on the original bulk
// trace. Matching pressure is installed afterward; it must not change that
// conservative inventory or the accepted species/condensed transfer rates.
CHMT_HD inline void setMaterialSurfaceGasDrive(SurfacePhysicsInput& in,
    const GasPrimitive& bulk,const GasGradient* gradient,Real phaseVolume,Real donorShare,
    const WallFaceSample* accepted,const PhysicsConfig& physics){
    in.bulk=bulk;if(gradient)in.gradient=*gradient;
    in.gasInventory=conservativeGas(bulk,phaseVolume*donorShare,physics);
    if(accepted){
        in.useAcceptedFlux=true;
        for(int s=0;s<Ns;++s){in.acceptedGasSpecies[s]=accepted->speciesRate[s];
            in.acceptedPoreSpecies[s]=accepted->poreRate[s];in.acceptedPoreSweep[s]=accepted->poreSweepRate[s];}
        for(int c=0;c<Nc;++c)in.acceptedCondensed[c]=accepted->condensedRate[c];
    }
}
CHMT_HD inline void setMaterialSurfaceMatching(SurfacePhysicsInput& in,const GasWallClosureContext& context){
    in.wallContext=context;in.bulk.pressure=context.matchingPressure;
}
CHMT_HD inline void setMaterialSurfaceGasGeometry(SurfacePhysicsInput& in,Real area,Vec3 normal,
    Vec3 gasAreaVector,Real gasDistance,Vec3 baseVelocity,Real dt){
    in.area=area;in.normal=normal;in.gasArea=mag(gasAreaVector);
    in.gasNormal=-gasAreaVector/in.gasArea;in.gasDistance=gasDistance;in.dt=dt;
    in.aux.pressure=in.bulk.pressure;in.baseVelocity=baseVelocity;
}
CHMT_HD inline void setMaterialSurfaceSolidGeometry(SurfacePhysicsInput& in,Real distance,Vec3 areaVector){
    in.solidDistance=distance;in.solidArea=mag(areaVector);in.solidNormal=areaVector/in.solidArea;
}
}
