#ifndef GPU_PARTICLE_WALL_HEAT_TRANSFER_INTERFACE_H
#define GPU_PARTICLE_WALL_HEAT_TRANSFER_INTERFACE_H

                                                                              
                                                                       
struct GpuParticleWallHeatTransferInput
{
    int particleSlot;                                                       
    unsigned long long particleId;                                          
    unsigned long long originalParticleId;                                                

    int cellId;                                       
    int localPlaneId;                                               
    int globalBoundaryFaceId;                                           

    double hitPositionXMetres;
    double hitPositionYMetres;
    double hitPositionZMetres;
    double wallUnitNormalX;                                                    
    double wallUnitNormalY;
    double wallUnitNormalZ;
    double eventTimeSeconds;                                            
    double remainingStepSeconds;                                                

    double particleVelocityBeforeXMetresPerSecond;
    double particleVelocityBeforeYMetresPerSecond;
    double particleVelocityBeforeZMetresPerSecond;
    double particleVelocityAfterXMetresPerSecond;
    double particleVelocityAfterYMetresPerSecond;
    double particleVelocityAfterZMetresPerSecond;
    double wallVelocityXMetresPerSecond;
    double wallVelocityYMetresPerSecond;
    double wallVelocityZMetresPerSecond;

    double particleTemperatureKelvin;
    double particleThetaSquareMetresPerSquareSecond;
    double particleDiameterMetres;
    double parcelMassKilograms;
    double particleHeatFactorJoulesPerKilogramKelvin;
    double wallTemperatureKelvin;

    int boundaryKind;
    double normalRestitutionCoefficient;
    double tangentialCollisionCoefficient;
    int particleStatusBefore;
    int particleStatusAfter;
};

struct GpuParticleWallHeatTransferSources
{
                                                                          
    double particleEnergySourceJ;

                                                                            
    double wallEnergyIncrementJ;
};

                                                                              
struct GpuParticleWallEnergyLedgerView
{
    double* energyJoulesByGlobalFace;
    int numberOfGlobalFaces;
};

#if defined(__CUDACC__)
                                                                         
                                                                      
                   
  
                                                         
  
                                                                            
                                                                                
__device__ GpuParticleWallHeatTransferSources
evaluateParticleWallHeatTransferAndAccumulateWall
(
    const GpuParticleWallHeatTransferInput& contactState,
    GpuParticleWallEnergyLedgerView wallEnergyOut,
    int wallGlobalFaceId
);
#endif

#endif
