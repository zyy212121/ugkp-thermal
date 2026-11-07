#pragma once
#include "GpuParticleRadiationMath.H"
#include "../gpu/ParticleRadiationResponse.H"
#include "../gpu/RadiationResidentView.H"
extern "C" int ugkwpGpuResidentStrictDownloadMobileParticleRadiationSums(void*,int,double*,double*,double*);
extern "C" int ugkwpGpuResidentStrictDownloadParticleWallOccupiedArea(void*,int,double*);
extern "C" int ugkwpGpuResidentStrictRefreshParticleEnthalpyPacked(void*);
// Extensive cell heat, distributed by available particle enthalpy. Returns the
// actually accepted extensive heat, including bounds, for paired wall settlement.
extern "C" int ugkwpGpuResidentStrictApplyParticleRadiationEnergy(void*,int,const double*,double*);

extern "C" int ugkwpGpuResidentStrictParticleRadiationResponse(void*,ugkp::gpu::ParticleRadiationResponseView*);

extern "C" int ugkwpGpuResidentStrictBorrowRadiationParticleState(void*,ugkp::gpu::RadiationParticleDeviceView*);
extern "C" int ugkwpGpuResidentStrictApplyParticleRadiationDeviceEnergy(void*,const ugkp::gpu::RadiationMaterialDeviceView*,double* accepted=nullptr);
extern "C" int ugkwpGpuResidentStrictBorrowAcceptedRadiationEnergy(void*,ugkp::gpu::RadiationAcceptedEnergyDeviceView*);
