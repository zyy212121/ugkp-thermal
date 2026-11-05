#pragma once
#include "GpuParticleRadiationMath.H"
extern "C" int ugkwpGpuResidentStrictDownloadMobileParticleRadiationSums(void*,int,double*,double*,double*);
extern "C" int ugkwpGpuResidentStrictDownloadParticleWallOccupiedArea(void*,int,double*);
extern "C" int ugkwpGpuResidentStrictRefreshParticleEnthalpyPacked(void*);
// Extensive cell heat, distributed by available particle enthalpy. Returns the
// actually accepted extensive heat, including bounds, for paired wall settlement.
extern "C" int ugkwpGpuResidentStrictApplyParticleRadiationEnergy(void*,int,const double*,double*);
