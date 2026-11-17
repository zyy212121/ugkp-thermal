#pragma once
#include "AnchoredRadiationSnapshot.cuh"
#include "AnchoredRadiationDerivatives.H"
#include "AnchoredRadiationLinearization.H"
#include "RadiationMaximum.cuh"
namespace ugkp { namespace gpu { namespace {
struct AnchorCp {RadiationReal cp;__host__ __device__ RadiationReal enthalpy(RadiationReal t)const{return cp*t;}__host__ __device__ RadiationReal temperature(RadiationReal h)const{return h/cp;}__host__ __device__ RadiationReal enthalpyDerivative(RadiationReal)const{return cp;}};

struct AnchorArrays {RadiationReal *request,*actual,*temperature,*B0,*slope,*plus,*minus,*emission,*feedback,*residual,*relative,*maximum,*fraction,*searchRequest,*searchActual,*searchTemperature;unsigned char *valid,*inDomain;RadiationEnergyDomain<RadiationReal>* domain;RadiationReal cutoff;
 RadiationReal *temperatureGrid=nullptr,*effectiveRelative=nullptr,*allowedResidual=nullptr;
 unsigned char* usedTemperatureGrid=nullptr;RadiationReal relativeTolerance=0;
};
__global__ void anchorAggregate(RadiationParticleDeviceView p,int n,const RadiationReal* volume,RadiationReal* mass,RadiationReal* t,RadiationReal* diameter,RadiationReal* eps,MieOpticalTableView table,RadiationReal cutoff){int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;const auto m=p.mass[c];mass[c]=m;t[c]=m>0?p.temperatureMass[c]/m:p.minimumTemperature;diameter[c]=m>0?p.diameterMass[c]/m:0;eps[c]=volume[c]>0&&p.materialDensity>0?m/(volume[c]*p.materialDensity):0;if(eps[c]>cutoff&&diameter[c]>0&&std::isfinite(diameter[c]))diameter[c]=fmin(table.diameter[table.diameters-1],fmax(table.diameter[0],diameter[c]));}
__global__ void anchorDirectory(ParticleRadiationResponseView borrowed,int cells,const int* fallback,int* offsets){auto v=resolveParticleRadiationResponse(borrowed);int c=blockIdx.x*blockDim.x+threadIdx.x;if(c<=cells)offsets[c]=v.cellOffsets?v.cellOffsets[c]:fallback[c];}
__global__ void anchorCapture(ParticleRadiationResponseView borrowed,const int* offsets,const int* indices,int cells,RadiationEntryParticle<GpuReal>* entries,int* mapping){auto v=resolveParticleRadiationResponse(borrowed);for(int slot=blockIdx.x*blockDim.x+threadIdx.x;slot<offsets[cells];slot+=blockDim.x*gridDim.x){int i=v.cellOffsets?slot:indices[slot];RadiationEntryParticle<GpuReal> a{};a.mass=v.mass[i];a.thermal.temperature=v.temperature[i];anchoredSnapshotDetail::copyCarry(a.thermal,v,i,0);a.eligible=v.status[i]==1&&v.wallState[i]==0;entries[slot]=a;mapping[slot]=i;}}
__global__ void anchorLumped(int n,const RadiationReal* mass,const RadiationReal* t,int* offsets,RadiationEntryParticle<GpuReal>* entries){int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>n)return;offsets[c]=c;if(c==n)return;RadiationEntryParticle<GpuReal> a{};a.mass=mass[c];a.thermal.temperature=t[c];a.eligible=mass[c]>0;entries[c]=a;}
__device__ RadiationReal anchorKappa(MieOpticalTableView table,RadiationReal eps,RadiationReal d,RadiationReal t,RadiationReal cutoff){if(!(eps>cutoff)||!(d>0))return 0;return RadiationReal(3)*eps*mieBilinear(table.absorption,table.temperatures,mieLogBracket(table.diameter,table.diameters,d),mieLinearBracket(table.temperature,table.temperatures,t))/(RadiationReal(2)*d);}
template<class Law> __device__ void anchorInitCell(int c,Law law,const RadiationEntryParticle<GpuReal>* entries,const int* offsets,RadiationReal lo,RadiationReal hi,RadiationReal dt,MieOpticalTableView table,RadiationDeviceView v,AnchorArrays a){
 AnchoredParticleRadiationResponse<RadiationReal,Law> r{entries+offsets[c],offsets[c+1]-offsets[c],lo,hi,law};auto domain=r.domain();a.domain[c]=domain;const auto t=domain.mass>0?domain.entryTemperature:v.temperature[c];a.temperature[c]=a.searchTemperature[c]=t;a.request[c]=a.actual[c]=a.fraction[c]=a.searchRequest[c]=a.searchActual[c]=0;
 auto plus=bandEmissionDerivative(table.temperature,table.band,table.temperatures,t,true),minus=bandEmissionDerivative(table.temperature,table.band,table.temperatures,t,false);auto dp=anchoredTemperatureDerivative(r,true),dm=anchoredTemperatureDerivative(r,false);
 a.B0[c]=plus.B;a.plus[c]=dp.available?plus.derivative*dp.derivative:0;a.minus[c]=dm.available?minus.derivative*dm.derivative:0;a.slope[c]=a.plus[c];auto l=linearizeAnchoredRadiation(RadiationReal(0),plus.B,plus.B,a.slope[c],dt,v.volume[c]*anchorKappa(table,v.epsilon[c],v.diameter[c],t,a.cutoff));a.emission[c]=l.linearizedEmission;a.feedback[c]=l.thermalFeedback;a.residual[c]=a.relative[c]=0;a.valid[c]=l.valid;a.inDomain[c]=1;
 const auto initial=r.evaluate(RadiationReal(0),domain);a.temperatureGrid[c]=initial.temperatureGridAllowance;a.effectiveRelative[c]=a.allowedResidual[c]=0;a.usedTemperatureGrid[c]=0;
}
__global__ void anchorInitialize(int n,bool parcels,RadiationReal cp,const RadiationEntryParticle<GpuReal>* entries,const int* offsets,RadiationReal lo,RadiationReal hi,RadiationReal dt,MieOpticalTableView table,RadiationDeviceView v,AnchorArrays a){int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;if(parcels)anchorInitCell(c,ugkwp::ParticleCaloric{},entries,offsets,lo,hi,dt,table,v,a);else anchorInitCell(c,AnchorCp{cp},entries,offsets,lo,hi,dt,table,v,a);}
template<class Law> __device__ void anchorUpdateCell(int c,Law law,const RadiationEntryParticle<GpuReal>* entries,const int* offsets,RadiationReal lo,RadiationReal hi,RadiationReal dt,MieOpticalTableView table,RadiationThermalIterationDeviceView v,AnchorArrays a){
 AnchoredParticleRadiationResponse<RadiationReal,Law> r{entries+offsets[c],offsets[c+1]-offsets[c],lo,hi,law};
 const auto domain=a.domain[c];const auto oldQ=a.searchRequest[c];
 const auto oldT=a.searchTemperature[c],G=v.incident[c];
 const auto slope=G>=a.B0[c]?a.plus[c]:a.minus[c];
 const auto oldB=bandEmissionDerivative(table.temperature,table.band,table.temperatures,oldT,true);
 const auto kap=anchorKappa(table,v.epsilon[c],v.diameter[c],oldT,a.cutoff);
 const auto searchResidual=materialRadiationResidual(a.searchActual[c],oldB.B,kap,v.volume[c],G,dt,RadiationReal(0));
 const auto l=linearizeAnchoredRadiation(a.searchActual[c],oldB.B,a.B0[c],slope,dt,kap*v.volume[c]);


 const auto predicted=oldQ-l.f*searchResidual.residual;
 a.inDomain[c]=domain.contains(predicted);RadiationReal q=oldQ;
 if(l.valid&&searchResidual.valid&&std::isfinite(predicted)){auto delta=predicted-oldQ;RadiationReal alpha=1;
  if(delta>0&&predicted>domain.upper)alpha=(domain.upper-oldQ)/delta;
  if(delta<0&&predicted<domain.lower)alpha=(domain.lower-oldQ)/delta;
  q=oldQ+alpha*delta;if(!domain.contains(q))q=alpha<1?(delta>0?domain.upper:domain.lower):oldQ;
 }
 const auto sample=r.evaluate(q,domain);
 const bool candidateValid=l.valid&&searchResidual.valid&&sample.withinDomain&&sample.publicationValid;
 if(candidateValid&&domain.mass>0){
  a.searchRequest[c]=q;a.searchActual[c]=sample.actualEnergy;a.searchTemperature[c]=sample.temperature;
  const auto candidateB=bandEmissionDerivative(table.temperature,table.band,table.temperatures,sample.temperature,G>=a.B0[c]);
  const auto candidateK=anchorKappa(table,v.epsilon[c],v.diameter[c],sample.temperature,a.cutoff);
  const auto candidateResidual=materialRadiationResidual(sample.actualEnergy,candidateB.B,candidateK,v.volume[c],G,dt,RadiationReal(0));


  const auto retainedB=bandEmissionDerivative(table.temperature,table.band,table.temperatures,a.temperature[c],G>=a.B0[c]);
  const auto retainedK=anchorKappa(table,v.epsilon[c],v.diameter[c],a.temperature[c],a.cutoff);
  const auto retainedResidual=materialRadiationResidual(a.actual[c],retainedB.B,retainedK,v.volume[c],G,dt,RadiationReal(0));
  if(candidateResidual.valid&&(!a.valid[c]||!retainedResidual.valid||candidateResidual.normalized<=retainedResidual.normalized)){
   a.request[c]=q;a.actual[c]=sample.actualEnergy;a.temperature[c]=sample.temperature;a.fraction[c]=sample.fraction;a.temperatureGrid[c]=sample.temperatureGridAllowance;
  }
 }
 a.valid[c]=candidateValid;


 const auto B=bandEmissionDerivative(table.temperature,table.band,table.temperatures,a.temperature[c],G>=a.B0[c]);
 const auto k=anchorKappa(table,v.epsilon[c],v.diameter[c],a.temperature[c],a.cutoff);
 const auto residual=materialRadiationResidual(a.actual[c],B.B,k,v.volume[c],G,dt,RadiationReal(0));
 a.residual[c]=residual.residual;a.relative[c]=residual.valid&&a.valid[c]?residual.normalized:RadiationReal(INFINITY);


 const auto assessment=assessMaterialPublication(a.actual[c],residual.requestedByPhysics,a.temperatureGrid[c],a.relativeTolerance);
 a.allowedResidual[c]=assessment.allowedResidual;a.usedTemperatureGrid[c]=assessment.usedTemperatureGrid;
 a.effectiveRelative[c]=assessment.valid&&residual.valid&&a.valid[c]?assessment.effectiveNormalized:RadiationReal(INFINITY);
 const auto next=linearizeAnchoredRadiation(a.actual[c],B.B,a.B0[c],slope,dt,k*v.volume[c]);
 a.slope[c]=next.b;a.emission[c]=next.linearizedEmission;a.feedback[c]=next.thermalFeedback;
 if(!next.valid)a.relative[c]=a.effectiveRelative[c]=RadiationReal(INFINITY);
}
struct AnchorWindowDescriptor {
 int cells=0;bool parcels=false;
 RadiationReal cp=0,minimum=0,maximum=0,dt=0;
 const RadiationEntryParticle<GpuReal>* entries=nullptr;
 const int* offsets=nullptr;
 MieOpticalTableView table{};
 AnchorArrays arrays{};
};
__global__ void anchorUpdate(const AnchorWindowDescriptor* window,RadiationThermalIterationDeviceView v){
 const auto& w=*window;const int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=w.cells)return;
 if(w.parcels)anchorUpdateCell(c,ugkwp::ParticleCaloric{},w.entries,w.offsets,w.minimum,w.maximum,w.dt,w.table,v,w.arrays);
 else anchorUpdateCell(c,AnchorCp{w.cp},w.entries,w.offsets,w.minimum,w.maximum,w.dt,w.table,v,w.arrays);
}
__global__ void anchorResidualMaximum(const RadiationMaximum* reduced,RadiationReal* maximum){if(!threadIdx.x&&!blockIdx.x)*maximum=reduced?applyRadiationMaximum(RadiationReal(0),*reduced):RadiationReal(0);}

}}}
