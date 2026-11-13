// Included at the backend API scope after DeviceState and syncDeviceState.
extern "C" int ugkwpGpuResidentStrictConfigureParticleInjectionVelocity
(void* handle, int specified, double ux, double uy, double uz)
{
 DeviceState* s=asState(handle);
 if(validateState(s,"particle injection velocity configuration")!=0)return 1;
 if(specified&&(!std::isfinite(ux)||!std::isfinite(uy)||!std::isfinite(uz)))return 1;
 s->particleInjectionVelocitySpecified=specified!=0;
 s->particleInjectionVelocity[0]=GpuReal(ux);s->particleInjectionVelocity[1]=GpuReal(uy);s->particleInjectionVelocity[2]=GpuReal(uz);
 return syncDeviceState(s,"particle injection velocity configuration");
}
