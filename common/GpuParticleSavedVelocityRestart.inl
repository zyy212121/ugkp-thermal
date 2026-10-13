extern "C" int ugkwpGpuResidentStrictUploadParticleSavedVelocity
(void* handle, int count, const double* x, const double* y, const double* z)
{
    DeviceState* s=asState(handle);
    if (validateState(s,"saved particle velocity upload")!=0) return 1;
    if (count && (!x || !y || !z))
    { setLastErrorText("invalid saved particle velocity upload");return 1; }
    if (copyToDevice(s->puxOld,x,count,"restart saved pux")!=0) return 1;
    if (copyToDevice(s->puyOld,y,count,"restart saved puy")!=0) return 1;
    return copyToDevice(s->puzOld,z,count,"restart saved puz");
}
extern "C" int ugkwpGpuResidentStrictDownloadParticleSavedVelocity
(void* handle, int count, double* x, double* y, double* z)
{
    DeviceState* s=asState(handle);
    if (validateState(s,"saved particle velocity download")!=0) return 1;
    if (count && (!x || !y || !z))
    { setLastErrorText("invalid saved particle velocity download");return 1; }
    if (copyToHost(x,s->puxOld,count,"restart saved pux")!=0) return 1;
    if (copyToHost(y,s->puyOld,count,"restart saved puy")!=0) return 1;
    return copyToHost(z,s->puzOld,count,"restart saved puz");
}
