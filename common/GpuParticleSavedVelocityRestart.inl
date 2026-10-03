extern "C" int ugkwpGpuResidentStrictUploadParticleSavedVelocity
(void* handle, int count, const double* x, const double* y, const double* z)
{
    DeviceState* s=asState(handle);
    if (validateState(s,"saved particle velocity upload")!=0) return 1;
    int actual=0;
    if (count<0 || count>s->particleCapacity
        || copyToHost(&actual,s->particleCountDevice,1,"saved velocity count")!=0
        || actual!=count || (count && (!x || !y || !z)))
    { setLastErrorText("invalid saved particle velocity upload");return 1; }
    typedef typename std::remove_pointer<decltype(s->puxOld)>::type SavedReal;
    const double bound=static_cast<double>(std::numeric_limits<SavedReal>::max());
    for (int i=0;i<count;++i)
        if (!std::isfinite(x[i]) || !std::isfinite(y[i]) || !std::isfinite(z[i])
            || std::fabs(x[i])>bound || std::fabs(y[i])>bound || std::fabs(z[i])>bound)
        { setLastErrorText("saved particle velocity is not representable");return 1; }
    if (copyToDevice(s->puxOld,x,count,"restart saved pux")!=0) return 1;
    if (copyToDevice(s->puyOld,y,count,"restart saved puy")!=0) return 1;
    return copyToDevice(s->puzOld,z,count,"restart saved puz");
}
extern "C" int ugkwpGpuResidentStrictDownloadParticleSavedVelocity
(void* handle, int count, double* x, double* y, double* z)
{
    DeviceState* s=asState(handle);
    if (validateState(s,"saved particle velocity download")!=0) return 1;
    int actual=0;
    if (count<0 || count>s->particleCapacity
        || copyToHost(&actual,s->particleCountDevice,1,"saved velocity count")!=0
        || actual!=count || (count && (!x || !y || !z)))
    { setLastErrorText("invalid saved particle velocity download");return 1; }
    if (copyToHost(x,s->puxOld,count,"restart saved pux")!=0) return 1;
    if (copyToHost(y,s->puyOld,count,"restart saved puy")!=0) return 1;
    return copyToHost(z,s->puzOld,count,"restart saved puz");
}
