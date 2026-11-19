#pragma once

template<class T>
int allocate(T*& ptr, const size_t n, const char* name)
{
    ptr = nullptr;
    if (n == 0)
    {
        return 0;
    }

    const cudaError_t err =
        cudaMalloc(reinterpret_cast<void**>(&ptr), n*sizeof(T));
    if (err != cudaSuccess)
    {
        setLastError(name, err);
        return 1;
    }
    return 0;
}

template<class T>
void release(T*& ptr)
{
    if (ptr != nullptr)
    {
        cudaFree(ptr);
        ptr = nullptr;
    }
}

template<class T>
int copyToDevice(T* dst, const T* src, const size_t n, const char* name)
{
    if (n == 0)
    {
        return 0;
    }
    const cudaError_t err =
        cudaMemcpy(dst, src, n*sizeof(T), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        setLastError(name, err);
        return 1;
    }
    return 0;
}

template<class T>
int copyToHost(T* dst, const T* src, const size_t n, const char* name)
{
    if (n == 0)
    {
        return 0;
    }
    const cudaError_t err =
        cudaMemcpy(dst, src, n*sizeof(T), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        setLastError(name, err);
        return 1;
    }
    return 0;
}

template<class D,class S,typename std::enable_if<!std::is_same<D,S>::value && std::is_arithmetic<D>::value && std::is_arithmetic<S>::value,int>::type = 0>
int copyToDevice(D* dst,const S* src,const size_t n,const char* name)
{
    if(!n)return 0;
    try
    {
        std::vector<D> staged(n);
        for(size_t i=0;i<n;++i)staged[i]=static_cast<D>(src[i]);
        return copyToDevice(dst,staged.data(),n,name);
    }
    catch(const std::exception& error)
    {
        std::snprintf(lastError,sizeof(lastError),"%s: %s",name,error.what());
        return 1;
    }
}
template<class D,class S,typename std::enable_if<!std::is_same<D,S>::value && std::is_arithmetic<D>::value && std::is_arithmetic<S>::value,int>::type = 0>
int copyToHost(D* dst,const S* src,const size_t n,const char* name)
{
    if(!n)return 0;
    try
    {
        std::vector<S> staged(n);
        if(copyToHost(staged.data(),src,n,name))return 1;
        for(size_t i=0;i<n;++i)dst[i]=static_cast<D>(staged[i]);
        return 0;
    }
    catch(const std::exception& error)
    {
        std::snprintf(lastError,sizeof(lastError),"%s: %s",name,error.what());
        return 1;
    }
}
