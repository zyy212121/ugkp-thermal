#pragma once
namespace ugkp { namespace gpu {
// Bind a scalar produced by an earlier device operation. Conversion happens
// only inside a kernel; the host may pass the address but cannot read its value.
// Numerical kernels can be instantiated with this view or an immutable scalar
// without a runtime storage-mode branch or a second implementation of the math.
template<class T>struct DeviceScalar {
    const T* address=nullptr;
    __device__ operator T()const{return *address;}
};
}}
