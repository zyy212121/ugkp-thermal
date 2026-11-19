#pragma once
namespace ugkp { namespace gpu {




template<class T>struct DeviceValue {
    const T* address=nullptr;
    __device__ operator T()const{return *address;}
};
}}
