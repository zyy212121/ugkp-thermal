#pragma once


template<class T>
int syncSteadyMember(DeviceState* s,T DeviceState::* member,const char* name)
{
    const auto offset=reinterpret_cast<const unsigned char*>(&(s->*member))
        -reinterpret_cast<const unsigned char*>(s);
    const auto error=cudaMemcpy(reinterpret_cast<unsigned char*>(s->deviceState)+offset,
        &(s->*member),sizeof(T),cudaMemcpyHostToDevice);
    if(error!=cudaSuccess){setLastError(name,error);return 1;}
    return 0;
}
