#pragma once
// Host calculation scalars are scrubbed after initial upload. Updating only
// steady configuration members preserves the authoritative device constants.
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
