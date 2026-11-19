#pragma once
#include <type_traits>

template<class Field>
void allocateGasTestField(Field& field,int count)
{
    if constexpr(std::is_pointer_v<Field>)field=new std::remove_pointer_t<Field>[count]{};
}

template<class State>
void allocateGasTestStorage(State& s,int count)
{
#define UGKWP_GAS_STATE_FIELD(Type,Name) allocateGasTestField(s.Name,count);
#include "gasTransport/GasStateFields.inc"
#undef UGKWP_GAS_STATE_FIELD
}
