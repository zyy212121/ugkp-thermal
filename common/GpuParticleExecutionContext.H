#pragma once
#include <type_traits>
#include <utility>
namespace ugkp { namespace gpu {
// Recording context only. Ordinary execution retains the particle owner's
// existing stream; callers supply the root when recording a conditional body.
template<class State,class=void>struct HasParticleCaptureContext:std::false_type{};
template<class State>struct HasParticleCaptureContext<State,std::void_t<
    decltype(std::declval<State>().particleCaptureStream),
    decltype(std::declval<State>().particleCaptureRoot)>>:std::true_type{};
template<class State>auto particleRecordingStream(const State* state){
    if constexpr(HasParticleCaptureContext<State>::value)return state->particleCaptureStream;
    else return nullptr;
}
template<class State>auto particleRecordingRoot(const State* state){
    if constexpr(HasParticleCaptureContext<State>::value)return state->particleCaptureRoot;
    else return nullptr;
}
}}
