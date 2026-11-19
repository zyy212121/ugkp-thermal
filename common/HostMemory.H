#pragma once
#include <cstdlib>
#if defined(__GLIBC__)
#include <malloc.h>
#endif
namespace ugkp {


inline void releaseUnusedHostPages(){
#if defined(__GLIBC__)
    ::malloc_trim(0);
#endif
}
}
