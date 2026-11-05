#pragma once
#include <cstdlib>
#if defined(__GLIBC__)
#include <malloc.h>
#endif
namespace ugkp {
// Return already-freed host pages between large immutable-history batches.
// No live allocation or numerical state is modified.
inline void releaseUnusedHostPages(){
#if defined(__GLIBC__)
    ::malloc_trim(0);
#endif
}
}
