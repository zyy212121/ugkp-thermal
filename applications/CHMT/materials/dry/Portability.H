#ifndef CHMT_DRY_PORTABILITY_H
#define CHMT_DRY_PORTABILITY_H
#if defined(__CUDACC__)
#define CHMT_DRY_HD __host__ __device__
#else
#define CHMT_DRY_HD
#endif
#endif
