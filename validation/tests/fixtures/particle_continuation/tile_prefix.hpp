
#include <algorithm>
#include <cassert>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
using std::min; using std::max;
#define __global__
#define __device__
#define __forceinline__ inline
struct Dim {unsigned x=0;};
Dim blockIdx,threadIdx,blockDim,gridDim;
int atomicAdd(int* p,int q){int a=*p;*p+=q;return a;}
