#pragma once
#include <cstring>
// FP32 collision state. Energy Q is measured about the weighted mean, never
// recovered by subtracting the bulk kinetic energy from a total energy.
#ifdef __CUDACC__
#define UGKP_CENTRAL_HD __host__ __device__
#else
#define UGKP_CENTRAL_HD
#endif
struct CollisionCentralMoment
{
    float mass, ux, uy, uz, q;
};
UGKP_CENTRAL_HD inline CollisionCentralMoment mergeCollisionCentralMoments
(CollisionCentralMoment a, const CollisionCentralMoment b)
{
    if (b.mass <= 0.0f) return a;
    if (a.mass <= 0.0f) return b;
    const float mass = a.mass + b.mass;
    const float weight = b.mass/mass;
    const float dx = b.ux-a.ux, dy = b.uy-a.uy, dz = b.uz-a.uz;
    a.q = a.q+b.q+0.5f*a.mass*weight*(dx*dx+dy*dy+dz*dz);
    a.ux += weight*dx; a.uy += weight*dy; a.uz += weight*dz;
    a.mass = mass;
    return a;
}
UGKP_CENTRAL_HD inline CollisionCentralMoment particleCollisionCentralMoment
(float mass, float ux, float uy, float uz, float theta)
{
    return {mass, ux, uy, uz, 1.5f*mass*theta};
}
// The eighth reduction lane transports an integer count bit-for-bit.
// Numeric conversion to float would lose odd counts above 2^24. Scratch
// remains a compact float array; only this count lane uses integer addition.
UGKP_CENTRAL_HD inline float collisionCountWord(int count)
{
#ifdef __CUDA_ARCH__
    return __int_as_float(count);
#else
    float word; std::memcpy(&word,&count,sizeof(word)); return word;
#endif
}
UGKP_CENTRAL_HD inline int collisionCountFromWord(float word)
{
#ifdef __CUDA_ARCH__
    return __float_as_int(word);
#else
    int count; std::memcpy(&count,&word,sizeof(count)); return count;
#endif
}
template<int N>
UGKP_CENTRAL_HD inline void mergeCollisionMomentGroup(float (&a)[N], const float (&b)[N])
{
    const auto q=mergeCollisionCentralMoments({a[0],a[1],a[2],a[3],a[4]},
                                              {b[0],b[1],b[2],b[3],b[4]});
    a[0]=q.mass;a[1]=q.ux;a[2]=q.uy;a[3]=q.uz;a[4]=q.q;
    for(int k=5;k<N;++k)
    {
        if(k==7)a[k]=collisionCountWord(collisionCountFromWord(a[k])+collisionCountFromWord(b[k]));
        else a[k]+=b[k];
    }
}
UGKP_CENTRAL_HD inline float collisionCentralTheta(const CollisionCentralMoment q)
{
    return q.mass>0.0f ? (2.0f/3.0f)*(q.q/q.mass) : 0.0f;
}
#undef UGKP_CENTRAL_HD
