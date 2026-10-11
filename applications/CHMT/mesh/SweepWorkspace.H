#ifndef CHMT_MESH_SWEEP_WORKSPACE_H
#define CHMT_MESH_SWEEP_WORKSPACE_H
#include "core/HostState.H"
#include <algorithm>
#include <memory>
#include <new>
#include <stdexcept>
#include <vector>

namespace chmt { namespace sweep_detail {
// This budget covers correction-owned allocation requests, including vector
// capacities and row objects. It excludes the caller's mesh and the existing
// geometry/certification copies, allocator metadata, and process RSS.
struct WorkspaceLimit : std::bad_alloc {};
inline std::size_t product(std::size_t a,std::size_t b) {
    if(b && a>std::numeric_limits<std::size_t>::max()/b)throw WorkspaceLimit{};
    return a*b;
}
inline std::size_t sum(std::size_t a,std::size_t b) {
    if(a>std::numeric_limits<std::size_t>::max()-b)throw WorkspaceLimit{};
    return a+b;
}
struct Budget {
    std::size_t limit,live=0;
    std::size_t& peak;
    Budget(std::size_t bytes,std::size_t& highWater):limit(bytes),peak(highWater) {}
    bool fits(std::size_t bytes)const {return live<=limit && bytes<=limit-live;}
    void check(std::size_t bytes)const {if(!fits(bytes))throw WorkspaceLimit{};}
    void acquired(std::size_t bytes) {live+=bytes;peak=std::max(peak,live);}
    void released(std::size_t bytes) {live-=bytes;}
};
template<class T> struct Allocator {
    using value_type=T;
    Budget* budget=nullptr;
    Allocator()=default;
    explicit Allocator(Budget& b):budget(&b) {}
    template<class U> Allocator(const Allocator<U>& a):budget(a.budget) {}
    T* allocate(std::size_t n) {
        const std::size_t bytes=product(n,sizeof(T));
        if(!budget)throw WorkspaceLimit{};
        budget->check(bytes); // before the underlying allocation, not afterward
        T* result=std::allocator<T>{}.allocate(n);
        budget->acquired(bytes);
        return result;
    }
    void deallocate(T* p,std::size_t n) {
        std::allocator<T>{}.deallocate(p,n);budget->released(n*sizeof(T));
    }
    template<class U> bool operator==(const Allocator<U>& a)const {return budget==a.budget;}
    template<class U> bool operator!=(const Allocator<U>& a)const {return !(*this==a);}
};
template<class T> using Vector=std::vector<T,Allocator<T>>;
using Reals=Vector<Real>;
using Points=Vector<Vec3>;
using Rows=Vector<Reals>;

// One solve-local, sorted adjacency. Summation order, inverse-edge weights,
// Jacobi iteration and stopping criterion match harmonicPointMotion. There is
// no topology cache surviving a geometry update or checkpoint restart.
class HarmonicModes {
    using Edge=std::pair<int,int>;
    const HostMesh& mesh_;
    const std::vector<unsigned char>& fixed_;
    Vector<Edge> edges_;
    Vector<std::size_t> offsets_;
    Reals weights_,totals_;
    Points displacement_,next_;
    Real tolerance_;
public:
    HarmonicModes(const HostMesh& mesh,const std::vector<unsigned char>& fixed,
        Budget& budget,Real tolerance):mesh_(mesh),fixed_(fixed),
        edges_(Allocator<Edge>(budget)),offsets_(mesh.points.size()+1,0,Allocator<std::size_t>(budget)),
        weights_(Allocator<Real>(budget)),totals_(mesh.points.size(),0,Allocator<Real>(budget)),
        displacement_(mesh.points.size(),Vec3{},Allocator<Vec3>(budget)),
        next_(mesh.points.size(),Vec3{},Allocator<Vec3>(budget)),tolerance_(tolerance) {
        edges_.reserve(product(2,mesh.facePoints.size()));
        for(std::size_t f=0;f<mesh.owner.size();++f) {
            const int begin=mesh.faceOffsets[f],end=mesh.faceOffsets[f+1];
            for(int k=begin;k<end;++k) {
                const int a=mesh.facePoints[k],b=mesh.facePoints[k+1==end?begin:k+1];
                edges_.emplace_back(a,b);edges_.emplace_back(b,a);
            }
        }
        std::sort(edges_.begin(),edges_.end());
        edges_.erase(std::unique(edges_.begin(),edges_.end()),edges_.end());
        weights_.resize(edges_.size());
        for(std::size_t k=0;k<edges_.size();++k) {
            const int p=edges_[k].first,other=edges_[k].second;
            ++offsets_[p+1];
            if(!fixed_[p]) {
                const Real length=mag(mesh.points[p]-mesh.points[other]);
                // The initial public harmonic solve already validates this.
                weights_[k]=1/length;totals_[p]+=weights_[k];
            }
        }
        for(std::size_t p=1;p<offsets_.size();++p)offsets_[p]+=offsets_[p-1];
    }
    bool solve(const Points& boundary,Points& output) {
        std::fill(displacement_.begin(),displacement_.end(),Vec3{});
        std::fill(next_.begin(),next_.end(),Vec3{});
        for(std::size_t p=0;p<fixed_.size();++p)if(fixed_[p])displacement_[p]=boundary[p];
        bool converged=mesh_.points.empty();
        for(int iteration=0;iteration<20000 && !converged;++iteration) {
            Real residual=0;
            for(std::size_t p=0;p<fixed_.size();++p) {
                if(fixed_[p]){next_[p]=displacement_[p];continue;}
                Vec3 weighted{};
                for(std::size_t k=offsets_[p];k<offsets_[p+1];++k)
                    weighted+=displacement_[edges_[k].second]*weights_[k];
                next_[p]=weighted/totals_[p];
                residual=std::max(residual,mag(next_[p]-displacement_[p]));
            }
            displacement_.swap(next_);converged=residual<=tolerance_;
        }
        if(!converged)return false;
        output.resize(mesh_.points.size());
        for(std::size_t p=0;p<output.size();++p) {
            output[p]=mesh_.points[p]+displacement_[p];
            output[p]-=mesh_.points[p];
            if(fixed_[p])output[p]=boundary[p];
        }
        return true;
    }
};
} }
#endif
