#ifndef UGKP_GPU_IMPLICIT_DIFFUSION_H
#define UGKP_GPU_IMPLICIT_DIFFUSION_H
#include <memory>
#include <vector>
namespace ugkp { namespace gpu {
struct DiffusionInput {
    int cells=0,internalFaces=0;
    std::vector<int> owner,neighbour;
    std::vector<double> volume,gamma,area,delta,gradientInternal,gradientBoundary,correction;
    bool corrected=false;
};
struct DiffusionMatrix {
    std::vector<double> diagonal,source,upper,lower,boundaryDiagonal,boundarySource,fluxCorrection;
};
class ImplicitDiffusion {
    struct Implementation;std::unique_ptr<Implementation> data_;
public:
    ImplicitDiffusion();~ImplicitDiffusion();
    void assemble(const DiffusionInput&,DiffusionMatrix&);
    void transient(const std::vector<double>& volume,const std::vector<double>& capacity,
        const std::vector<double>& rhs,double dt,bool sourceFromCapacity,DiffusionMatrix&);
};
}}
#endif
