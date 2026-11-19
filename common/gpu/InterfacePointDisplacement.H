#ifndef UGKP_GPU_INTERFACE_POINT_DISPLACEMENT_H
#define UGKP_GPU_INTERFACE_POINT_DISPLACEMENT_H
#include "PolyhedralGeometry.H"
#include <cstddef>
namespace ugkp { namespace gpu { namespace geometry {


struct InterfaceThicknessField {const double* first=nullptr;std::size_t stride=sizeof(double);};
struct InterfacePointDisplacementInput {
    const Point *oldSolid=nullptr,*solidEndpoint=nullptr,*solidArea=nullptr,*oldGas=nullptr;
    InterfaceThicknessField pointThickness,previousFilmThickness,filmThickness;
};
struct InterfacePointDisplacementView {
    const Point* displacement=nullptr;
    const int* changed=nullptr;
};




class ResidentInterfacePointDisplacement {
    struct Implementation;std::unique_ptr<Implementation> data_;
public:
    ResidentInterfacePointDisplacement(const Topology& solid,int solidPoints,int gasPoints,
        const std::vector<int>& solidFaces,const std::vector<int>& solidToGas);
    ~ResidentInterfacePointDisplacement();
    InterfacePointDisplacementView view()const;
    void evaluate(const InterfacePointDisplacementInput*,void* stream=nullptr);
};
}}}
#endif
