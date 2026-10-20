#include "common/gasNumerics/OpenFoamWallFunctions.cuh"
#include <iostream>
#include <iomanip>
int main(){
    const double mu=1.7244421537657099e-5,cp=1004.675,R=287.0500676,Tw=290.53216418474562;
    double y,p,T,Ux,Uy,Uz,k;
    std::cout<<std::setprecision(17);
    while(std::cin>>y>>p>>T>>Ux>>Uy>>Uz>>k){
        const double U=std::sqrt(Ux*Ux+Uy*Uy+Uz*Uz),rho=p/(R*Tw);
        auto momentum=ugkpwall::spaldingWallStateFromNormalGradient(U,y,mu/rho,U/y,.41,9.8);
        auto heat=ugkpwall::sstJayatillekeThermalTransport(rho,cp,mu,.72,.9,.09,.41,9.8,
            ugkpwall::jayatillekeSmoothP(.72/.9),ugkpwall::jayatillekeThermalYPlus(.72/.9,.41,9.8),k,y,U,0.,(Tw-T)/y);
        std::cout<<(mu+rho*momentum.nut)*Ux/y<<" "<<-heat.heatFlux<<"\n";
    }
}
