#include "gasNumerics/GasWindowStep.H"
#include <cassert>
#include <cmath>
#include <iostream>
int main(){
 const double proposed=1e-7,tail=3.3087224502121107e-24;
 const auto a=ugkwp::selectGasWindowStep(proposed,tail,2e-7,1.1,1e-6);
 assert(a.accepted==tail && a.next==1.1*proposed);
 const auto b=ugkwp::selectGasWindowStep(a.next,1e-6,2e-7,1.1,1e-6);
 assert(b.accepted==a.next);
 const auto limited=ugkwp::selectGasWindowStep(proposed,tail,2e-8,1.1,1e-6);
 assert(limited.accepted==tail && limited.next==1.1*2e-8);
 const auto regular=ugkwp::selectGasWindowStep(proposed,1e-6,2e-7,1.1,1e-6);
 assert(regular.accepted==proposed && regular.next==a.next);
 const auto capped=ugkwp::selectGasWindowStep(1e-6,2e-6,2e-6,1.1,1e-6);
 assert(capped.accepted==1e-6 && capped.next==1e-6);
 std::cout<<"PASS exact window tail, next interval growth, CFL shrink, normal step, maximum cap\n";
}
