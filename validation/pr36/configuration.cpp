#include "FoamSommerfeldConfiguration.H"
#include "FoamParticleDiameterDistribution.H"
#include "IStringStream.H"
#include <iostream>
#include <cmath>
int main(int argc,char**argv){if(argc!=2)return 2;Foam::IStringStream text(argv[1]);Foam::dictionary d(text);
 const auto threshold=Foam::readSommerfeldThreshold(d),n=Foam::readParticleDiameterShape(d);
 std::cout<<"CONFIG "<<std::isinf(threshold)<<" "<<threshold<<" "<<n<<"\n";}
