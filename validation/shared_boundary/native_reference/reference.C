#include "fvCFD.H"
#include <iomanip>
int main(int argc,char** argv) {
    Foam::argList::noParallel();Foam::argList args(argc,argv);
    Foam::Time runTime(Foam::Time::controlDictName,args);
    Foam::fvMesh mesh(Foam::IOobject(Foam::fvMesh::defaultRegion,runTime.timeName(),runTime,Foam::IOobject::MUST_READ));
    Foam::volVectorField U(Foam::IOobject("U","0",mesh,Foam::IOobject::MUST_READ,Foam::IOobject::NO_WRITE),mesh);
    Foam::volScalarField p(Foam::IOobject("p","0",mesh,Foam::IOobject::MUST_READ,Foam::IOobject::NO_WRITE),mesh);
    Foam::volScalarField T(Foam::IOobject("T","0",mesh,Foam::IOobject::MUST_READ,Foam::IOobject::NO_WRITE),mesh);
    Foam::dimensionedScalar R("R",Foam::dimEnergy/Foam::dimMass/Foam::dimTemperature,287.05);
    Foam::volScalarField rho(Foam::IOobject("rho","0",mesh),p/(R*T));
    Foam::volScalarField psi(Foam::IOobject("thermo:psi","0",mesh),1/(R*T));
    Foam::surfaceScalarField phi(Foam::IOobject("phi","0",mesh),Foam::fvc::flux(U));
    for(int iteration=0;iteration<300;++iteration) {
        psi=1/(R*T);rho=p/(R*T);phi=Foam::fvc::flux(U);
        U.correctBoundaryConditions();T.correctBoundaryConditions();p.correctBoundaryConditions();
    }
    std::cout<<std::setprecision(17);
    forAll(mesh.boundary(),patch)if(p.boundaryField()[patch].type()=="totalPressure")
        forAll(p.boundaryField()[patch],f) {
            const auto u=U.boundaryField()[patch][f];
            std::cout<<"REFERENCE "<<mesh.boundary()[patch].name()<<' '<<f<<' '
                <<u.x()<<' '<<u.y()<<' '<<u.z()<<' '<<p.boundaryField()[patch][f]<<' '<<T.boundaryField()[patch][f]<<'\n';
        }
}
