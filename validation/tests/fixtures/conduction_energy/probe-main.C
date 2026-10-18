
}}
namespace {
chmt::HostMesh testMesh(const Foam::fvMesh& mesh){
    chmt::HostMesh out;
    for(const auto& p:mesh.points())out.points.push_back({p.x(),p.y(),p.z()});
    out.oldPoints=out.points;out.referencePoints=out.points;
    const int nf=mesh.faces().size();out.faceOffsets.push_back(0);
    out.neighbour.assign(nf,-1);out.periodicPartner.assign(nf,-1);
    out.boundaryKind.assign(nf,chmt::BoundaryKind::Internal);out.boundaryPrimitive.resize(nf);out.boundarySst.resize(nf);
    for(int f=0;f<nf;++f){out.owner.push_back(mesh.faceOwner()[f]);
        if(f<mesh.faceNeighbour().size())out.neighbour[f]=mesh.faceNeighbour()[f];
        for(int point:mesh.faces()[f])out.facePoints.push_back(point);
        out.faceOffsets.push_back(out.facePoints.size());out.faceIds.push_back(f);}
    for(const auto& patch:mesh.boundaryMesh()){assert(patch.type()=="wall"||patch.coupled());
        for(int f=0;f<patch.size();++f)out.boundaryKind[patch.start()+f]=chmt::BoundaryKind::NoSlip;}
    std::string error;assert(chmt::rebuildGeometry(out,error));out.oldVolumes=out.volumes;return out;
}
}


using namespace chmt;
int main(int argc,char** argv){
 Foam::argList args(argc,argv);if(!args.checkRootCase())return 2;
 Foam::Time time(Foam::Time::controlDictName,args);Foam::fvMesh mesh(Foam::IOobject(Foam::polyMesh::defaultRegion,time.timeName(),time,Foam::IOobject::MUST_READ));
 int id=0;std::cout<<std::setprecision(17);
 for(double cp1:{0.,2.,20.})for(double dt:{.01,10.,10000.})for(double tol:{1e-2,1e-5,1e-8}){
 ModelConfig model;auto&p=model.physics;p.gasMode=ugkwp::GasMode::MixtureFrozen;
 for(auto&c:p.condensed){c.rho=1000;c.cp0=1000;c.cp1=cp1;c.conductivity=1000;c.Tmin=100;c.Tmax=3000;}
 for(auto&t:p.species){t.R=300;t.cp0=1000;t.Tmin=100;t.Tmax=3000;}p.liquid=p.condensed[0];p.spatialOrder=1;
 HostState base;base.solidMesh=testMesh(mesh);base.solid.resize(mesh.nCells());
 base.solidMesh.thermalBoundary.assign(base.solidMesh.owner.size(),ThermalBoundaryKind::ZeroGradient);
 for(std::size_t f=0;f<base.solidMesh.owner.size();++f)if(base.solidMesh.neighbour[f]<0)base.solidMesh.thermalBoundary[f]=ThermalBoundaryKind::FixedValue;
 for(auto&f:base.solidMesh.boundaryPrimitive)f.temperature=900;
 for(std::size_t c=0;c<base.solid.size();++c){auto&q=base.solid[c];q.condensed[0]=1000*base.solidMesh.volumes[c];const auto& center=base.solidMesh.cellCentres[c];
    const int globalCell=int(std::lround(2*center.x-.5))+2*int(std::lround(2*center.y-.5))+4*int(std::lround(2*center.z-.5));
    q.energy=q.condensed[0]*condensedE(p.condensed[0],400+25*globalCell);}
 CpuMaterialControls controls;controls.nonlinearMaxIterations=100;controls.nonlinearRelativeTolerance=tol;controls.energyAbsoluteTolerance=1e-8;controls.energyRelativeTolerance=1e-12;
 HostState next=base;CpuMaterialReport report;std::string error;
 if(Foam::Pstream::master())std::cout<<"CASE "<<id++<<' '<<cp1<<' '<<dt<<' '<<tol<<'\n';
 bool ok=nativeConduction(mesh,base,next,p,dt,controls,report,error);

 // Independent per-cell balance from returned inventories and Cartesian
 // Fourier face fluxes, not the nonlinear solver's reported global residual.
 Foam::wordList types(mesh.boundary().size(),Foam::fixedValueFvPatchScalarField::typeName);
 forAll(mesh.boundary(),patch)if(mesh.boundary()[patch].coupled())types[patch]=mesh.boundary()[patch].type();
 Foam::volScalarField checkT(Foam::IOobject("probeCheckT",mesh.time().timeName(),mesh,Foam::IOobject::NO_READ,Foam::IOobject::NO_WRITE,false),mesh,Foam::dimensionedScalar("T",Foam::dimTemperature,900),types);
 for(std::size_t c=0;c<next.solid.size();++c){const double specific=next.solid[c].energy/next.solid[c].condensed[0];
    checkT[c]=cp1==0?specific/1000:2*specific/(1000+std::sqrt(1000000+2*cp1*specific));}
 checkT.correctBoundaryConditions();
 Foam::surfaceScalarField conductiveFlux(-1000*mesh.magSf()*Foam::fvc::snGrad(checkT));
 std::vector<long double> balance(next.solid.size());
 for(std::size_t c=0;c<balance.size();++c)balance[c]=static_cast<long double>(next.solid[c].energy)-base.solid[c].energy;
 forAll(mesh.neighbour(),f){balance[mesh.owner()[f]]+=static_cast<long double>(dt)*conductiveFlux[f];balance[mesh.neighbour()[f]]-=static_cast<long double>(dt)*conductiveFlux[f];}
 forAll(mesh.boundary(),patch)forAll(mesh.boundary()[patch],f)balance[mesh.owner()[mesh.boundary()[patch].start()+f]]+=static_cast<long double>(dt)*conductiveFlux.boundaryField()[patch][f];
 double cellL1=0,cellMax=0;for(auto value:balance){cellL1+=std::abs(value);cellMax=std::max(cellMax,double(std::abs(value)));}
 Foam::reduce(cellL1,Foam::sumOp<double>());Foam::reduce(cellMax,Foam::maxOp<double>());
 if(Foam::Pstream::master())std::cout<<"CELL_BALANCE "<<cellL1<<' '<<cellMax<<'\n';
 long double energy=0;for(auto&q:next.solid)energy+=q.energy;
 if(Foam::Pstream::master())std::cout<<"RESULT "<<ok<<' '<<energy<<' '<<error<<'\n';
 }
}
