#include "fvCFD.H"
#include "mesh/Geometry.H"
#include "materials/CpuMaterialDriver.H"
#include "materials/MaterialCaloric.H"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
namespace {
// Import this test's actual OpenFOAM topology without including ModelIO and
// claiming the test executable has a CUDA/application build identity.
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
    for(const auto& patch:mesh.boundaryMesh()){assert(patch.type()=="wall");
        for(int f=0;f<patch.size();++f)out.boundaryKind[patch.start()+f]=chmt::BoundaryKind::NoSlip;}
    std::string error;assert(chmt::rebuildGeometry(out,error));out.oldVolumes=out.volumes;return out;
}
chmt::IntervalHistory history(int pieces){chmt::IntervalHistory h;chmt::CouplingInterval interval;interval.identity.sequence=1;interval.end=10000;std::string error;assert(h.begin(interval,error));
    for(int i=0;i<pieces;++i){chmt::GasIntervalRecord r;r.microSequence=i;r.begin=10000.*i/pieces;r.end=10000.*(i+1)/pieces;assert(h.appendAccepted(r,error));}return h;}
}
int main(int argc,char** argv){
    Foam::argList::noParallel();Foam::argList args(argc,argv);if(!args.checkRootCase())return 2;
    Foam::Time time(Foam::Time::controlDictName,args);Foam::fvMesh mesh(Foam::IOobject(Foam::polyMesh::defaultRegion,time.timeName(),time,Foam::IOobject::MUST_READ));
    assert(mesh.nCells()==8);const Foam::pointField original=mesh.points();const bool moving=mesh.moving();
    const Foam::label timeIndex=time.timeIndex();const Foam::scalar timeValue=time.value();
    forAll(mesh.C(),c){assert(std::abs(mesh.V()[c]-.125)<1e-12);for(int j=0;j<3;++j)assert(std::abs(mesh.C()[c][j]-.25)<1e-12||std::abs(mesh.C()[c][j]-.75)<1e-12);}
    for(int nonlinear=0;nonlinear<2;++nonlinear){
        chmt::ModelConfig model;auto& p=model.physics;p.enableGas=false;p.spatialOrder=1;p.minDt=1e-9;p.maxDt=10000;p.cfl=.4;p.modelFingerprint=1;
        for(int c=0;c<chmt::Nc;++c){auto& t=p.condensed[c];t.rho=1000;t.cp0=1000;t.cp1=nonlinear?.2:0;t.e0=-1e8;t.conductivity=1000;t.Tmin=100;t.Tmax=2000;}
        chmt::HostState base;base.solidMesh=testMesh(mesh);base.solid.resize(mesh.nCells());long double before=0;
        forAll(mesh.C(),c){const auto x=mesh.C()[c];const chmt::Real T=400+100*x.x()+30*x.y()+80*x.z();auto& q=base.solid[c];q.condensed[0]=1000*mesh.V()[c];q.energy=q.condensed[0]*chmt::condensedE(p.condensed[0],T);before+=q.energy;}
        chmt::CpuMaterialDriver driver(model,&mesh);chmt::CpuMaterialControls controls;controls.maxSubstep=10000;
        chmt::HostState endpoint=base;endpoint.time=10000;chmt::HostState result;chmt::WallProgram wall;chmt::CpuMaterialReport report;std::string error;
        if(!driver.advanceCandidate(base,endpoint,history(1),controls,result,wall,report,error)){std::cerr<<error<<'\n';return 1;}
        long double after=0;chmt::Real maxError=0;std::vector<chmt::Real> temperatures(result.solid.size());
        for(std::size_t c=0;c<result.solid.size();++c){chmt::MaterialPrimitive material;assert(chmt::recoverMaterial(result.solid[c],result.solidMesh.volumes[c],p,material));after+=result.solid[c].energy;
            temperatures[c]=material.temperature;assert(material.temperature>=452.5&&material.temperature<=557.5);
            if(!nonlinear){const auto x=mesh.C()[c];const chmt::Real initial=400+100*x.x()+30*x.y()+80*x.z();
                // All three independent alternating 2-cell modes have lambda
                // 2*alpha/dx^2 with insulated ends. Exact discrete BE amplitude.
                const chmt::Real exact=505+(initial-505)/(1+2*.001*10000/(.5*.5));maxError=std::max(maxError,std::abs(material.temperature-exact));}}
        assert(std::abs(after-before)<1e-8L*std::abs(before));if(!nonlinear)assert(maxError<1e-7);
        assert(report.linearSolves>0);assert(mesh.moving()==moving);assert(mesh.points()==original);
        assert(time.timeIndex()==timeIndex&&time.value()==timeValue);
        // Check every actual native result against the local nonlinear BE
        // equation, not just the global energy sum (which cannot detect a
        // conservative but incorrect conduction operator).
        forAll(mesh.C(),c){const auto x=mesh.C()[c];const chmt::Real initial=400+100*x.x()+30*x.y()+80*x.z();
            const chmt::Real T=temperatures[c],mass=base.solid[c].condensed[0];
            long double residual=static_cast<long double>(mass)*(1000+.5*(nonlinear?.2:0)*(T+initial))*(T-initial);
            long double scale=std::abs(residual);
            forAll(mesh.neighbour(),f){const int a=mesh.owner()[f],b=mesh.neighbour()[f];if(c!=a&&c!=b)continue;
                const long double transfer=10000.L*1000*.25/.5*(T-temperatures[c==a?b:a]);residual+=transfer;scale+=std::abs(transfer);}
            assert(std::abs(residual)<1e-3L+1e-8L*scale);
        }
        chmt::HostState refined;chmt::WallProgram refinedWall;chmt::CpuMaterialReport refinedReport;
        assert(driver.advanceCandidate(base,endpoint,history(100),controls,refined,refinedWall,refinedReport,error));
        assert(refinedReport.materialSteps==report.materialSteps&&refinedReport.materialRhsEvaluations==report.materialRhsEvaluations&&refinedReport.linearSolves==report.linearSolves);
        for(std::size_t c=0;c<result.solid.size();++c)assert(refined.solid[c].energy==result.solid[c].energy);
        // A real sparse solve followed by deliberate nonlinear nonconvergence
        // must leave both caller-owned outputs and the accepted native mesh
        // untouched. This is not an orchestration mock.
        chmt::CpuMaterialControls rejectedControls=controls;rejectedControls.nonlinearMaxIterations=1;
        chmt::HostState rejected;rejected.time=-123;rejected.solid.resize(1);rejected.solid[0].energy=17;rejected.solidSweepRemainder={123};
        chmt::WallProgram rejectedWall;rejectedWall.interval.identity.sequence=987;
        chmt::CpuMaterialReport rejectedReport;
        assert(!driver.advanceCandidate(base,endpoint,history(1),rejectedControls,rejected,rejectedWall,rejectedReport,error));
        assert(error=="OpenFOAM material caloric nonlinear iteration failed");assert(rejectedReport.linearSolves==1);
        assert(rejected.time==-123&&rejected.solid.size()==1&&rejected.solid[0].energy==17);
        assert(rejected.solidSweepRemainder==std::vector<chmt::Real>{123});
        assert(rejectedWall.interval.identity.sequence==987&&rejectedWall.knots.empty());
        assert(mesh.points()==original&&mesh.moving()==moving&&time.timeIndex()==timeIndex&&time.value()==timeValue);
        forAll(mesh.C(),c)assert(std::abs(mesh.V()[c]-.125)<1e-12);
        std::cout<<(nonlinear?"NATIVE_OF10 nonlinear caloric energy check":"NATIVE_OF10 exact three-axis discrete backward-Euler modal check")<<" passed; linear solves="<<report.linearSolves<<", material RHS="<<report.materialRhsEvaluations<<", max modal error="<<maxError<<'\n';
    }
    {
        // Native coupled calorics component test: stationary condensed matter
        // and positive NASA7 pore gas share the actual nonlinear sparse solve.
        // This is not a pure-flow CHMT application entry or a physical mechanism.
        chmt::ModelConfig model;auto& p=model.physics;p.enableGas=false;p.spatialOrder=1;p.minDt=1e-9;p.maxDt=10000;p.cfl=.4;p.modelFingerprint=1;
        p.gasConductivity=10;p.permeability=0;
        for(int c=0;c<chmt::Nc;++c){auto& t=p.condensed[c];t.rho=1000;t.cp0=1000;t.cp1=.2;t.e0=-1e8;t.conductivity=1000;t.Tmin=200;t.Tmax=2000;}
        auto& gas=p.species[0];gas.model=ugkwp::SpeciesThermoModel::NASA7;gas.R=300;gas.Tmin=200;gas.Tmid=1000;gas.Tmax=3000;
        const chmt::Real low[7]={3.5,.001,0,0,0,-10000,2},high[7]={4,.0005,0,0,0,-10250,2};
        for(int j=0;j<7;++j){gas.nasa7[j]=low[j];gas.nasa7[j+7]=high[j];}
        // Independent mass-specific e=h-R*T, including unchanged integration
        // constants. No production gas/condensed evaluator supplies the oracle.
        const auto poreEnergy=[](long double T){return T<=1000
            ?300.L*(-10000.L+2.5L*T+.0005L*T*T)
            :300.L*(-10250.L+3.L*T+.00025L*T*T);};
        const auto condensedEnergy=[](long double T){return -1e8L+1000.L*T+.1L*T*T;};
        chmt::HostState base;base.solidMesh=testMesh(mesh);base.solid.resize(mesh.nCells());
        std::vector<chmt::Real> initialTemperature(mesh.nCells()),temperature(mesh.nCells());long double before=0;
        forAll(mesh.C(),c){const auto x=mesh.C()[c];const chmt::Real T=800+200*x.x()+100*x.y()+150*x.z();initialTemperature[c]=T;
            auto& q=base.solid[c];q.condensed[0]=800*mesh.V()[c];q.pore[0]=4*mesh.V()[c];q.porosity=.2;
            q.energy=static_cast<chmt::Real>(q.condensed[0]*condensedEnergy(T)+q.pore[0]*poreEnergy(T));before+=q.energy;}
        chmt::CpuMaterialDriver driver(model,&mesh);chmt::CpuMaterialControls controls;controls.maxSubstep=10000;
        chmt::HostState endpoint=base;endpoint.time=10000;chmt::HostState result;chmt::WallProgram wall;chmt::CpuMaterialReport report;std::string error;
        if(!driver.advanceCandidate(base,endpoint,history(1),controls,result,wall,report,error)){std::cerr<<"NASA7 pore conduction: "<<error<<'\n';return 1;}
        long double after=0,variation=0,maximumResidual=0;bool crossedBreakpoint=false;
        for(std::size_t c=0;c<result.solid.size();++c){chmt::MaterialPrimitive material;
            assert(chmt::recoverMaterial(result.solid[c],result.solidMesh.volumes[c],p,material));temperature[c]=material.temperature;
            assert(material.temperature>=912.5&&material.temperature<=1137.5);
            assert(result.solid[c].condensed[0]==base.solid[c].condensed[0]);
            for(int species=0;species<chmt::Ns;++species)assert(result.solid[c].pore[species]==base.solid[c].pore[species]);
            const long double expected=base.solid[c].condensed[0]*condensedEnergy(material.temperature)+base.solid[c].pore[0]*poreEnergy(material.temperature);
            assert(std::abs(static_cast<long double>(result.solid[c].energy)-expected)<5e-5L);
            after+=result.solid[c].energy;variation+=std::abs(static_cast<long double>(result.solid[c].energy)-base.solid[c].energy);
            crossedBreakpoint=crossedBreakpoint||(initialTemperature[c]<1000&&material.temperature>1000);
        }
        assert(crossedBreakpoint);assert(std::abs(after-before)<1e-3L+1e-9L*variation);
        // k_eff=(1-phi)*k_condensed+phi*k_gas=802 W/(m K).
        // The derivative-independent residual catches incorrect NASA cv or
        // e=cp(T)*T even when a solver still conserves the global energy sum.
        forAll(mesh.C(),c){const long double T=temperature[c],oldT=initialTemperature[c];
            long double residual=base.solid[c].condensed[0]*(1000.L+.1L*(T+oldT))*(T-oldT)
                +base.solid[c].pore[0]*(poreEnergy(T)-poreEnergy(oldT));
            long double scale=std::abs(residual);
            forAll(mesh.neighbour(),f){const int a=mesh.owner()[f],b=mesh.neighbour()[f];if(c!=a&&c!=b)continue;
                const long double transfer=10000.L*802.L*.25L/.5L*(T-temperature[c==a?b:a]);residual+=transfer;scale+=std::abs(transfer);}
            maximumResidual=std::max(maximumResidual,std::abs(residual));
            assert(std::abs(residual)<1e-3L+1e-8L*scale);
        }
        assert(report.linearSolves>0&&report.materialSteps==1);assert(report.budgetDelta.boundaryEnergy==0);
        assert(mesh.points()==original&&mesh.moving()==moving&&time.timeIndex()==timeIndex&&time.value()==timeValue);
        std::cout<<"NATIVE_OF10 NASA7 pore/condensed conduction and local nonlinear backward-Euler residual passed; linear solves="
            <<report.linearSolves<<", energy residual="<<static_cast<double>(after-before)<<", max local residual="<<static_cast<double>(maximumResidual)<<'\n';
    }
    {
        chmt::ModelConfig model;auto& p=model.physics;p.enableGas=false;p.spatialOrder=1;p.minDt=1e-9;p.maxDt=10000;p.cfl=.4;p.modelFingerprint=1;
        for(int c=0;c<chmt::Nc;++c){auto& t=p.condensed[c];t.rho=1000;t.cp0=1000;t.e0=-1e8;t.conductivity=1000;t.Tmin=100;t.Tmax=2000;}
        chmt::HostState base;base.solidMesh=testMesh(mesh);base.solid.resize(mesh.nCells());
        forAll(mesh.C(),c){auto& q=base.solid[c];q.condensed[0]=1000*mesh.V()[c];q.energy=q.condensed[0]*chmt::condensedE(p.condensed[0],400);}
        base.solidMesh.thermalBoundary.assign(mesh.nFaces(),chmt::ThermalBoundaryKind::ZeroGradient);
        forAll(mesh.boundary(),patch)forAll(mesh.boundary()[patch],f){const int face=mesh.boundary()[patch].start()+f;
            base.solidMesh.boundaryPrimitive[face].temperature=650;base.solidMesh.thermalBoundary[face]=chmt::ThermalBoundaryKind::FixedValue;}
        chmt::HostState endpoint=base;endpoint.time=10000;chmt::CpuMaterialDriver driver(model,&mesh);chmt::CpuMaterialControls controls;
        chmt::HostState result;chmt::WallProgram wall;chmt::CpuMaterialReport report;std::string error;
        if(!driver.advanceCandidate(base,endpoint,history(1),controls,result,wall,report,error)){std::cerr<<error<<'\n';return 1;}
        // Each corner cell has three fixed-temperature faces of area .25 and
        // centre-to-boundary distance .25. Internal gradients remain zero.
        const chmt::Real ratio=10000*3*1000*.25/.25/(125*1000),exact=(400+ratio*650)/(1+ratio);
        long double sensibleChange=0;
        for(std::size_t c=0;c<result.solid.size();++c){chmt::MaterialPrimitive material;assert(chmt::recoverMaterial(result.solid[c],result.solidMesh.volumes[c],p,material));
            assert(std::abs(material.temperature-exact)<1e-7);sensibleChange+=125000.L*(material.temperature-400);}
        assert(report.budgetDelta.boundaryEnergy<0);
        assert(std::abs(sensibleChange+report.budgetDelta.boundaryEnergy)<1e-3L);
        assert(result.budget.boundaryEnergy==report.budgetDelta.boundaryEnergy);
        assert(mesh.points()==original&&mesh.moving()==moving&&time.timeIndex()==timeIndex&&time.value()==timeValue);
        std::cout<<"NATIVE_OF10 fixed-temperature heating and signed boundary-energy closure passed\n";
    }
    {
        // The conservative geometry and Foam independently sum the same real
        // tetrahedral/pyramidal volumes. A valid dense inventory near the EOS
        // roundoff boundary must not be tested against both rounded volumes.
        bool exercised=false;
        for(int variant=0;variant<8&&!exercised;++variant){
            Foam::pointField points(original);
            forAll(points,i){const auto p=original[i];points[i]=Foam::point(
                .0131+.001*variant+.01*p.x()+.0013*p.y(),
                .0237+.0037*p.y()+.0007*p.z(),.0419+.0189*p.z()+.0009*p.x());}
            Foam::fvMesh probe(Foam::IOobject(mesh.name(),time.timeName(),time,Foam::IOobject::NO_READ,Foam::IOobject::NO_WRITE,false),
                std::move(points),Foam::faceList(mesh.faces()),Foam::labelList(mesh.faceOwner()),Foam::labelList(mesh.faceNeighbour()));
            Foam::List<Foam::polyPatch*> patches(mesh.boundaryMesh().size());
            forAll(patches,patch)patches[patch]=mesh.boundaryMesh()[patch].clone(probe.boundaryMesh()).ptr();probe.addFvPatches(patches);
            chmt::ModelConfig model;auto& p=model.physics;p.enableGas=false;p.spatialOrder=1;p.minDt=1e-9;p.maxDt=10000;p.cfl=.4;p.modelFingerprint=1;
            for(int c=0;c<chmt::Nc;++c){auto& t=p.condensed[c];t.rho=10;t.cp0=1000;t.conductivity=0;t.Tmin=100;t.Tmax=2000;}
            chmt::HostState base;base.solidMesh=testMesh(probe);base.solid.resize(probe.nCells());
            forAll(probe.C(),c){auto& q=base.solid[c];q.condensed[0]=10*base.solidMesh.volumes[c];q.energy=q.condensed[0]*chmt::condensedE(p.condensed[0],600);}
            int chosen=-1;
            forAll(probe.C(),c)for(int eps=24;eps<=32&&chosen<0;++eps){
                const chmt::Real V=base.solidMesh.volumes[c];auto q=base.solid[c];
                q.condensed[0]=10*(V+eps*std::numeric_limits<chmt::Real>::epsilon()*V);q.energy=q.condensed[0]*chmt::condensedE(p.condensed[0],600);
                chmt::MaterialPrimitive canonical,native;
                if(chmt::recoverMaterial(q,V,p,canonical)&&!chmt::recoverMaterial(q,probe.V()[c],p,native)){
                    base.solid[c]=q;chosen=c;
                }
            }
            if(chosen<0)continue;
            chmt::CpuMaterialDriver driver(model,&probe);chmt::CpuMaterialControls controls;
            chmt::HostState endpoint=base;endpoint.time=10000;chmt::HostState result;chmt::WallProgram wall;chmt::CpuMaterialReport report;std::string error;
            if(!driver.advanceCandidate(base,endpoint,history(1),controls,result,wall,report,error)){
                std::cerr<<"canonical-volume regression: "<<error<<" cell="<<chosen<<" hostV="<<base.solidMesh.volumes[chosen]<<" nativeV="<<probe.V()[chosen]<<'\n';return 1;
            }
            for(std::size_t c=0;c<base.solid.size();++c){
                assert(result.solid[c].condensed[0]==base.solid[c].condensed[0]);
                chmt::MaterialPrimitive material;assert(chmt::recoverMaterial(result.solid[c],result.solidMesh.volumes[c],p,material));
                assert(std::abs(material.temperature-600)<1e-8);
            }
            exercised=true;
        }
        assert(exercised);
        std::cout<<"NATIVE_OF10 canonical material-volume ownership and unchanged dense inventory passed\n";
    }
}
