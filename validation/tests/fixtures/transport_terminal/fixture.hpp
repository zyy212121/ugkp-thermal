#include "materials/MaterialTransport.H"
#include <cassert>
#include <iostream>
using namespace chmt;
PhysicsConfig config(){PhysicsConfig p; p.spatialOrder=1;p.permeability=1e-14;p.poreViscosity=1e-5;
 for(int c=0;c<Nc;++c){auto&t=p.condensed[c];t.rho=1000;t.cp0=1000;t.conductivity=1;t.Tmin=100;t.Tmax=3000;}
 for(int s=0;s<Ns;++s){auto&t=p.species[s];t.R=287;t.cp0=1000;t.Tmin=100;t.Tmax=3000;}return p;}
void fixture(PhysicsConfig& p,HostMesh& m,std::vector<SolidQ>& q){p=config();m.volumes.assign(4,.001);m.cellCentres={{0,0,0},{.1,0,0},{0,.1,0},{0,0,.1}};
 m.owner={0,0,0};m.neighbour={1,2,3};m.areaVectors={{.01,0,0},{0,.01,0},{0,0,.01}};m.faceCentres={{.05,0,0},{0,.05,0},{0,0,.05}};
 m.periodicPartner.assign(3,-1);m.boundaryKind.assign(3,BoundaryKind::Internal);
 m.cellFaceOffsets={0,3,4,5,6};m.cellFaces={0,1,2,0,1,2};m.cellFaceSigns={1,1,1,-1,-1,-1};
 q.resize(4);for(int c=0;c<4;++c){q[c].condensed[0]=.9;q[c].pore[0]=c?1e-4:2e-4;q[c].porosity=.1;q[c].energy=.9*condensedE(p.condensed[0],600)+q[c].pore[0]*speciesE(p.species[0],600);}
}
