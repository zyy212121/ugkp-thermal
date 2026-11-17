#pragma once
__device__ RadiationReal domHomogeneousIncoming(Geometry g,Fields f,const RadiationReal* x,int angle,int face){
 const int cell=g.boundaryOwner[face];RadiationReal value=0;
 if(g.boundaryType[face]==RadiationBoundary::Specular){const int row=angle*g.boundary+face;for(int e=g.reflectionOffsets[row];e<g.reflectionOffsets[row+1];++e)value+=g.reflectionWeights[e]*x[std::size_t(g.reflectionDonors[e])*g.cells+cell];}
 else if(g.boundaryType[face]==RadiationBoundary::Diffuse){for(int a=0;a<g.directions;++a){RadiationReal mu=g.cosine[a*g.boundary+face];if(mu>0)value+=g.weights[a]*mu*x[std::size_t(a)*g.cells+cell];}value*=(1-f.emissivity[face])/g.denominator[face];}
 return value;
}
__global__ void domApply(Geometry g,Fields f,const OrderedUpwindDevice* matrices,const RadiationReal* x,RadiationReal* y){
 const std::size_t row=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(row>=std::size_t(g.cells)*g.directions)return;
 const int angle=row/g.cells,cell=row%g.cells;const auto m=matrices[angle];const RadiationReal* xa=x+std::size_t(angle)*g.cells;
 RadiationReal value=(g.transportDiagonal[row]+(f.absorption[cell]+f.scattering[cell])*g.volume[cell])*x[row];
 for(int j=m.offsets[cell];j<m.offsets[cell+1];++j){int e=m.residualOrder[j];RadiationReal v=m.flux[e]*xa[m.neighbour[e]];if(m.subtract[e])value+=v;else value-=v;}
 for(int j=g.boundaryOffsets[cell];j<g.boundaryOffsets[cell+1];++j){int face=g.boundaryCells[j];RadiationReal flux=g.boundaryFlux[angle*g.boundary+face];if(flux<0)value+=flux*domHomogeneousIncoming(g,f,x,angle,face);}
 RadiationReal mean=0,G=0;
 for(int a=0;a<g.directions;++a){RadiationReal v=x[std::size_t(a)*g.cells+cell];G+=g.weights[a]*v;if(f.scattering[cell]!=0){const auto d=mieLogBracket(f.table.diameter,f.table.diameters,f.query[cell].diameter);const auto t=mieLinearBracket(f.table.temperature,f.table.temperatures,f.query[cell].temperature);mean+=mieDetail::phaseProbability(f.table,d,t,g.weights,f.phaseBrackets,f.scaling+std::size_t(cell)*g.directions,g.directions,angle,a)*v;}}
 RadiationReal feedback=f.thermal&&f.thermal->linearizedEmission&&f.thermal->thermalFeedback?f.thermal->thermalFeedback[cell]:0;
 y[row]=value-g.volume[cell]*(f.scattering[cell]*mean+f.absorption[cell]*feedback*G/fourPi);
}
__global__ void domRhs(Geometry g,Fields f,RadiationReal* rhs){
 const std::size_t row=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(row>=std::size_t(g.cells)*g.directions)return;const int angle=row/g.cells,cell=row%g.cells;
 RadiationReal emission=f.thermal&&f.thermal->linearizedEmission&&f.thermal->thermalFeedback?f.thermal->linearizedEmission[cell]:f.emission[cell];RadiationReal value=g.volume[cell]*f.absorption[cell]*emission;
 for(int j=g.boundaryOffsets[cell];j<g.boundaryOffsets[cell+1];++j){int face=g.boundaryCells[j];RadiationReal flux=g.boundaryFlux[angle*g.boundary+face];if(flux<0){RadiationReal incoming=0;auto type=g.boundaryType[face];if(type==RadiationBoundary::Fixed)incoming=g.fixed[face];else if(type==RadiationBoundary::Black)incoming=f.wallBlack[face];else if(type==RadiationBoundary::Diffuse)incoming=f.emissivity[face]*f.wallBlack[face];value-=flux*incoming;}}
 rhs[row]=value;
}
__global__ void domRestrict(Geometry g,const RadiationReal* input,RadiationReal* scalar){
 const int cell=blockIdx.x*blockDim.x+threadIdx.x;if(cell>=g.cells)return;RadiationReal sum=0,weight=0;for(int a=0;a<g.directions;++a){sum+=g.weights[a]*input[std::size_t(a)*g.cells+cell];weight+=g.weights[a];}scalar[cell]=sum/weight;
}
__global__ void domProlong(Geometry g,const RadiationReal* scalar,RadiationReal* angular){
 const std::size_t row=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(row<std::size_t(g.cells)*g.directions)angular[row]=scalar[row%g.cells];
}
__device__ void domSweepCell(int cell,OrderedUpwindDevice matrix,Geometry g,Fields f,int angle,const RadiationReal* rhs,RadiationReal* solution){
 RadiationReal value=rhs[cell];
 for(int j=matrix.offsets[cell];j<matrix.offsets[cell+1];++j){int neighbour=matrix.neighbour[j];if(neighbour<cell){RadiationReal v=matrix.flux[j]*solution[neighbour];if(matrix.subtract[j])value-=v;else value+=v;}}
 const RadiationReal diagonal=g.transportDiagonal[std::size_t(angle)*g.cells+cell]+(f.absorption[cell]+f.scattering[cell])*g.volume[cell];
 solution[cell]=diagonal!=0?value/diagonal:RadiationReal(0);
}
__global__ void domSweepLevel(int begin,int end,OrderedUpwindDevice matrix,Geometry g,Fields f,int angle,const RadiationReal* rhs,RadiationReal* solution){
 int row=begin+blockIdx.x*blockDim.x+threadIdx.x;if(row>=end)return;domSweepCell(matrix.levelCells[row],matrix,g,f,angle,rhs,solution);
}
__global__ void domSweepNarrowLevels(int firstLevel,int endLevel,const int* levelOffsets,OrderedUpwindDevice matrix,Geometry g,Fields f,int angle,const RadiationReal* rhs,RadiationReal* solution){
 for(int level=firstLevel;level<endLevel;++level){
  for(int row=levelOffsets[level]+threadIdx.x;row<levelOffsets[level+1];row+=blockDim.x)domSweepCell(matrix.levelCells[row],matrix,g,f,angle,rhs,solution);
  __syncthreads();
 }
}
__global__ void domLowFace(Geometry g,RadiationReal* upper,RadiationReal* lower){
 int face=blockIdx.x*blockDim.x+threadIdx.x;if(face>=g.faces)return;RadiationReal sum=0,weight=0;for(int a=0;a<g.directions;++a){sum+=g.weights[a]*fabs(g.internalFlux[a*g.faces+face]);weight+=g.weights[a];}upper[face]=lower[face]=-sum/(2*weight);
}
__global__ void domLowRows(Geometry g,Fields f,const int* cellOffsets,const int* cellFaces,const RadiationReal* upper,RadiationReal* diag){
 int cell=blockIdx.x*blockDim.x+threadIdx.x;if(cell>=g.cells)return;RadiationReal feedback=f.thermal&&f.thermal->linearizedEmission&&f.thermal->thermalFeedback?f.thermal->thermalFeedback[cell]:0;RadiationReal value=f.absorption[cell]*(1-feedback)*g.volume[cell];for(int j=cellOffsets[cell];j<cellOffsets[cell+1];++j){int face=cellFaces[j];if(face>=0&&face<g.faces)value-=upper[face];}diag[cell]=value;
}
__global__ void domLowBoundary(Geometry g,Fields f,RadiationReal* bdiag){
 int face=blockIdx.x*blockDim.x+threadIdx.x;if(face>=g.boundary)return;RadiationReal sum=0,weight=0;for(int a=0;a<g.directions;++a){RadiationReal flux=g.boundaryFlux[a*g.boundary+face];if(flux>0)sum+=g.weights[a]*flux;weight+=g.weights[a];}auto type=g.boundaryType[face];RadiationReal factor=type==RadiationBoundary::Specular?0:type==RadiationBoundary::Diffuse?f.emissivity[face]:1;bdiag[face]=factor*sum/weight;
}
