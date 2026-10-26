#include "gpu/MaterialThermalBackend.H"
#include "gpu/Buffer.H"
namespace chmt {
namespace {
__global__ void evaluateMaterialThermal(int n,const MaterialThermalInput* input,const Real* temperature,Real dt,
    const PhysicsConfig* physics,MaterialThermalCoefficients* output){
    const int c=blockIdx.x*blockDim.x+threadIdx.x;if(c<n)output[c]=materialThermalCoefficients(input[c],temperature[c],dt,*physics);
}
}
struct MaterialThermalBackend::Implementation {
    CudaFault fault;DeviceStream stream;
    Buffer<MaterialThermalInput> input;Buffer<PhysicsConfig> physics;
    Buffer<Real> temperature;Buffer<MaterialThermalCoefficients> output;
    bool ready=false;
};
MaterialThermalBackend::MaterialThermalBackend():data_(new Implementation){}
MaterialThermalBackend::~MaterialThermalBackend()=default;
bool MaterialThermalBackend::begin(const std::vector<MaterialThermalInput>& input,const PhysicsConfig& physics,std::string& error){
    auto& d=*data_;if(!d.ready){if(!d.stream.create(d.fault)){error=d.fault.message;return false;}d.ready=true;}
    PhysicsConfig copy=physics;copy.diagnostics=nullptr;
    if(!d.input.upload(input,d.fault)||!d.physics.uploadOne(copy,d.fault)||!d.output.resize(input.size(),d.fault)){error=d.fault.message;return false;}
    return true;
}
bool MaterialThermalBackend::evaluate(const std::vector<Real>& temperature,Real dt,std::vector<MaterialThermalCoefficients>& output,std::string& error){
    auto& d=*data_;
    if(!d.temperature.upload(temperature,d.fault)){error=d.fault.message;return false;}
    const int n=temperature.size();if(n)evaluateMaterialThermal<<<(n+127)/128,128,0,d.stream.get()>>>(n,d.input.data(),d.temperature.data(),dt,d.physics.data(),d.output.data());
    if(!d.fault.check(cudaPeekAtLastError(),"evaluate material implicit coefficients")||!d.output.download(output,d.stream.get())){error=d.fault.message;return false;}
    return true;
}
}
