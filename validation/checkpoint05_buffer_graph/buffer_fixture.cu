#include "applications/CHMT/gpu/Buffer.H"
#include <stdexcept>
#include <iostream>
#include <type_traits>
static void require(bool ok,const char* name){if(!ok)throw std::runtime_error(name);}
template<class T> void exercise(){
    chmt::CudaFault fault;chmt::DeviceStream stream;require(stream.create(fault),"stream");
    chmt::Buffer<T> a,b;std::vector<T> original(16,T(7)),out;
    require(a.upload(original,fault),"upload");auto* address=a.data();
    require(a.capacity()==16&&a.size()==16,"initial dimensions");
    require(a.resize(4,fault)&&a.data()==address&&a.capacity()==16,"shrink reuse");
    require(a.zero(stream.get())&&stream.sync(),"zero prefix");
    out.resize(16);require(fault.check(cudaMemcpy(out.data(),address,16*sizeof(T),cudaMemcpyDeviceToHost),"inspect allocation"),"read capacity");
    for(int i=0;i<16;++i)require(out[i]==(i<4?T(0):T(7)),"zero only logical range");
    require(a.resize(12,fault)&&a.data()==address,"regrow reuse");
    require(a.resize(0,fault)&&a.data()==nullptr&&a.capacity()==16,"empty retains private capacity");
    require(a.resize(1,fault)&&a.data()==address,"reuse after zero");
    require(a.upload(std::vector<T>{T(2),T(3)},fault)&&a.data()==address,"upload within capacity");
    require(b.upload(std::vector<T>(8,T(9)),fault),"destination allocation");auto* baddress=b.data();
    require(b.copy(a,fault,stream.get())&&b.size()==2&&b.capacity()==8&&b.data()==baddress,"copy uses logical size");
    require(b.download(out,stream.get())&&out==std::vector<T>({T(2),T(3)}),"download only logical range");
    std::vector<T> tail(8);require(fault.check(cudaMemcpy(tail.data(),baddress,8*sizeof(T),cudaMemcpyDeviceToHost),"copy tail"),"tail read");
    for(int i=2;i<8;++i)require(tail[i]==T(9),"copy leaves capacity tail");
    chmt::Buffer<T> moved(std::move(b));require(moved.data()==baddress&&moved.capacity()==8&&b.size()==0&&b.capacity()==0,"move ctor");
    a.swap(moved);require(a.capacity()==8&&moved.capacity()==16,"swap capacity");
    b=std::move(a);require(b.capacity()==8&&a.capacity()==0,"move assign");
    require(b.resize(17,fault)&&b.size()==17&&b.capacity()==17,"grow allocation");
    require(b.release()&&b.size()==0&&b.capacity()==0&&b.data()==nullptr,"release resets");
    require(moved.release(),"release retained capacity");
    require(!fault.runtimeFailure,"no CUDA failure");
    std::cout<<"BUFFER_PASS bits="<<8*sizeof(T)<<"\n";
}
int main(){try{exercise<float>();exercise<double>();return 0;}catch(const std::exception& e){std::cerr<<"BUFFER_FAIL "<<e.what()<<"\n";return 1;}}
