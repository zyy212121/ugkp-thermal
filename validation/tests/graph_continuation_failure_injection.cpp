#include <atomic>
#include <cstdlib>
#include <new>
#include <cerrno>
#include <dlfcn.h>
#include <pthread.h>
thread_local bool failNextAllocation=false;
std::atomic<bool> failNextThread{false};
void* operator new(std::size_t size){
    if(failNextAllocation){failNextAllocation=false;throw std::bad_alloc();}
    if(auto* p=std::malloc(size?size:1))return p;throw std::bad_alloc();
}
void operator delete(void* p)noexcept{std::free(p);}
void operator delete(void* p,std::size_t)noexcept{std::free(p);}
extern "C" int pthread_create(pthread_t* thread,const pthread_attr_t* attributes,void*(*entry)(void*),void* argument)noexcept{
    if(failNextThread.exchange(false))return EAGAIN;
    using Create=int(*)(pthread_t*,const pthread_attr_t*,void*(*)(void*),void*);
    static auto real=reinterpret_cast<Create>(dlsym(RTLD_NEXT,"pthread_create"));
    return real(thread,attributes,entry,argument);
}
