#include <execinfo.h>
#include <signal.h>
#include <unistd.h>
static void handler(int){void* stack[64];int n=backtrace(stack,64);backtrace_symbols_fd(stack,n,2);_exit(139);}
__attribute__((constructor)) static void install(){struct sigaction a{};a.sa_handler=handler;sigemptyset(&a.sa_mask);sigaction(SIGSEGV,&a,nullptr);}
