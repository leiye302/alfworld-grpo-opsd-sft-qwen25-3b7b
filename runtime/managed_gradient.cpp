#include <cstddef>
#include <cstdio>
#include <unordered_map>
#include <mutex>
extern "C" int cudaGetDevice(int*);
extern "C" int cudaSetDevice(int);
extern "C" int cudaHostAlloc(void**, std::size_t, unsigned int);
extern "C" int cudaHostGetDevicePointer(void**, void*, unsigned int);
extern "C" int cudaFreeHost(void*);
extern "C" const char* cudaGetErrorString(int);
static std::unordered_map<void*,void*> hosts;
static std::mutex host_lock;

extern "C" void* managed_gradient_alloc(std::size_t size, int device, void*) {
    int previous = 0;
    cudaGetDevice(&previous);
    cudaSetDevice(device);
    void* host = nullptr;
    void* pointer = nullptr;
    int error = cudaHostAlloc(&host, size, 2); // cudaHostAllocMapped
    if (!error) error = cudaHostGetDevicePointer(&pointer, host, 0);
    if (error && host) cudaFreeHost(host);
    if (!error) {
        std::lock_guard<std::mutex> guard(host_lock);
        hosts[pointer]=host;
    }
    cudaSetDevice(previous);
    if (error) std::fprintf(stderr, "managed_gradient_alloc: %s\n", cudaGetErrorString(error));
    return pointer;
}

extern "C" void managed_gradient_free(void* pointer, std::size_t, int device, void*) {
    int previous = 0;
    cudaGetDevice(&previous);
    cudaSetDevice(device);
    void* host = nullptr;
    {
        std::lock_guard<std::mutex> guard(host_lock);
        host=hosts.at(pointer);
        hosts.erase(pointer);
    }
    int error = cudaFreeHost(host);
    cudaSetDevice(previous);
    if (error) std::fprintf(stderr, "managed_gradient_free: %s\n", cudaGetErrorString(error));
}
