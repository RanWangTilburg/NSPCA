//
// Created by user on 23-11-16.
//
#include "managed.h"

namespace cuExec {


    void *Managed::operator new(size_t size) {

        void *ptr;
        cudaMallocManaged(&ptr, size);
        return ptr;
    }

    void Managed::operator delete(void *ptr) {
        cudaFree(ptr);
    }
}

