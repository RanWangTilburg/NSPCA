#pragma once

#include "cuda_runtime.h"

////This is the class that overload the "new" and "delete" operator so that class
////that inherited from "managed" can be passed to kernel argument


namespace cuStat {
    namespace internal {
        class Managed {
        public:
            void *operator new(size_t size);

            void operator delete(void *ptr);
        };


        void *Managed::operator new(size_t size) {
            void *ptr;
            cudaMallocManaged(&ptr, size);
            return ptr;
        }

        void Managed::operator delete(void *ptr) {
            cudaFree(ptr);
        }

    }
}