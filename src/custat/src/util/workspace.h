#pragma once

#include <cstdlib>
#include "exception.h"

////A class that captures the workspace
////This class supports resize based on query results

namespace cuStat {

    class WorkSpace {
    private:
        size_t lspace_size;
        size_t rspace_size;
        void *lwork_ptr;
        void *rwork_ptr;

    public:
        WorkSpace(const size_t init_lsize = 8, const size_t init_rsize = 8) throw(cudaRunTimeError);

        ~WorkSpace() throw(cudaRunTimeError);

        size_t lsize();

        size_t rsize();

        void *lwork();

        void *rwork();

        void resize_lspace(const size_t newSize) throw(cudaRunTimeError);

        void resize_rspace(const size_t newSize) throw(cudaRunTimeError);
    };


    WorkSpace::WorkSpace(const size_t default_lsize, const size_t default_rsize) throw(cudaRunTimeError) {
        try {
            handle_cuda_runtime_error(cudaMalloc((void **) &lwork_ptr, default_lsize));
            lspace_size = default_lsize;
            handle_cuda_runtime_error(cudaMalloc((void **) &rwork_ptr, default_rsize));
            rspace_size = default_rsize;
        }
        catch (cudaRunTimeError error) {
            throw error;
        }
    }


    WorkSpace::~WorkSpace() throw(cudaRunTimeError) {
        try {
            handle_cuda_runtime_error(cudaFree(lwork_ptr));
            handle_cuda_runtime_error(cudaFree(rwork_ptr));
        }
        catch(cudaRunTimeError error){
            throw error;
        }
    }


    void WorkSpace::resize_lspace(const size_t newSize) throw(cudaRunTimeError) {
        try {
            if (newSize > lspace_size) {
                handle_cuda_runtime_error(cudaFree(lwork_ptr));
                handle_cuda_runtime_error(cudaMalloc((void **) &lwork_ptr, newSize));
                lspace_size = newSize;
            }
        }
        catch (cudaRunTimeError error){
            throw error;
        }
    }


    void *WorkSpace::lwork() {
        return lwork_ptr;
    }


    size_t WorkSpace::lsize() {
        return lspace_size;
    }


    void *WorkSpace::rwork() {
        return rwork_ptr;
    }

    size_t WorkSpace::rsize() {
        return rspace_size;
    }


    void WorkSpace::resize_rspace(const size_t newSize) throw(cudaRunTimeError) {
        try {
            if (newSize > rsize()) {
                handle_cuda_runtime_error(cudaFree(rwork_ptr));
                handle_cuda_runtime_error(cudaMalloc((void **) &rwork_ptr, newSize));
                rspace_size = newSize;
            }
        }
        catch (cudaRunTimeError error){
            throw error;
        }

    }

}