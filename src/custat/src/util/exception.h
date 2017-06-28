#pragma once

#include <exception>
#include <string>
#include <iostream>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolver_common.h>
#include <cusolverDn.h>

////Exception classes as well as functions that throws exceptions
////Each exception class contains information on the exception
////Note that clients should implement their own methods of handling exceptions


namespace cuStat {


    ////Exception class that captures cuda runtime errors
    class cudaRunTimeError : public std::exception {
    private:
        std::string errorMessage;
        cudaError_t error;
    public:
        cudaRunTimeError(cudaError_t _error);
        virtual const char * what() const noexcept ;
    };

    ////Exception class that captures cuSolver errors
    class cuSolverError : public std::exception {
    private:
        std::string errorMessage;
        cusolverStatus_t status;
    public:
        cuSolverError(cusolverStatus_t _status);
        virtual const char * what() const noexcept ;
    };

    ////Exception class that captures cuBlas errors
    class cuBlasError : public std::exception {
    private:
        std::string errorMessage;
        cublasStatus_t status;
    public:
        cuBlasError(cublasStatus_t _status);
        virtual const char * what() const noexcept ;
    };

    ////Exception class that captures errors that occur if the matrices dimensions do not match

    class dimMisMatch : public std::exception {
    private:
        std::string errorMessage;
    public:
        dimMisMatch(const std::string _errorMessage);

    };

    class numThreadsError : public std::exception {
    private:
        std::string errorMessage;
    public:
        numThreadsError();
        virtual const char * what() const noexcept ;
    };

    class lowerBoundLargerThanUpperBound : public std::exception{
    private:
        std::string errorMessage;
    public:
        lowerBoundLargerThanUpperBound();
        virtual const char* what() const noexcept;
    };

    lowerBoundLargerThanUpperBound::lowerBoundLargerThanUpperBound() {
        errorMessage = "Lower bound is larger than higher bound";
    }

    const char *lowerBoundLargerThanUpperBound::what() const  noexcept {
        return errorMessage.c_str();
    }


    void handle_cuda_runtime_error(cudaError_t error) throw(cudaRunTimeError);

    void handle_cublas_error(cublasStatus_t status) throw(cuBlasError);

    void handle_cusolver_error(cusolverStatus_t status) throw(cuSolverError);

    void handle_num_threads_error(unsigned int numThreads) throw(numThreadsError) {
        if (numThreads != 1 && numThreads != 2 && numThreads != 4 && numThreads != 8 && numThreads != 16 &&
            numThreads != 32 && numThreads != 64 && numThreads != 128 && numThreads != 256 && numThreads != 512 &&
            numThreads != 1024){
            throw numThreadsError();
        }


    }


    ////
    ////Implementations
    ////


    dimMisMatch::dimMisMatch(const std::string _errorMessage) : errorMessage(_errorMessage) {}

    cudaRunTimeError::cudaRunTimeError(cudaError_t _error) : error(_error) {
        errorMessage = cudaGetErrorString(_error);
    }


    const char *cudaRunTimeError::what() const noexcept {
        return errorMessage.c_str();
    }


    cuBlasError::cuBlasError(cublasStatus_t _status) : status(_status) {
        switch (_status) {
            case CUBLAS_STATUS_NOT_INITIALIZED:
                errorMessage = "cuBlas is not properly initialized\n";
                break;
            case CUBLAS_STATUS_ALLOC_FAILED:
                errorMessage = "Allocation fails\n";
                break;
            case CUBLAS_STATUS_INTERNAL_ERROR:
                errorMessage = "Internal Error Occurs\n";
                break;
            case CUBLAS_STATUS_ARCH_MISMATCH:
                errorMessage = "Arch mismatch\n";
                break;
            case CUBLAS_STATUS_EXECUTION_FAILED:
                errorMessage = "Execution Failed\n";
                break;
            case CUBLAS_STATUS_MAPPING_ERROR:
                errorMessage = "Texture Memory binding failure\n";
                break;
            case CUBLAS_STATUS_NOT_SUPPORTED:
                errorMessage = "The functionality is not currently supported\n";
                break;
            default:
                break;
        }
    }

    const char *cuBlasError::what() const noexcept {
        return errorMessage.c_str();
    }


    cuSolverError::cuSolverError(cusolverStatus_t _status) : status(_status) {
        switch (_status) {
            case CUSOLVER_STATUS_NOT_INITIALIZED:
                errorMessage = "Library cuSolver not initialized correctly\n";
                break;
            case CUSOLVER_STATUS_ALLOC_FAILED:
                errorMessage = "Allocation Failed\n";
                break;
            case CUSOLVER_STATUS_INVALID_VALUE:
                errorMessage = "Invalid parameters passed\n";
                break;
            case CUSOLVER_STATUS_ARCH_MISMATCH:
                errorMessage = "Arch mistach \n";
                break;
            case CUSOLVER_STATUS_EXECUTION_FAILED:
                errorMessage = "Kernel Launch Failed\n";
                break;
            case CUSOLVER_STATUS_INTERNAL_ERROR:
                errorMessage = "Internal operation failed\n";
                break;
            case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
                errorMessage = "Matrix Type not Supported\n";
                break;
            default:
                break;
        }

    }



    const char *cuSolverError::what() const noexcept {
        return errorMessage.c_str();
    }


    void handle_cuda_runtime_error(cudaError_t error) throw(cudaRunTimeError) {
        if (error != cudaSuccess) {
            throw cudaRunTimeError(error);
        }
    }

    void handle_cublas_error(cublasStatus_t status) throw(cuBlasError) {
        if (status != CUBLAS_STATUS_SUCCESS) {
            throw cuBlasError(status);
        }
    }

    void handle_cusolver_error(cusolverStatus_t status) throw(cuSolverError) {
        if (status != CUSOLVER_STATUS_SUCCESS) {
            throw cuSolverError(status);
        }

    }


    numThreadsError::numThreadsError() {
        errorMessage = "Number of threads illegal, the number should be a multitude of 2";
    }

    const char *numThreadsError::what() const noexcept {
        return errorMessage.c_str();
    }


}
