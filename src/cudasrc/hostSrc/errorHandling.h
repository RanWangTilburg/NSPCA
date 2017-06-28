//
// Created by user on 24-11-16.
//

#ifndef NSPCA_ERRORHANDLING_H
#define NSPCA_ERRORHANDLING_H

#include <exception>
#include <string>

using std::string;

#include <cuda_runtime.h>
#include <cusolver_common.h>
#include <cublas_v2.h>

class linedError : public std::exception {
public:
    string errorName;
    string fileName;
    int lineNumber;

    linedError(string _error, string _fileName, int _lineNumber);

    void traceback();

    virtual void printDetails()=0;

    void printNameAndLineNumber();

    void handleError(bool abort = true);
};

class cudaRuntimeError : public linedError {
public:
    cudaError_t error;

    cudaRuntimeError(cudaError_t _error, string _fileName, int _lineNumber);

    ~cudaRuntimeError();

    virtual void printDetails();
};

class cuSolverError : public linedError {
public:
    cusolverStatus_t status;

    cuSolverError(cusolverStatus_t _status, string _fileName, int _lineNumer);

    virtual void printDetails();
};

class dimErrorMatMul : public linedError {
public:
    size_t lhsRows;
    size_t lhsCols;

    dimErrorMatMul(string _fileName, int _lineNumber, size_t lhsRows, size_t lhsCols, size_t rhsRows,
                   size_t rhsCols);

    virtual void printDetails();

    size_t rhsRows;
    size_t rhsCols;

};

class cuBlasError : public linedError {
public:
    cublasStatus_t cublasStatus;

    cuBlasError(cublasStatus_t cublasStatus, string _fileName, int _lineNumber);

    virtual void printDetails();

};

void throwCUDAError(cudaError_t error, string file, int line) throw(cudaRuntimeError);

void throwCUSolverError(cusolverStatus_t status, string file, int line) throw(cuSolverError);

void throwCUBlasError(cublasStatus_t status, string file, int line) throw(cuBlasError);

#define HANDLE_CUDA_ERROR(error) throwCUDAError(error, __FILE__, __LINE__);

#define HANDLE_CUBLAS_ERROR(status)  throwCUBlasError(status, __FILE__, __LINE__);

#define HANDLE_CUSOLVER_ERROR(status) throwCUSolverError(status, __FILE__, __LINE__);


#endif //NSPCA_ERRORHANDLING_H
