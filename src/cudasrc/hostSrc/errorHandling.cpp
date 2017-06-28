//
// Created by user on 24-11-16.
//

#include <iostream>
#include "errorHandling.h"
#include <cstdlib>

cudaRuntimeError::cudaRuntimeError(cudaError_t _error, string _fileName, int _lineNumber) : linedError(
        "CUDA Runtime Error", _fileName, _lineNumber), error(_error) {}

cudaRuntimeError::~cudaRuntimeError() {}

void cudaRuntimeError::printDetails() {
    std::cerr << "The error type is " << cudaGetErrorString(error) << std::endl;
}

void linedError::printNameAndLineNumber() {
    std::cerr << "An Error of Type " << errorName << "Occured at file: " << fileName << "line: " << lineNumber
              << std::endl;
}

void linedError::handleError(bool abort) {
    if (abort) {
        std::abort();
    }

}

linedError::linedError(string, string _fileName, int _lineNumber) : errorName("CUDA Runtime Error"),
                                                                    fileName(_fileName), lineNumber(_lineNumber) {}

void linedError::traceback() {
    printNameAndLineNumber();
    printDetails();
    handleError();
}

cuSolverError::cuSolverError(cusolverStatus_t _status, string _fileName, int _lineNumer) : linedError("cuSolver Error",
                                                                                                      _fileName,
                                                                                                      _lineNumer),
                                                                                           status(_status) {}

void cuSolverError::printDetails() {
    switch (status) {
        case CUSOLVER_STATUS_NOT_INITIALIZED:
            std::cerr << "Library cuSolver not initialized correctly\n";
            break;
        case CUSOLVER_STATUS_ALLOC_FAILED:
            std::cerr << "Allocation Failed\n";
            break;
        case CUSOLVER_STATUS_INVALID_VALUE:
            std::cerr << "Invalid parameters passed\n";
            break;
        case CUSOLVER_STATUS_ARCH_MISMATCH:
            std::cerr << "Arch mistach \n";
            break;
        case CUSOLVER_STATUS_EXECUTION_FAILED:
            std::cerr << "Kernel Launch Failed\n";
            break;
        case CUSOLVER_STATUS_INTERNAL_ERROR:
            std::cerr << "Internal operation failed\n";
            break;
        case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            std::cerr << "Matrix Type not Supported\n";
            break;
    }

}


void throwCUDAError(cudaError_t error, string file, int line) throw(cudaRuntimeError) {
    if (error != cudaSuccess) {
        throw cudaRuntimeError(error, file, line);
    }

}

void throwCUSolverError(cusolverStatus_t status, string file, int line) throw(cuSolverError){
    if (status != CUSOLVER_STATUS_SUCCESS) {
        throw cuSolverError(status, file, line);
    }
}

void throwCUBlasError(cublasStatus_t status, string file, int line) throw(cuBlasError) {
    if (status != CUBLAS_STATUS_SUCCESS){
        throw(cuBlasError(status, file, line));
    }
}

dimErrorMatMul::dimErrorMatMul(string _fileName, int _lineNumber, size_t lhsRows, size_t lhsCols, size_t rhsRows,
                               size_t rhsCols) : linedError("Dimension MisMatch in Matrix Product", _fileName,
                                                            _lineNumber),
                                                 lhsRows(lhsRows), lhsCols(lhsCols),
                                                 rhsRows(rhsRows), rhsCols(rhsCols) {}

void dimErrorMatMul::printDetails() {
    std::cerr << "Left hand side matrix is of dimension " << lhsRows << " times " << lhsCols << "\n";
    std::cerr << "Right hand side matrix if of dimension " << rhsRows << " times " << rhsCols << "\n";
}

cuBlasError::cuBlasError(cublasStatus_t cublasStatus, string _fileName, int _lineNumber)
        : linedError("CUBlas Error ", _fileName, _lineNumber), cublasStatus(cublasStatus) {}

void cuBlasError::printDetails() {
    switch (cublasStatus) {
        case CUBLAS_STATUS_NOT_INITIALIZED:
            std::cerr << "cuBlas is not properly initialized\n";
            break;
        case CUBLAS_STATUS_ALLOC_FAILED:
            std::cerr << "Allocation fails\n";
            break;
        case CUBLAS_STATUS_INTERNAL_ERROR:
            std::cerr << "Internal Error Occurs\n";
            break;
        case CUBLAS_STATUS_ARCH_MISMATCH:
            std::cerr << "Arch mismatch\n";
            break;
        case CUBLAS_STATUS_EXECUTION_FAILED:
            std::cerr << "Execution Failed\n";
            break;
        case CUBLAS_STATUS_MAPPING_ERROR:
            std::cerr << "Texture Memory binding failure\n";
            break;
        case CUBLAS_STATUS_NOT_SUPPORTED:
            std::cerr << "The functionality is not currently supported\n";
            break;
    }

}
