//
// Created by user on 15-11-16.
//
#include <cuda_runtime.h>
#include <iostream>
#include "nspcaSrc.h"
using std::cout;
using std::endl;

#include "cuView.h"
#include "cuMatrix.h"
namespace cuExec {


    template<typename Scalar, unsigned int numThreads, typename Op>
    __global__ void map_kernel(Scalar *data, const size_t size, const Op* op) {
        unsigned int tid = threadIdx.x;
        unsigned int bid = blockIdx.x;
        unsigned int position = tid + bid * numThreads;

        for (; position < size; position += numThreads * blockDim.x) {
            data[position] = op->operator()(position);
        }
    }
    
    template<typename Scalar, typename Op>
    void map(Scalar *data, const size_t size, const Op *op, unsigned int numThreads, unsigned int numBlocks,
             cudaStream_t stream) {
        switch (numThreads) {
            case (1):
                map_kernel<Scalar, 1, Op> <<< numBlocks, 1, 0, stream >>> (data, size, op);
            case (2):
                map_kernel<Scalar, 2, Op> <<< numBlocks, 2, 0, stream >>> (data, size, op);
            case (4):
                map_kernel<Scalar, 4, Op> <<< numBlocks, 4, 0, stream >>> (data, size, op);
            case (8):
                map_kernel<Scalar, 8, Op> <<< numBlocks, 8, 0, stream >>> (data, size, op);
            case (16):
                map_kernel<Scalar, 16, Op> <<< numBlocks, 16, 0, stream >>> (data, size, op);
            case (32):
                map_kernel<Scalar, 32, Op> <<< numBlocks, 32, 0, stream >>> (data, size, op);
            case (64):
                map_kernel<Scalar, 64, Op> <<< numBlocks, 64, 0, stream >>> (data, size, op);
            case (128):
                map_kernel<Scalar, 128, Op> <<< numBlocks, 128, 0, stream >>> (data, size, op);
            case (256):
                map_kernel<Scalar, 256, Op> <<< numBlocks, 256, 0, stream >>> (data, size, op);
            case (512):
                map_kernel<Scalar, 512, Op> <<< numBlocks, 512, 0, stream >>> (data, size, op);
            case (1024):
                map_kernel<Scalar, 1024, Op> <<< numBlocks, 1024, 0, stream >>> (data, size, op);

        }
    };

    template<typename Scalar>
    CUDA_CALLABLE_MEMBER
    cuView<Scalar>::cuView(Scalar *data, size_t nRow, size_t nCol):data(data), nRow(nRow), nCol(nCol) {}

    template<typename Scalar>
    CUDA_CALLABLE_MEMBER size_t cuView<Scalar>::rows() const {
        return nRow;
    }

    template<typename Scalar>
    CUDA_CALLABLE_MEMBER size_t cuView<Scalar>::cols() const {
        return nCol;
    }

    template<typename Scalar>
    CUDA_CALLABLE_MEMBER const Scalar *cuView<Scalar>::getPtr() const {
        return data;
    }

    template<typename T>
    CUDA_HOST ostream &print_vec(ostream &os, const cuView<T> &dst) {
        T *host_ptr = (T *) malloc(sizeof(T) * dst.rows() * dst.cols());
        cudaMemcpy(host_ptr, dst.data, sizeof(T) * dst.rows() * dst.cols(), cudaMemcpyDeviceToHost);
        os << host_ptr[0];
        for (int i = 0; i < dst.cols(); i++) {
            os << "\t" << host_ptr[i];
        }
        free(host_ptr);
        return os;

    }

    template<typename T>
    CUDA_HOST ostream &print_mat(ostream &os, const cuView<T> &dst) {
        T *host_ptr = (T *) malloc(sizeof(T) * dst.rows() * dst.cols());
        cudaMemcpy(host_ptr, dst.data, sizeof(T) * dst.rows() * dst.cols(), cudaMemcpyDeviceToHost);
        for (int i = 0; i < dst.rows(); i++) {
            os << host_ptr[i];
            for (int j = 1; j < dst.cols(); j++) {
                os << "\t" << host_ptr[i + dst.rows() * j];
            }
            os << endl;
        }
        free(host_ptr);
        return os;
    }

    template<typename Scalar>
    CUDA_HOST void cuView<Scalar>::print() {
        cout << *this << endl;
    }

//    template<typename Scalar>
//
//    Matrix<Scalar, Dynamic, Dynamic> cuView<Scalar>::getEigenMatrix() const {
//        using namespace Eigen;
//        Matrix<Scalar, Dynamic, Dynamic> tmp = Matrix<Scalar, Dynamic, Dynamic>::Constant(this->rows(), this->cols(), 0);
//        Scalar * tmpPtr = (Scalar *)tmp.data();
//        cudaMemcpy(tmpPtr, data, sizeof(Scalar) * nRow*nCol, cudaMemcpyDeviceToHost);
//        return tmp;
//    }
    template<typename Scalar>
    CUDA_CALLABLE_MEMBER Scalar *cuView<Scalar>::getPtr() {
        return data;
    }

    template<typename T>
    CUDA_HOST std::ostream &operator<<(std::ostream &os, const cuView<T> &dt) {
        if (dt.cols() == 1) {
            return print_vec(os, dt);
        } else {
            return print_mat(os, dt);
        }
    }

    template<typename Scalar>
    __device__ Scalar &cuView<Scalar>::operator()(const size_t i, const size_t j) {
        return data[j + i * nRow];
    }

    template<typename Scalar>
    CUDA_DEVICE Scalar cuView<Scalar>::operator()(const size_t i, const size_t j) const {
        return data[j + i * nRow];
    }

    template<typename Scalar>
    CUDA_DEVICE Scalar& cuView<Scalar>::operator()(const unsigned int i) {
        return data[i];
    }


    template<typename Scalar>
    size_t cuMatrix<Scalar>::rows() const {
        return nRow;
    }

    template<typename Scalar>
    size_t cuMatrix<Scalar>::cols() const {
        return nCol;
    }

    template<typename Scalar>
    cuMatrix<Scalar>::cuMatrix(const size_t _rows, const size_t _cols): nRow(_rows), nCol(_cols) {
        cudaMalloc((void **) &devPtr, sizeof(Scalar) * _rows * _cols);
    }

    template<typename Scalar>
    cuMatrix<Scalar>::cuMatrix(const size_t _rows, const size_t _cols, const Scalar value):nRow(_rows), nCol(_cols) {
        const int size = _rows * _cols;
        cudaMalloc((void **) &devPtr, sizeof(Scalar) * size);
//        cout << value << endl;
        int numBlocks = std::max((int) size / 512, 1);
        cuAlgorithm<Scalar, 256>::setConstant(numBlocks, devPtr, size, value);
    }

    template<typename T>
    ostream &output_vec(ostream &os, const cuMatrix<T> &ds) {
        T *hostPtr;
        cudaMallocHost((void **) &hostPtr, sizeof(T) * ds.rows() * ds.cols());
        cudaMemcpy(hostPtr, ds.getPtr(), sizeof(T) * ds.rows() * ds.cols(), cudaMemcpyDeviceToHost);
        os << hostPtr[0];
        for (int i = 1; i < ds.rows(); i++) {
            os << "\t" << hostPtr[i];
        }
        cudaFreeHost(hostPtr);
        return os;
    }

    template<typename T>
    ostream &output_mat(ostream &os, const cuMatrix<T> &ds) {
        T *hostPtr;
        cudaMallocHost((void **) &hostPtr, sizeof(T) * ds.rows() * ds.cols());
        cudaMemcpy(hostPtr, ds.getPtr(), sizeof(T) * ds.rows() * ds.cols(), cudaMemcpyDeviceToHost);
        int rows = ds.rows();
        for (int i = 0; i < rows; i++) {
            os << hostPtr[i];
            for (int j = 1; j < ds.cols(); j++) {
                os << "\t" << hostPtr[i + j * rows];
            }
            os << endl;
        }
        cudaFreeHost(hostPtr);
        return os;
    }

    template<typename T>
    ostream &operator<<(ostream &os, const cuMatrix<T> &ds) {
        if (ds.cols() == 1) {
            return output_vec(os, ds);
        } else {
            return output_mat(os, ds);
        }
    }
//
//    void init_cu_matrix() {
//        cuMatrix<double> a(10,10);
//        cuMatrix<int> b(10,10);
//        cuMatrix<float> c(10,10);
//        cout << a << b<< c << endl;
//
//    }

    template<typename Scalar>
    void cuMatrix<Scalar>::print() {
        cout << *this << endl;
    }

    template<typename Scalar>
    const Scalar *cuMatrix<Scalar>::getPtr() const {
        return devPtr;
    }

    template<typename Scalar>
    Scalar *cuMatrix<Scalar>::getPtr() {
        return devPtr;
    }

    template<typename Scalar>
    thrust::device_ptr<Scalar> cuMatrix<Scalar>::getThrustPtr() {
        Scalar *raw_ptr = getPtr();
        return thrust::device_ptr<Scalar>(raw_ptr);
    }

    template<typename Scalar>
    void cuMatrix<Scalar>::randn(const Scalar mean, const Scalar std) {
        Scalar *data = getPtr();
        const size_t size = nRow * nCol;
        int numBlocks = std::max((int) size / 512, 1);
        cuAlgorithm<Scalar, 256>::rnorm(numBlocks, data, size, mean, std);

    }

    template<typename Scalar>
    cuMatrix<Scalar>::~cuMatrix() {
        cudaFree(devPtr);
    }

    template<typename Scalar>
    CUDA_DEVICE Scalar& cuMatrix<Scalar>::operator()(const size_t i, const size_t j) {
        return devPtr[i + j * nRow];
    }

    template<typename Scalar>
    CUDA_DEVICE Scalar& cuMatrix<Scalar>::operator()(const size_t i) {
        return devPtr[i];
    }

    template<typename Scalar>
    CUDA_DEVICE Scalar cuMatrix<Scalar>::operator()(const size_t i, const size_t j) const {
        return devPtr[i + j * nRow];
    }

    template<typename Scalar>
    CUDA_DEVICE Scalar cuMatrix<Scalar>::operator()(const size_t i) const {
        return devPtr[i];
    }
    template<typename Scalar>
    template<typename Op>
    cuMatrix<Scalar> &cuMatrix<Scalar>::operator=(const Assignable<Op> *assign){
        map(getPtr(), rows()*cols(), assign->operation, assign->numThreads, assign->numBlocks, assign->stream);
        cudaStreamSynchronize(assign->stream);
        delete assign;
        return *this;
    }
    template<typename Scalar>
    cuView<Scalar> *cuMatrix<Scalar>::v() {
        return new cuView<Scalar>(devPtr, nRow, nCol);
    }


    void test(){
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        cuMatrix<double> a(10, 10, 1);
        a.print();
        cuMatrix<double> b(10, 10, 1);
        cuMatrix<double> c(10,10,0.0);
        cudaDeviceSynchronize();
        auto lhs = *a.v();
        auto rhs = *a.v();
        c = (lhs+rhs)->run(1,1,stream);
//        c = (*a.v()+*b.v())->run(1, 1, stream);
        cudaStreamSynchronize(stream);
        c.print();
        cudaStreamDestroy(stream);
    }




}


namespace NSPCA{

    __device__ __forceinline__ void solve_p_positive(double * ATZ, double * Pptr, const size_t N, const size_t P, const size_t p,
                                                     const int i, const int j, const double lambda, const double scale_square){
        cuView<double> ZtAview(ATZ, p,P);
        cuView<double> PView(Pptr, p,P);


        double t = (2*ZtAview(i,j)-lambda);
        if (t > 0){
            PView(i,j) = t/2*scale_square;

        }

    }

    __device__ __forceinline__ void solve_p_negative(double * ATZ, double * Pptr, const size_t N, const size_t P, const size_t p,
                                                     const int i, const int j, const double lambda, const double scale_square) {

        cuView<double> ZtAview(ATZ, p,P);
        cuView<double> PView(Pptr, p,P);


        double t = (2*ZtAview(i,j)+lambda);
        if (t < 0){
            PView(i,j) = t/(2*scale_square);

        }
    }

    __device__ __forceinline__ void solve_p_general(double * ATZ, double * Pptr, const size_t N, const size_t P, const size_t p,
                                                    const int i, const int j, const double lambda, const double scale_square){
        cuView<double> ZtAview(ATZ, p,P);
        cuView<double> PView(Pptr, p,P);

        double t = (2*ZtAview(i,j)-lambda);
        if (t > 0){
            PView(i,j) = t/(2*scale_square);
        }
        else {
            t = (2*ZtAview(i,j)+lambda);{
                if (t<0){
                    PView(i,j)=t/(2*scale_square);
                }
            }
        }
    }

    template<unsigned int numThreads>
    __global__ void solve_p_in_nspca(double * devp, const size_t N, const size_t P, const size_t p,  double * ATZ,
                                     int * restriction, const double lambda, const double scale_square ){
        const int tid = threadIdx.x;
        const int offset = numThreads*blockIdx.x;
        cuView<int> resView(restriction, p, P);
        for (int index = tid+offset;index <p*P;index+=numThreads*blockDim.x){
            int j = index/p;
            int i = index - j*p;

            if (resView(i,j)==2){
                solve_p_general(ATZ, devp, N, P, p, i, j, lambda, scale_square);

            }
            else if (resView(i,j)==1){
                solve_p_positive(ATZ, devp, N, P, p, i, j, lambda, scale_square);
            }
            else if (resView(i,j)==-1){
                solve_p_negative(ATZ, devp, N, P, p, i, j, lambda, scale_square);
            }
        }
    };

    void solve_p_nspca(double * devp, const size_t N, const size_t P, const size_t p,  double * ATZ,
                       int * restriction, const double lambda, const double scale_square, const unsigned int numThreads, const unsigned int numBlocks ){
        switch (numThreads){
            case (1):
                solve_p_in_nspca<1> <<< numBlocks, 1 >>> (devp, N, P, p, ATZ, restriction, lambda, scale_square);
            case (2):
                solve_p_in_nspca<2> <<< numBlocks, 2 >>> (devp, N, P, p, ATZ, restriction, lambda, scale_square);
            case (4):
                solve_p_in_nspca<4> <<< numBlocks, 4 >>> (devp, N, P, p, ATZ, restriction, lambda, scale_square);
            case (8):
                solve_p_in_nspca<8> <<< numBlocks, 8 >>> (devp, N, P, p, ATZ, restriction, lambda, scale_square);
            case (16):
                solve_p_in_nspca<16> <<< numBlocks, 16 >>> (devp, N, P, p, ATZ, restriction, lambda, scale_square);
            case (32):
                solve_p_in_nspca<32> <<< numBlocks, 32 >>> (devp, N, P, p, ATZ, restriction, lambda, scale_square);
            case (64):
                solve_p_in_nspca<64> <<< numBlocks, 64 >>> (devp, N, P, p, ATZ, restriction, lambda, scale_square);
            case (128):
                solve_p_in_nspca<128> <<< numBlocks, 128 >>> (devp, N, P, p, ATZ, restriction, lambda, scale_square);
            case (256):
                solve_p_in_nspca<256> <<< numBlocks, 256 >>> (devp, N, P, p, ATZ, restriction, lambda, scale_square);
            case (512):
                solve_p_in_nspca<512> <<< numBlocks, 512 >>> (devp, N, P, p, ATZ, restriction, lambda, scale_square);
            case (1024):
                solve_p_in_nspca<1024> <<< numBlocks, 1024 >>>
                                                      (devp, N, P, p, ATZ, restriction, lambda, scale_square);

        }

    }
    template<typename ScalarType, unsigned int numThreads>
    CUDA_KERNEL void frobenius_kernel(ScalarType * result_dev_ptr, const ScalarType * input1_devptr, const size_t size){
        extern __shared__ ScalarType sPartials[];
        const  int tid = threadIdx.x;
        double sum = 0.0;
        for (size_t i = tid+numThreads*blockIdx.x; i < size; i += numThreads * blockDim.x) {
            ScalarType temp = input1_devptr[i];
//            printf("This is from thread %d with value %0.12lf \n", i, temp);
            sum += temp*temp;
        }
        sPartials[tid] = sum;
//        printf("This is thread %d, block %d, and value %0.12lf after calculating the sum \n", tid,  blockIdx.x, sPartials[tid]);
        __syncthreads();

        if (numThreads >= 1024) {
            if (tid < 512) {
                sPartials[tid] += sPartials[tid + 512];
            }
            __syncthreads();
        }
        if (numThreads >= 512) {
            if (tid < 256) {
                sPartials[tid] += sPartials[tid + 256];
            }
            __syncthreads();
        }
        if (numThreads >= 256) {
            if (tid < 128) {
                sPartials[tid] += sPartials[tid + 128];
            }
            __syncthreads();
        }
        if (numThreads >= 128) {
            if (tid < 64) {
                sPartials[tid] += sPartials[tid + 64];
            }
            __syncthreads();
        }
        // warp synchronous at the end
        if (tid < 32) {
            volatile double *wsSum = sPartials;

            if (numThreads >= 64) { wsSum[tid] += wsSum[tid + 32]; }
            if (numThreads >= 32) { wsSum[tid] += wsSum[tid + 16]; }
            if (numThreads >= 16) { wsSum[tid] += wsSum[tid + 8]; }
            if (numThreads >= 8) { wsSum[tid] += wsSum[tid + 4]; }
            if (numThreads >= 4) { wsSum[tid] += wsSum[tid + 2]; }
            if (numThreads >= 2) { wsSum[tid] += wsSum[tid + 1]; }
            if (tid == 0) {

                result_dev_ptr[blockIdx.x] = wsSum[0];
                printf("This is after reduction for block %d with value  %0.12lf \n ", blockIdx.x, wsSum[0]);
            }
        }

    }


    template<typename ScalarType, unsigned int numThreads>
    CUDA_KERNEL void l2_diff_kernel(ScalarType * result_dev_ptr, const ScalarType * input1_devptr, const ScalarType * input2_devptr, const size_t size){
        extern __shared__ ScalarType sPartials[];
        const unsigned int tid = threadIdx.x;
        double sum = 0.0;
        for (size_t i = tid+numThreads*blockIdx.x; i < size; i += numThreads * blockDim.x) {
            ScalarType temp = input1_devptr[i]- input2_devptr[i];
            sum += temp*temp;
//        printf("from Thread %d the value is %0.12lf \n", tid, temp);
        }
        sPartials[tid] = sum;
        __syncthreads();
        if (numThreads >= 1024) {
            if (tid < 512) {
                sPartials[tid] += sPartials[tid + 512];
            }
            __syncthreads();
        }
        if (numThreads >= 512) {
            if (tid < 256) {
                sPartials[tid] += sPartials[tid + 256];
            }
            __syncthreads();
        }
        if (numThreads >= 256) {
            if (tid < 128) {
                sPartials[tid] += sPartials[tid + 128];
            }
            __syncthreads();
        }
        if (numThreads >= 128) {
            if (tid < 64) {
                sPartials[tid] += sPartials[tid + 64];
            }
        }
        __syncthreads();
        // warp synchronous at the end
        if (tid < 32) {
            volatile ScalarType *wsSum = sPartials;
            if (numThreads >= 64) { wsSum[tid] += wsSum[tid + 32]; }
            if (numThreads >= 32) { wsSum[tid] += wsSum[tid + 16]; }
            if (numThreads >= 16) { wsSum[tid] += wsSum[tid + 8]; }
            if (numThreads >= 8) { wsSum[tid] += wsSum[tid + 4]; }
            if (numThreads >= 4) { wsSum[tid] += wsSum[tid + 2]; }
            if (numThreads >= 2) { wsSum[tid] += wsSum[tid + 1]; }
            if (tid == 0) {
                result_dev_ptr[blockIdx.x] = wsSum[0];
            }
        }
    }

    ////The numblocks is assumed to be equal to the number of columns (P)
    ////Size of shared memory is equal to 3 times P
    template<unsigned int numThreads>
    __global__ void count_incidences(int * data, const size_t N, const size_t P, int * incidence_count){
        extern __shared__ int partials[];
        unsigned int tid = threadIdx.x;
        unsigned int bid = blockIdx.x;
        cuView<int> data_view(data, N, P);
        cuView<int> incidence_count_view(incidence_count, 3, P);


        int num_pos     =    0;
        int num_neutral =    0;
        int num_neg     =    0;
        int index       =  tid;


        for (;index<N;index+=numThreads){
            if (data_view(index, bid)==-1){
                num_neg+=1;
            }
            else if (data_view(index, bid)==0){
                num_neutral+=1;
            }
            else {
                num_pos+=1;
            }
        }
        partials[3*tid] =num_neg;
        partials[3*tid+1] = num_neutral;
        partials[3*tid+1] = num_pos;

        __syncthreads();

        if (numThreads >= 1024) {
            if (tid < 512) {
                partials[3*tid] += partials[3*tid + 3*512];
                partials[3*tid+1] += partials[3*tid+ 3*512+1];
                partials[3*tid+2] += partials[3*tid+3*512+2];
            }
            __syncthreads();
        }
        if (numThreads >= 512) {
            if (tid < 256) {
                partials[3*tid] += partials[3*tid + 3*256];
                partials[3*tid+1] += partials[3*tid+ 3*256+1];
                partials[3*tid+2] += partials[3*tid+3*256+2];
            }
            __syncthreads();
        }
        if (numThreads >= 256) {
            if (tid < 128) {
                partials[3*tid] += partials[3*tid + 3*128];
                partials[3*tid+1] += partials[3*tid+ 3*128+1];
                partials[3*tid+2] += partials[3*tid+3*128+2];
            }
            __syncthreads();
        }
        if (numThreads >= 128) {
            if (tid < 64) {
                partials[3*tid] += partials[3*tid + 3*64];
                partials[3*tid+1] += partials[3*tid+ 3*64+1];
                partials[3*tid+2] += partials[3*tid+3*64+2];
            }
        }
        __syncthreads();
        // warp synchronous at the end
        if (tid < 32) {
            volatile int *wsSum = partials;
            if (numThreads >= 64) {
                wsSum[3 * tid] += wsSum[3 * tid + 3 * 32];
                wsSum[3 * tid + 1] += wsSum[3 * tid + 3 * 32 + 1];
                wsSum[3 * tid + 2] = wsSum[3 * tid + 3 * 32 + 2];
            }
            if (numThreads >= 32) {
                wsSum[3 * tid] += wsSum[3 * tid + 3 * 16];
                wsSum[3 * tid + 1] += wsSum[3 * tid + 3 * 16 + 1];
                wsSum[3 * tid + 2] = wsSum[3 * tid + 3 * 16 + 2];
            }
            if (numThreads >= 16) {
                wsSum[3 * tid] += wsSum[3 * tid + 3 * 8];
                wsSum[3 * tid + 1] += wsSum[3 * tid + 3 * 8 + 1];
                wsSum[3 * tid + 2] = wsSum[3 * tid + 3 * 8 + 2];
            }
            if (numThreads >= 8) {
                wsSum[3 * tid] += wsSum[3 * tid + 3 * 4];
                wsSum[3 * tid + 1] += wsSum[3 * tid + 3 * 4 + 1];
                wsSum[3 * tid + 2] = wsSum[3 * tid + 3 * 4 + 2];
            }
            if (numThreads >= 4) {
                wsSum[3 * tid] += wsSum[3 * tid + 3 * 2];
                wsSum[3 * tid + 1] += wsSum[3 * tid + 3 * 2 + 1];
                wsSum[3 * tid + 2] = wsSum[3 * tid + 3 * 2 + 2];
            }
            if (numThreads >= 2) {
                wsSum[3 * tid] += wsSum[3 * tid + 3];
                wsSum[3 * tid + 1] += wsSum[3 * tid + 3 + 1];
                wsSum[3 * tid + 2] = wsSum[3 * tid + 3 + 2];
            }
            if (tid == 0) {
                incidence_count_view(0, bid) = wsSum[0];
                incidence_count_view(1, bid) = wsSum[1];
                incidence_count_view(2, bid) = wsSum[2];
            }
        }
    }

    ////The numblocks is assumed to be equal to the number of columns (P)
    ////Size of shared memory is equal to 3 times P

    template<unsigned int numThreads>
    __global__ void cumulate_kernel(int * data, const size_t N, const size_t P, double * AP double *n_count, double * e_count, double * p_count ){
        extern __shared__ double partials[];
        unsigned int tid = threadIdx.x;
        unsigned int bid = blockIdx.x;
        cuView<int> data_view(data, N, P);
        cuView<double> AP_view(AP, N, P);


        double num_pos     =    0;
        double num_neutral =    0;
        double num_neg     =    0;

        int index       =  tid;

        for (;index<N;index+=numThreads){
            if (data_view(index, bid)==-1){
                num_neg+=AP_view(index, bid);
            }
            else if (data_view(tid, bid)==0){
                num_neutral+=AP_view(index, bid);
            }
            else {
                num_pos+=AP_view(index, bid);
            }
        }
        partials[3*tid] =num_neg;
        partials[3*tid+1] = num_neutral;
        partials[3*tid+1] = num_pos;

        __syncthreads();

        if (numThreads >= 1024) {
            if (tid < 512) {
                partials[3*tid] += partials[3*tid + 3*512];
                partials[3*tid+1] += partials[3*tid+ 3*512+1];
                partials[3*tid+2] += partials[3*tid+3*512+2];
            }
            __syncthreads();
        }
        if (numThreads >= 512) {
            if (tid < 256) {
                partials[3*tid] += partials[3*tid + 3*256];
                partials[3*tid+1] += partials[3*tid+ 3*256+1];
                partials[3*tid+2] += partials[3*tid+3*256+2];
            }
            __syncthreads();
        }
        if (numThreads >= 256) {
            if (tid < 128) {
                partials[3*tid] += partials[3*tid + 3*128];
                partials[3*tid+1] += partials[3*tid+ 3*128+1];
                partials[3*tid+2] += partials[3*tid+3*128+2];
            }
            __syncthreads();
        }
        if (numThreads >= 128) {
            if (tid < 64) {
                partials[3*tid] += partials[3*tid + 3*64];
                partials[3*tid+1] += partials[3*tid+ 3*64+1];
                partials[3*tid+2] += partials[3*tid+3*64+2];
            }
        }
        __syncthreads();
        // warp synchronous at the end
        if (tid < 32) {
            volatile int *wsSum = partials;
            if (numThreads >= 64) {
                wsSum[3 * tid] += wsSum[3 * tid + 3 * 32];
                wsSum[3 * tid + 1] += wsSum[3 * tid + 3 * 32 + 1];
                wsSum[3 * tid + 2] = wsSum[3 * tid + 3 * 32 + 2];
            }
            if (numThreads >= 32) {
                wsSum[3 * tid] += wsSum[3 * tid + 3 * 16];
                wsSum[3 * tid + 1] += wsSum[3 * tid + 3 * 16 + 1];
                wsSum[3 * tid + 2] = wsSum[3 * tid + 3 * 16 + 2];
            }
            if (numThreads >= 16) {
                wsSum[3 * tid] += wsSum[3 * tid + 3 * 8];
                wsSum[3 * tid + 1] += wsSum[3 * tid + 3 * 8 + 1];
                wsSum[3 * tid + 2] = wsSum[3 * tid + 3 * 8 + 2];
            }
            if (numThreads >= 8) {
                wsSum[3 * tid] += wsSum[3 * tid + 3 * 4];
                wsSum[3 * tid + 1] += wsSum[3 * tid + 3 * 4 + 1];
                wsSum[3 * tid + 2] = wsSum[3 * tid + 3 * 4 + 2];
            }
            if (numThreads >= 4) {
                wsSum[3 * tid] += wsSum[3 * tid + 3 * 2];
                wsSum[3 * tid + 1] += wsSum[3 * tid + 3 * 2 + 1];
                wsSum[3 * tid + 2] = wsSum[3 * tid + 3 * 2 + 2];
            }
            if (numThreads >= 2) {
                wsSum[3 * tid] += wsSum[3 * tid + 3];
                wsSum[3 * tid + 1] += wsSum[3 * tid + 3 + 1];
                wsSum[3 * tid + 2] = wsSum[3 * tid + 3 + 2];
            }
            if (tid == 0) {
                n_count[bid] = wsSum[0];
                e_count[bid] = wsSum[1];
                p_count[bid] = wsSum[2];
            }
        }
    }
}

