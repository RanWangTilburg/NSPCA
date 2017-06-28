#include "macro.h"
#include <cstdlib>
#include <cstdio>
#include "view.h"
#include <cuda_runtime.h>

namespace NSPCA {


    __device__ __forceinline__

    void
    solve_p_positive_v2(double *ATZ, double *Pptr, const size_t N, const size_t P, const size_t p,
                        const int i, const int j, double *n_lambda, double *omegaVec, const double n_scale_square) {
        cuView<double> AtZview(ATZ, p, P);
        cuView<double> PView(Pptr, p, P);
        double tmp = 2.0 * AtZview(i, j);
        double t = -n_lambda[j] + tmp;

        if (t > 0) {

            PView(i, j) = t / (2.0 * n_scale_square*omegaVec[j]);
        }
        else PView(i,j) = 0;


    }

    __device__ __forceinline__

    void
    solve_p_negative_v2(double *ATZ, double *Pptr, const size_t N, const size_t P, const size_t p,
                        const int i, const int j, double *n_lambda, double *omegaVec, const double n_scale_square) {

        cuView<double> AtZview(ATZ, p, P);
        cuView<double> PView(Pptr, p, P);
        double tmp = 2.0 * AtZview(i, j);
        double t = n_lambda[j] + tmp;

        if (t < 0) {
            PView(i, j) = t / (2 * n_scale_square *omegaVec[j]);
        }

    }

    __device__ __forceinline__

    void
    solve_p_general_v2(double *ATZ, double *Pptr, const size_t N, const size_t P, const size_t p,
                       const int i, const int j, double *n_lambda, double *omegaVec, const double n_scale_square) {
        cuView<double> AtZview(ATZ, p, P);
        cuView<double> PView(Pptr, p, P);

        double tmp = 2.0 * AtZview(i, j);
        double t = -n_lambda[j] + tmp;
        if (t > 0) {
            PView(i, j) = t / (2.0 * n_scale_square*omegaVec[j]);
        } else {
            t = n_lambda[j] + tmp;
            if (t < 0) {
                PView(i, j) = t / (2.0 * n_scale_square*omegaVec[j]);
            } else {
                PView(i, j) = 0;
            }
        }
    }

    template<unsigned int numThreads>
    __global__ void solve_p_in_nspca_v2(double *devp, const size_t N, const size_t P, const size_t p, double *ATZ,
                                        int *restriction, double *n_lambda, double *omegaVec,
                                        const double n_scale_square) {
        const int tid = threadIdx.x;
        const int offset = numThreads * blockIdx.x;
        cuView<int> resView(restriction, p, P);
        cuView<double> AtZview = cuView<double>(ATZ, p, P);
        cuView<double> PView = cuView<double>(devp, p, P);
        for (int index = tid + offset; index < p * P; index += numThreads * blockDim.x) {
            int j = index / p;
            int i = index - j * p;

            PView(i, j) = 0;
            if (resView(i, j) == 2) {

                solve_p_general_v2(ATZ, devp, N, P, p, i, j, n_lambda, omegaVec,  n_scale_square);

            } else if (resView(i, j) == 1) {
                solve_p_positive_v2(ATZ, devp, N, P, p, i, j, n_lambda, omegaVec, n_scale_square);
            } else if (resView(i, j) == -1) {
                solve_p_negative_v2(ATZ, devp, N, P, p, i, j, n_lambda, omegaVec, n_scale_square);
            }
        }
    };


    __global__

    void update_weights_kernel(double *lambda_weights, double new_lambda, double P) {
        const int tid = threadIdx.x;
        const int bid = blockIdx.x;
        const int stride = tid + bid * blockDim.x;

        if (stride < P) {
            lambda_weights[stride] *= new_lambda;
        }
    }

    void update_weights(double *lambda_weights, double new_lambda, double P, const unsigned int numThreads,
                        const unsigned int numBlocks, cudaStream_t stream) {
        update_weights_kernel << < numBlocks, numThreads, 0, stream >> > (lambda_weights, new_lambda, P);
    }


    void solve_p_nspca_v2(double *devp, const size_t N, const size_t P, const size_t p, double *ATZ,
                          int *restriction, double *lambda, double *omegaVec, const double scale_square,
                          const unsigned int numThreads,
                          const unsigned int numBlocks, cudaStream_t stream) {
        switch (numThreads) {
            case (1):
                solve_p_in_nspca_v2<1> << < numBlocks, 1, 0, stream >> >
                                                             (devp, N, P, p, ATZ, restriction, lambda, omegaVec, scale_square);
                break;
            case (2):
                solve_p_in_nspca_v2<2> << < numBlocks, 2, 0, stream >> >
                                                             (devp, N, P, p, ATZ, restriction, lambda, omegaVec, scale_square);
                break;
            case (4):
                solve_p_in_nspca_v2<4> << < numBlocks, 4, 0, stream >> >
                                                             (devp, N, P, p, ATZ, restriction, lambda, omegaVec, scale_square);
                break;
            case (8):
                solve_p_in_nspca_v2<8> << < numBlocks, 8, 0, stream >> >
                                                             (devp, N, P, p, ATZ, restriction, lambda, omegaVec, scale_square);
                break;
            case (16):
                solve_p_in_nspca_v2<16> << < numBlocks, 16, 0, stream >> >
                                                               (devp, N, P, p, ATZ, restriction, lambda, omegaVec, scale_square);
                break;
            case (32):
                solve_p_in_nspca_v2<32> << < numBlocks, 32, 0, stream >> >
                                                               (devp, N, P, p, ATZ, restriction, lambda, omegaVec, scale_square);
                break;
            case (64):
                solve_p_in_nspca_v2<64> << < numBlocks, 64, 0, stream >> >
                                                               (devp, N, P, p, ATZ, restriction, lambda, omegaVec, scale_square);
                break;
            case (128):
                solve_p_in_nspca_v2<128> << < numBlocks, 128, 0, stream >> >
                                                                 (devp, N, P, p, ATZ, restriction, lambda, omegaVec, scale_square);
                break;
            case (256):
                solve_p_in_nspca_v2<256> << < numBlocks, 256, 0, stream >> >
                                                                 (devp, N, P, p, ATZ, restriction, lambda, omegaVec, scale_square);
                break;
            case (512):
                solve_p_in_nspca_v2<512> << < numBlocks, 512, 0, stream >> >
                                                                 (devp, N, P, p, ATZ, restriction, lambda, omegaVec, scale_square);
                break;
            case (1024):
                solve_p_in_nspca_v2<1024> << < numBlocks, 1024, 0, stream >> >
                                                                   (devp, N, P, p, ATZ, restriction, lambda, omegaVec, scale_square);
                break;

        }

    }


    ////The numblocks is assumed to be equal to the number of columns (P)
    ////Size of shared memory is equal to 3 times P
    template<unsigned int numThreads>
    __global__ void count_incidence_kernel(int *data, const size_t N, const size_t P, int *incidence_count) {
        extern __shared__
        int partials[];
        unsigned int tid = threadIdx.x;
        unsigned int bid = blockIdx.x;
        cuView<int> data_view(data, N, P);
        cuView<int> incidence_count_view(incidence_count, 3, P);


        int num_pos = 0;
        int num_neutral = 0;
        int num_neg = 0;
        int index = tid;


        for (; index < N; index += numThreads) {
            if (data_view(index, bid) == -1) {
                num_neg += 1;
            } else if (data_view(index, bid) == 0) {
                num_neutral += 1;
            } else {
                num_pos += 1;
            }
        }
        partials[3 * tid] = num_neg;
        partials[3 * tid + 1] = num_neutral;
        partials[3 * tid + 2] = num_pos;

        __syncthreads();

        if (numThreads >= 1024) {
            if (tid < 512) {
                partials[3 * tid] += partials[3 * tid + 3 * 512];
                partials[3 * tid + 1] += partials[3 * tid + 3 * 512 + 1];
                partials[3 * tid + 2] += partials[3 * tid + 3 * 512 + 2];
            }
            __syncthreads();
        }
        if (numThreads >= 512) {
            if (tid < 256) {
                partials[3 * tid] += partials[3 * tid + 3 * 256];
                partials[3 * tid + 1] += partials[3 * tid + 3 * 256 + 1];
                partials[3 * tid + 2] += partials[3 * tid + 3 * 256 + 2];
            }
            __syncthreads();
        }
        if (numThreads >= 256) {
            if (tid < 128) {
                partials[3 * tid] += partials[3 * tid + 3 * 128];
                partials[3 * tid + 1] += partials[3 * tid + 3 * 128 + 1];
                partials[3 * tid + 2] += partials[3 * tid + 3 * 128 + 2];
            }
            __syncthreads();
        }
        if (numThreads >= 128) {
            if (tid < 64) {
                partials[3 * tid] += partials[3 * tid + 3 * 64];
                partials[3 * tid + 1] += partials[3 * tid + 3 * 64 + 1];
                partials[3 * tid + 2] += partials[3 * tid + 3 * 64 + 2];
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


    void
    count_incidence_gpu(int *data, const size_t N, const size_t P, int *incidence_count,
                        const unsigned int numThreads,
                        cudaStream_t stream) {
        switch (numThreads) {
            case (1):
                count_incidence_kernel<1> << < P, 1,
                        3 * P * sizeof(double), stream >> > (data, N, P, incidence_count);
                break;
            case (2):
                count_incidence_kernel<2> << < P, 2,
                        3 * P * sizeof(double), stream >> > (data, N, P, incidence_count);
                break;
            case (4):
                count_incidence_kernel<4> << < P, 4,
                        3 * P * sizeof(double), stream >> > (data, N, P, incidence_count);
                break;
            case (8):
                count_incidence_kernel<8> << < P, 8,
                        3 * P * sizeof(double), stream >> > (data, N, P, incidence_count);
                break;
            case (16):
                count_incidence_kernel<16> << < P, 16, 3 * P * sizeof(double), stream >> >
                                                                               (data, N, P, incidence_count);
                break;
            case (32):
                count_incidence_kernel<32> << < P, 32, 3 * P * sizeof(double), stream >> >
                                                                               (data, N, P, incidence_count);
                break;
            case (64):
                count_incidence_kernel<64> << < P, 64, 3 * P * sizeof(double), stream >> >
                                                                               (data, N, P, incidence_count);
                break;
            case (128):
                count_incidence_kernel<128> << < P, 128, 3 * P * sizeof(double), stream >> >
                                                                                 (data, N, P, incidence_count);
                break;
            case (256):
                count_incidence_kernel<256> << < P, 256, 3 * P * sizeof(double), stream >> >
                                                                                 (data, N, P, incidence_count);
                break;
            case (512):
                count_incidence_kernel<512> << < P, 512, 3 * P * sizeof(double), stream >> >
                                                                                 (data, N, P, incidence_count);
                break;
            case (1024):
                count_incidence_kernel<1024> << < P, 1024, 3 * P * sizeof(double), stream >> >
                                                                                   (data, N, P, incidence_count);
                break;
        }
    }

    ////The numblocks is assumed to be equal to the number of columns (P)
    ////Size of shared memory is equal to 3 times P

    template<unsigned int numThreads>
    __global__ void
    cumulate_kernel(int *data, const size_t N, const size_t P, double *AP, double *cumulate) {
        extern __shared__
        double partials2[];
        unsigned int tid = threadIdx.x;
        unsigned int bid = blockIdx.x;
        cuView<int> data_view(data, N, P);
        cuView<double> AP_view(AP, N, P);
        cuView<double> cumu_view(cumulate, 3, P);

        double num_pos = 0;
        double num_neutral = 0;
        double num_neg = 0;

//        size_t index = (size_t) tid;
        int index = tid;
        for (; index < N; index += numThreads) {
//            printf("This is row %d and column %d with value %f\n", index, bid, AP_view(index, bid));
            if (data_view(index, bid) == -1) {
                num_neg += AP_view(index, bid);
            } else if (data_view(index, bid) == 0) {
                num_neutral += AP_view(index, bid);
            } else {
                num_pos += AP_view(index, bid);
            }
//            printf("This is thread %d, with value %f \n", index, AP_view(index, bid));
        }


        __syncthreads();
        partials2[3 * tid] = num_neg;
        partials2[3 * tid + 1] = num_neutral;
        partials2[3 * tid + 2] = num_pos;
//        printf("num_pos is from thread %d, block %d, %f\n", tid, bid, partials2[3*tid+2]);

        if (numThreads >= 1024) {
            if (tid < 512) {
                partials2[3 * tid] += partials2[3 * tid + 3 * 512];
                partials2[3 * tid + 1] += partials2[3 * tid + 3 * 512 + 1];
                partials2[3 * tid + 2] += partials2[3 * tid + 3 * 512 + 2];
            }
            __syncthreads();
        }
        if (numThreads >= 512) {
            if (tid < 256) {
                partials2[3 * tid] += partials2[3 * tid + 3 * 256];
                partials2[3 * tid + 1] += partials2[3 * tid + 3 * 256 + 1];
                partials2[3 * tid + 2] += partials2[3 * tid + 3 * 256 + 2];
            }
            __syncthreads();
        }
        if (numThreads >= 256) {
            if (tid < 128) {
                partials2[3 * tid] += partials2[3 * tid + 3 * 128];
                partials2[3 * tid + 1] += partials2[3 * tid + 3 * 128 + 1];
                partials2[3 * tid + 2] += partials2[3 * tid + 3 * 128 + 2];
            }
            __syncthreads();
        }
        if (numThreads >= 128) {
            if (tid < 64) {
                partials2[3 * tid] += partials2[3 * tid + 3 * 64];
                partials2[3 * tid + 1] += partials2[3 * tid + 3 * 64 + 1];
                partials2[3 * tid + 2] += partials2[3 * tid + 3 * 64 + 2];
            }
        }
        __syncthreads();
        // warp synchronous at the end
        if (tid < 32) {
            volatile double *wsSum = partials2;
            if (numThreads >= 64) {
                if (tid < 32) {
                    wsSum[3 * tid] += wsSum[3 * tid + 3 * 32];
                    wsSum[3 * tid + 1] += wsSum[3 * tid + 3 * 32 + 1];
                    wsSum[3 * tid + 2] += wsSum[3 * tid + 3 * 32 + 2];
                }
            }
            if (numThreads >= 32) {
                if (tid < 16) {
                    wsSum[3 * tid] += wsSum[3 * tid + 3 * 16];
                    wsSum[3 * tid + 1] += wsSum[3 * tid + 3 * 16 + 1];
                    wsSum[3 * tid + 2] += wsSum[3 * tid + 3 * 16 + 2];
                }
            }
            if (numThreads >= 16) {
                if (tid < 8) {
                    wsSum[3 * tid] += wsSum[3 * tid + 3 * 8];
                    wsSum[3 * tid + 1] += wsSum[3 * tid + 3 * 8 + 1];
                    wsSum[3 * tid + 2] += wsSum[3 * tid + 3 * 8 + 2];
                }
            }
            if (numThreads >= 8) {
                if (tid < 4) {
                    wsSum[3 * tid] += wsSum[3 * tid + 3 * 4];
                    wsSum[3 * tid + 1] += wsSum[3 * tid + 3 * 4 + 1];
                    wsSum[3 * tid + 2] += wsSum[3 * tid + 3 * 4 + 2];
                }
            }
            if (numThreads >= 4) {
                if (tid < 2) {
                    wsSum[3 * tid] += wsSum[3 * tid + 3 * 2];
                    wsSum[3 * tid + 1] += wsSum[3 * tid + 3 * 2 + 1];
                    wsSum[3 * tid + 2] += wsSum[3 * tid + 3 * 2 + 2];
//                    printf("In the first step tid %d, bid %d, %f\n", tid, bid, wsSum[3 * tid + 2]);
                }
            }
            if (numThreads >= 2) {
                if (tid == 0) {
                    wsSum[3 * tid] += wsSum[3 * tid + 3];
                    wsSum[3 * tid + 1] += wsSum[3 * tid + 3 + 1];
                    wsSum[3 * tid + 2] += wsSum[3 * tid + 3 + 2];
//                    printf("In the second step tid %d, bid %d, %f\n", tid, bid,wsSum[3 * tid + 2]);
                }
            }
            if (tid == 0) {
                cumu_view(0, bid) = wsSum[0];
                cumu_view(1, bid) = wsSum[1];
                cumu_view(2, bid) = wsSum[2];
            }
        }
    }

    void
    cumulate_kernel_gpu(int *data, const size_t N, const size_t P, double *AP, double *cumulate,
                        unsigned int numThreads, cudaStream_t stream) {
        switch (numThreads) {
            case (1):
                cumulate_kernel<1> << < P, 1, 6 * numThreads * sizeof(double), stream >> > (data, N, P, AP, cumulate);
                break;
            case (2):
                cumulate_kernel<2> << < P, 2, 6 * numThreads * sizeof(double), stream >> > (data, N, P, AP, cumulate);
                break;
            case (4):
                cumulate_kernel<4> << < P, 4, 6 * numThreads * sizeof(double), stream >> > (data, N, P, AP, cumulate);
                break;
            case (8):
                cumulate_kernel<8> << < P, 8, 6 * numThreads * sizeof(double), stream >> > (data, N, P, AP, cumulate);
                break;
            case (16):
                cumulate_kernel<16> << < P, 16,
                        6 * numThreads * sizeof(double), stream >> > (data, N, P, AP, cumulate);
                break;
            case (32):
                cumulate_kernel<32> << < P, 32,
                        6 * numThreads * sizeof(double), stream >> > (data, N, P, AP, cumulate);
                break;
            case (64):
                cumulate_kernel<64> << < P, 64,
                        6 * numThreads * sizeof(double), stream >> > (data, N, P, AP, cumulate);
                break;
            case (128):
                cumulate_kernel<128> << < P, 128,
                        6 * numThreads * sizeof(double), stream >> > (data, N, P, AP, cumulate);
                break;
            case (256):
                cumulate_kernel<256> << < P, 256,
                        6 * numThreads * sizeof(double), stream >> > (data, N, P, AP, cumulate);
                break;
            case (512):
                cumulate_kernel<512> << < P, 512,
                        6 * numThreads * sizeof(double), stream >> > (data, N, P, AP, cumulate);
                break;
            case (1024):
                cumulate_kernel<1024> << < P, 1024,
                        6 * numThreads * sizeof(double), stream >> > (data, N, P, AP, cumulate);
                break;

        }
    }

    CUDA_KERNEL
    void
    set_z_kernel(int *data, double *matZ, double *solutions, size_t N, size_t P) {
        cuView<int> data_view(data, N, P);
        cuView<double> mat_z_view(matZ, N, P);
        cuView<double> solutions_view(solutions, 3, P);

        unsigned int tid = threadIdx.x;
        unsigned int bid = blockIdx.x;
//        printf("This is thread %d from block %d \n", tid, bid);
        for (; tid < N; tid += blockDim.x) {
            if (data_view(tid, bid) == -1) {
                mat_z_view(tid, bid) = solutions_view(0, bid);
            } else if (data_view(tid, bid) == 0) {
                mat_z_view(tid, bid) = solutions_view(1, bid);
            } else {
                mat_z_view(tid, bid) = solutions_view(2, bid);
            }
        }
    }

    void set_z_gpu(int *data, double *matZ, double *solutions, size_t N, size_t P, unsigned int numThreads,
                   cudaStream_t stream) {
        set_z_kernel << < P, numThreads, 0, stream >> > (data, matZ, solutions, N, P);
    }
}////End of namespace NSPCA
