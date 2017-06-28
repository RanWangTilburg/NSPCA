#include "nspca_cuda.h"
#include "view.h"

namespace NSPCA {

    __device__ __forceinline__ void solve_p_positive(double *ATZ, double *Pptr, const size_t N, const size_t P, const size_t p,
                                          const int i, const int j, const double n_lambda,
                                          const double n_scale_square) {
        cuView<double> AtZview(ATZ, p, P);
        cuView<double> PView(Pptr, p, P);
        double tmp = 2.0 * AtZview(i, j);
        double t = -n_lambda + tmp;


        if (t > 0) {
            PView(i, j) = t / (2.0 * n_scale_square);
        } else {
            PView(i, j) = 0;
        }

    }

    __device__ __forceinline__ void
    solve_p_negative(double *ATZ, double *Pptr, const size_t N, const size_t P, const size_t p,
                     const int i, const int j, const double n_lambda, const double n_scale_square) {

        cuView<double> AtZview(ATZ, p, P);
        cuView<double> PView(Pptr, p, P);
        double tmp = 2.0 * AtZview(i, j);
        double t = n_lambda + tmp;

        if (t < 0) {
            PView(i, j) = t / (2 * n_scale_square);
        }

    }

    __device__ __forceinline__ void
    solve_p_general(double *ATZ, double *Pptr, const size_t N, const size_t P, const size_t p,
                    const int i, const int j, const double n_lambda, const double n_scale_square) {
        cuView<double> AtZview(ATZ, p, P);
        cuView<double> PView(Pptr, p, P);

//        double t = (2 * ZtAview(i, j) - n_lambda);
//        if (t > 0) {
//            PView(i, j) = t / (2 * n_scale_square);
//        } else {
//            t = (2 * ZtAview(i, j) + n_lambda);
//            {
//                if (t < 0) {
//                    PView(i, j) = t / (2 * n_scale_square);
//                }
//            }
//        }
        double tmp = 2.0 * AtZview(i, j);
        double t = -n_lambda + tmp;
        if (t > 0) {
            PView(i, j) = t / (2.0 * n_scale_square);
        } else {
            t = n_lambda + tmp;
            if (t < 0) {
                PView(i, j) = t / (2.0 * n_scale_square);
            } else {
                PView(i, j) = 0;
            }
        }


//        if self.n_alpha - 2.0 * temp_t < 0:
//        self.component_loading[row, col] = -(self.n_alpha - 2.0 * temp_t) / (2.0 * self.n_scale_square)
//        elif self.n_alpha + 2.0 * temp_t < 0:
//        self.component_loading[row, col] = (self.n_alpha + 2.0 * temp_t) / (2.0 * self.n_scale_square)
//        else:
//        self.component_loading[row, col] = 0.0
    }

    template<unsigned int numThreads>
    __global__ void solve_p_in_nspca(double *devp, const size_t N, const size_t P, const size_t p, double *ATZ,
                                     int *restriction, const double n_lambda, const double n_scale_square) {
        const int tid = threadIdx.x;
        const int offset = numThreads * blockIdx.x;
        cuView<int> resView(restriction, p, P);
        cuView<double> AtZview = cuView<double>(ATZ, p, P);
        for (int index = tid + offset; index < p * P; index += numThreads * blockDim.x) {
            int j = index / p;
            int i = index - j * p;
//            printf("Row %d and col %d \n", i ,j);
//            AtZview(i, j) = 0.0;
            if (resView(i, j) == 2) {

                solve_p_general(ATZ, devp, N, P, p, i, j, n_lambda, n_scale_square);

            } else if (resView(i, j) == 1) {
                solve_p_positive(ATZ, devp, N, P, p, i, j, n_lambda, n_scale_square);
            } else if (resView(i, j) == -1) {
                solve_p_negative(ATZ, devp, N, P, p, i, j, n_lambda, n_scale_square);
            }
        }
    };

    void solve_p_nspca(double *devp, const size_t N, const size_t P, const size_t p, double *ATZ,
                       int *restriction, const double lambda, const double scale_square, const unsigned int numThreads,
                       const unsigned int numBlocks, cudaStream_t stream) {
        switch (numThreads) {
            case (1):
                solve_p_in_nspca<1> << < numBlocks, 1, 0, stream >> >
                                                          (devp, N, P, p, ATZ, restriction, lambda, scale_square);
                break;
            case (2):
                solve_p_in_nspca<2> << < numBlocks, 2, 0, stream >> >
                                                          (devp, N, P, p, ATZ, restriction, lambda, scale_square);
                break;
            case (4):
                solve_p_in_nspca<4> << < numBlocks, 4, 0, stream >> >
                                                          (devp, N, P, p, ATZ, restriction, lambda, scale_square);
                break;
            case (8):
                solve_p_in_nspca<8> << < numBlocks, 8, 0, stream >> >
                                                          (devp, N, P, p, ATZ, restriction, lambda, scale_square);
                break;
            case (16):
                solve_p_in_nspca<16> << < numBlocks, 16, 0, stream >> >
                                                            (devp, N, P, p, ATZ, restriction, lambda, scale_square);
                break;
            case (32):
                solve_p_in_nspca<32> << < numBlocks, 32, 0, stream >> >
                                                            (devp, N, P, p, ATZ, restriction, lambda, scale_square);
                break;
            case (64):
                solve_p_in_nspca<64> << < numBlocks, 64, 0, stream >> >
                                                            (devp, N, P, p, ATZ, restriction, lambda, scale_square);
                break;
            case (128):
                solve_p_in_nspca<128> << < numBlocks, 128, 0, stream >> >
                                                              (devp, N, P, p, ATZ, restriction, lambda, scale_square);
                break;
            case (256):
                solve_p_in_nspca<256> << < numBlocks, 256, 0, stream >> >
                                                              (devp, N, P, p, ATZ, restriction, lambda, scale_square);
                break;
            case (512):
                solve_p_in_nspca<512> << < numBlocks, 512, 0, stream >> >
                                                              (devp, N, P, p, ATZ, restriction, lambda, scale_square);
                break;
            case (1024):
                solve_p_in_nspca<1024> << < numBlocks, 1024, 0, stream >> >
                                                                (devp, N, P, p, ATZ, restriction, lambda, scale_square);
                break;

        }

    }

}