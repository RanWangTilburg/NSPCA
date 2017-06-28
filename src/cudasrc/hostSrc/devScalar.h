//
// Created by user on 17-11-16.
//

#ifndef NSPCA_DEVSCALAR_H
#define NSPCA_DEVSCALAR_H
#include <cuda_runtime.h>
namespace cuExec {
    template<typename Scalar>
    class devScalar {
        Scalar * dev_ptr;
        Scalar host_value;
    public:
        devScalar(const Scalar _host_value);
        ~devScalar();
        void syncDeviceToHost();
        void syncHostToDevice();
        Scalar getHostValue();
        Scalar * getPtr();
    };

    template class devScalar<double>;
    template class devScalar<int>;
    template class devScalar<float>;
}


#endif //NSPCA_DEVSCALAR_H
