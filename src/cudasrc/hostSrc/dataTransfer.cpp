//
// Created by user on 25-11-16.
//

#include "dataTransfer.h"
#include <cassert>
#include "../devSrc/cuMatrix.h"
#include "../devSrc/cuView.h"

namespace cuExec {
//
    template<typename Scalar, template<typename> class Host,
            template<typename> class Device>
    void copyFromHostAsync(const Host<Scalar> &host, Device<Scalar> &device, cudaStream_t stream) {
        assert(host.rows() == device.rows());
        assert(host.cols() == device.cols());

        cudaMemcpyAsync(device.getPtr(), host.getPtr(), sizeof(Scalar) * host.rows() * host.cols(),
                        cudaMemcpyHostToDevice, stream);

    }

    template<typename Scalar, template<typename> class Host,
            template<typename> class Device>
    void copyToHostAsync(Host<Scalar> &host, const Device<Scalar> &device, cudaStream_t stream) {
        assert(host.rows() == device.rows());
        assert(host.cols() == device.cols());

        cudaMemcpyAsync(host.getPtr(), device.getPtr(), sizeof(Scalar) * host.rows() * host.cols(),
                        cudaMemcpyDeviceToHost, stream);
    }

    template void
    copyFromHostAsync<double, cuView, cuView>(const cuView<double> &, cuView<double> &, cudaStream_t);

    template void copyFromHostAsync<int, cuView, cuView>(const cuView<int> &, cuView<int> &, cudaStream_t);

    template void copyFromHostAsync<double, cuView, cuMatrix>(const cuView<double> &, cuMatrix<double> &,
                                                              cudaStream_t);

    template void copyFromHostAsync<int, cuView, cuMatrix>(const cuView<int> &, cuMatrix<int> &, cudaStream_t);

    template void copyToHostAsync<int, cuView, cuView>(cuView<int> &, const cuView<int> &, cudaStream_t);

    template void copyToHostAsync<int, cuView, cuMatrix>(cuView<int> &, const cuMatrix<int> &, cudaStream_t);

    template void copyToHostAsync<double, cuView, cuView>(cuView<double> &, const cuView<double> &, cudaStream_t);

    template void copyToHostAsync<double, cuView, cuMatrix>(cuView<double> &, const cuMatrix<double> &, cudaStream_t);


}