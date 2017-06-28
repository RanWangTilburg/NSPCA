//
// Created by user on 25-11-16.
//

#ifndef NSPCA_DATATRANSFER_H
#define NSPCA_DATATRANSFER_H
#include <cuda_runtime.h>
namespace cuExec{


    template<typename Scalar, template<typename> class Host, template<typename> class Device>
    void copyFromHostAsync(const Host<Scalar> &host, Device<Scalar> &device, cudaStream_t stream);


    template<typename Scalar, template<typename> class Host, template<typename> class Device>
    void copyToHostAsync( Host<Scalar> &host, const Device<Scalar> &device, cudaStream_t stream);

}

#endif //NSPCA_DATATRANSFER_H
