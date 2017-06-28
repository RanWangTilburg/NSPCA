//
// Created by user on 18-11-16.
//

#ifndef NSPCA_WORKSPACE_H
#define NSPCA_WORKSPACE_H
#include <cstdlib>
#include <cuda_runtime.h>
namespace cuExec{
    template<typename Scalar>
    class devWorkSpace {
    private:
        Scalar * lwork;
    public:
        Scalar *getLWork() const;

    private:
        size_t lsize;
    public:
        size_t getLSize() const;

    private:
        Scalar * rwork;
    public:
        Scalar *getRWork() const;

    private:
        size_t  rsize;
    public:
        size_t  getRSize() const;

    public:
        devWorkSpace(const size_t default_lsize, const size_t default_rsize);
//        devWorkSpace(const size_t initSize);
        ~devWorkSpace();
        void resizeLWork(const size_t newSize);
        void resizeRWork(const size_t newSize);
    };

    template class devWorkSpace<double>;
    template class devWorkSpace<float>;
    template class devWorkSpace<int>;
}


#endif //NSPCA_WORKSPACE_H
