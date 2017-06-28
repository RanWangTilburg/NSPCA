//
// Created by user on 23-11-16.
//

#ifndef NSPCA_MANAGED_H
#define NSPCA_MANAGED_H
namespace cuExec {
    class Managed {
    public:
        void *operator new(size_t size);

        void operator delete(void *ptr);
    };
}

#endif //NSPCA_MANAGED_H
