//
// Created by user on 25-11-16.
//

#ifndef NSPCA_INTERNAL_H
#define NSPCA_INTERNAL_H
#include "../devSrc/cuView.h"
#include "../devSrc/cuMatrix.h"

namespace cuExec{
    namespace internal{

        template<typename...T, template<typename...> class Class>
        struct get_type_impl{
        };

        template<typename Scalar>
        struct get_type_impl<Scalar, cuView<Scalar>>{
            using type = Scalar;
        };


    }
}
#endif //NSPCA_INTERNAL_H
