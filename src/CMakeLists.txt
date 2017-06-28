CMAKE_MINIMUM_REQUIRED(VERSION 3.6)
project(NSPCAlib C CXX Fortran)

#include(CMakeForceCompiler)
#set(INTEL_HOME /opt/intel/bin)
#CMAKE_FORCE_C_COMPILER(${INTEL_HOME}/icc "Intel C Compiler")
#CMAKE_FORCE_CXX_COMPILER(${INTEL_HOME}/icpc "Intel C++ Compiler")
#CMAKE_FORCE_Fortran_COMPILER(${INTEL_HOME}/ifort "Intel Fortran Compiler")

if (CMAKE_BUILD_TYPE MATCHES Debug)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0   -Wall -std=c++11 -fopenmp -pthread ")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}; -g -G -O0  -gencode arch=compute_50,code=sm_50")
    set(CMAKE_Fortran_FLAGS "-O0 -fopenmp  ")
ELSEIF (CMAKE_BUILD_TYPE MATCHES Release)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -O2 -Wno-error -g -std=c++11 -fopenmp -pthread ")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};  -O2 -G -g -gencode arch=compute_50,code=sm_50")
    set(CMAKE_Fortran_FLAGS "-g -O2 -fopenmp  ")
ENDIF ()
###########################################################################
find_package(CUDA)
SET(EXTRA_LIBS ${CMAKE_THREAD_LIBS_INIT} ${CUDA_LIBRARIES})
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")
find_package(Boost)

include_directories(${Boost_INCLUDE_DIRS})

include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
include_directories(../lib/Eigen3/)
include_directories(/home/user/anaconda2/include/python2.7)
include_directories(/home/user/anaconda2/lib/python2.7/site-packages/numpy/core/include/)
include_directories(/usr/local/cuda-8.0/include)

link_directories(/usr/local/cuda-8.0/lib64)
###########################################################################
#############################Fortran Sources###############################
###########################################################################

add_library(nspcaFortranlib SHARED
        fortransrc/cumulate_cpu.f90
        fortransrc/GramSchmidt.f90
        )

###########################################################################
#############################Cuda Sources##################################
###########################################################################
#cuda_add_library(nspcaCudalib SHARED
#        cudasrc/devSrc/cuMatExecDevice.cu
#        cudasrc/devSrc/cuMatExecDevice.h
#        cudasrc/devSrc/cuMatrix.cu
#        cudasrc/devSrc/cuMatrix.h
#        cudasrc/devSrc/cuView.cu
#        cudasrc/devSrc/managed.cu
#        cudasrc/devSrc/base.cu
#        )

cuda_add_library(cuStat SHARED
        custat/cuStat.cu)
target_link_libraries(cuStat
        -lcusolver -lcublas -lcusparse -lcurand -lcudart)

cuda_add_library(nspcaCudalib SHARED
        nspcaCudalib/nspca_cuda_v2.cu
        nspcaCudalib/nspca_cuda_v1.cu
        nspcaCudalib/util.cu)
target_link_libraries(nspcaCudalib
        cuStat)
###########################################################################
#############################CXX Sources###################################
###########################################################################
add_library(nspcaCXXlib SHARED
        cxxsrc/Solver.cpp
        cxxsrc/SolverV2Init.h
        cxxsrc/Solver.h
        cxxsrc/Constants.h
        cxxsrc/Dim.h
        cxxsrc/TmpMatrices.h
        cxxsrc/transformed_score.h
        cxxsrc/util.h
        )
set_target_properties(nspcaCXXlib PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(nspcaCXXlib
        nspcaFortranlib
        nspcaCudalib
        ${Boost_LIBRARIES})
###########################################################################
#############################Cython Compile################################
###########################################################################
add_custom_command(
        OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/cython/nspcagpu.cpp
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cython/nspcagpu.pyx
        COMMAND cython nspcagpu.pyx --cplus
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/cython
)
add_library(nspcagpu SHARED
        cython/nspcagpu.cpp)
set_target_properties(nspcagpu PROPERTIES PREFIX "")
target_link_libraries(nspcagpu nspcaCXXlib)

install(TARGETS cuStat nspcagpu nspcaCXXlib nspcaFortranlib nspcaCudalib
        DESTINATION /home/user/anaconda2/lib/python2.7
        )


#add_executable(test_init test/test.cpp)
#target_link_libraries(test_init gtest gtest_main nspcaCXXlib)

###########################################################################
#############################TESTS#########################################
###########################################################################
