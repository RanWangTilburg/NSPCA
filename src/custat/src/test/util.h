#pragma once

////This is a file created majorly for the purpose of interop with Eigen
////Note that no error handling is created as this is only for test purposes
////This file also contains a class to print out the progress
#include <iostream>
using std::cout;
using std::flush;
#include <Eigen/Dense>
#include <cuda_runtime.h>
#include "predeclare.h"

////A macro to indicate the number of lines where exception occurs
#ifndef OUTPUT_FILE_LINE
#define OUTPUT_FILE_LINE (std::cerr << "Error occurs in file: " << __FILE__ << " line: " << __LINE__ << "\n")
#endif

template<typename Scalar>
void copy_from_eigen(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& from, cuStat::Matrix<Scalar>& to);


template<typename Scalar>
void copy_to_eigen(Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& from, cuStat::Matrix<Scalar>& to);



////Check whether the results are similar in a elementwise sense
////If any of the element exceeds a difference of threshold, a "false" will be returned
template<typename Scalar>
bool assert_near(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& lhs, cuStat::Matrix<Scalar>& rhs, const double threshold);


////Implementation

template<typename Scalar>
void copy_from_eigen(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &from, cuStat::Matrix<Scalar> &to) {
    Scalar * hostPtr = (Scalar*)from.data();
    Scalar * devPtr  = to.data();

    cudaDeviceSynchronize();
    cudaMemcpy(devPtr, hostPtr, sizeof(Scalar)*to.rows()*to.cols(), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
}

template<typename Scalar>
void copy_to_eigen(Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &from, cuStat::Matrix<Scalar> &to) {
    Scalar * hostPtr = (Scalar*)from.data();
    Scalar * devPtr  = to.data();

    cudaDeviceSynchronize();
    cudaMemcpy(hostPtr, devPtr, sizeof(Scalar)*to.rows()*to.cols(), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
}

template<typename Scalar>
void copy_to_eigen(Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &from, cuStat::View<Scalar> &to) {
    Scalar * hostPtr = (Scalar*)from.data();
    Scalar * devPtr  = to.data();

    cudaDeviceSynchronize();
    cudaMemcpy(hostPtr, devPtr, sizeof(Scalar)*to.rows()*to.cols(), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
}

template<typename Scalar>
bool assert_near(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &lhs, cuStat::Matrix<Scalar> &rhs,
                 const double threshold) {
    if (lhs.rows()!=rhs.rows()||lhs.cols()!=rhs.cols()){
        return false;
    }
    else {
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> temp = lhs;

        copy_to_eigen(temp, rhs);

        for (size_t i = 0; i < temp.rows(); i++) {
            for (size_t j =0; j< temp.cols();j++){
                if (threshold < std::abs( temp(i,j)-lhs(i,j))){
                    return false;
                }
            }
        }
        return true;
    }
}

template<typename Scalar>
bool assert_near_abs(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &lhs, cuStat::Matrix<Scalar> &rhs,
                 const double threshold) {
    if (lhs.rows()!=rhs.rows()||lhs.cols()!=rhs.cols()){
        return false;
    }
    else {
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> temp = lhs;

        copy_to_eigen(temp, rhs);

        for (size_t i = 0; i < temp.rows(); i++) {
            for (size_t j =0; j< temp.cols();j++){
                if (threshold < std::abs( std::abs(temp(i,j))-std::abs(lhs(i,j)))){
                    return false;
                }
            }
        }
        return true;
    }
}
template<typename Scalar>
bool assert_near_abs(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &lhs, cuStat::View<Scalar> &rhs,
                     const double threshold) {
    if (lhs.rows()!=rhs.rows()||lhs.cols()!=rhs.cols()){
        return false;
    }
    else {
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> temp = lhs;

        copy_to_eigen(temp, rhs);

        for (size_t i = 0; i < temp.rows(); i++) {
            for (size_t j =0; j< temp.cols();j++){
                if (threshold < std::abs( std::abs(temp(i,j))-std::abs(lhs(i,j)))){
                    return false;
                }
            }
        }
        return true;
    }
}
class pBar {
public:
    void update(double newProgress) {
        currentProgress += newProgress;
        amountOfFiller = (int)((currentProgress / neededProgress)*(double)pBarLength);
    }
    void print() {
        currUpdateVal %= pBarUpdater.length();
        cout << "\r" //Bring cursor to start of line
             << firstPartOfpBar; //Print out first part of pBar
        for (int a = 0; a < amountOfFiller; a++) { //Print out current progress
            cout << pBarFiller;
        }
        cout << pBarUpdater[currUpdateVal];
        for (int b = 0; b < pBarLength - amountOfFiller; b++) { //Print out spaces
            cout << " ";
        }
        cout << lastPartOfpBar //Print out last part of progress bar
             << " (" << (int)(100*(currentProgress/neededProgress)) << "%)" //This just prints out the percent
             << flush;
        currUpdateVal += 1;
    }
    void complete(){
        cout << "\r" //Bring cursor to start of line
             << firstPartOfpBar; //Print out first part of pBar
        for (int a = 0; a < pBarLength; a++) { //Print out current progress
            cout << pBarFiller;
        }
        cout << lastPartOfpBar //Print out last part of progress bar
             << " (" << 100 << "%)" //This just prints out the percent
             << flush;
    }
    std::string firstPartOfpBar = "[", //Change these at will (that is why I made them public)
            lastPartOfpBar = "]",
            pBarFiller = "|",
            pBarUpdater = "/-\\|";
private:
    int amountOfFiller,
            pBarLength = 50, //I would recommend NOT changing this
            currUpdateVal = 0; //Do not change
    double currentProgress = 0, //Do not change
            neededProgress = 100; //I would recommend NOT changing this
};