//
// Created by user on 16-11-16.
//

#ifndef NSPCA_STREAMPOOL_H
#define NSPCA_STREAMPOOL_H

#include <vector>

using std::vector;

#include <string>

using std::string;

#include <cuda_runtime.h>

namespace cuExec {

    struct namedStream {
        int id;
        string name;
        cudaStream_t stream;

        namedStream(const int _id, const string _name);

        namedStream(const int _id);

        ~namedStream();

    };


    class streamPool {
    private:
        vector<namedStream *> streams;
        size_t size;
        size_t counter;
    public:
        streamPool(const int numStreams);

        ~streamPool();

        void createStream(string name);

        cudaStream_t getCurrentStream();
        cudaStream_t getNextStream();
        cudaStream_t getStream(int id);

        cudaStream_t getStream(string name);

        size_t getTotalStream();

        void syncStream(const int id);

        void syncStream(string name);

        void syncAll();
    };


}

#endif //NSPCA_STREAMPOOL_H
