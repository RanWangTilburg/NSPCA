//
// Created by user on 16-11-16.
//

#include <string>
#include "streamPool.h"
#include <algorithm>

namespace cuExec {

    streamPool::streamPool(const int numStreams) {
        for (int i = 0; i < numStreams; i++) {
            streams.push_back(new namedStream(i, std::to_string(i)));
        }
        size = numStreams;
        counter = 0;
    }

    void streamPool::createStream(string name) {
        streams.push_back(new namedStream(size, name));
        size += 1;
    }


    size_t streamPool::getTotalStream() {
        return size;
    }

    cudaStream_t streamPool::getStream(int i) {
        for (auto j = 0; j < streams.size(); j++) {
            if (i == streams[j]->id) {
                return streams[j]->stream;
            }
        }

        return 0;
    }

    cudaStream_t streamPool::getStream(string name) {
        for (auto i = 0; i < streams.size(); i++) {
            if (name == streams[i]->name) {
                return streams[i]->stream;
            }
        }
        return 0;
    }

    void streamPool::syncStream(const int id) {
        cudaStream_t stream = getStream(id);
        cudaStreamSynchronize(stream);
    }

    void streamPool::syncStream(string name) {
        cudaStream_t stream = getStream(name);
        cudaStreamSynchronize(stream);
    }

    streamPool::~streamPool() {
        syncAll();
        for (int i = 0; i < size; i++) {
            delete streams[i];
        }
    }

    void streamPool::syncAll() {
        for (int i = 0; i < getTotalStream(); i++) {
            cudaStreamSynchronize(getStream(i));
        }
        counter = 0;

    }

    cudaStream_t streamPool::getCurrentStream() {
        return streams[counter]->stream;
    }

    cudaStream_t streamPool::getNextStream() {
        if (counter < size-1){
            counter++;
            return getCurrentStream();
        }
        else {
            counter =0;
            return getCurrentStream();
        }

    }


    namedStream::namedStream(const int _id, const string _name) : id(_id), name(_name) {
        cudaStreamCreate(&stream);
    }

    namedStream::namedStream(const int _id) : id(_id) {
        name = std::to_string(id);
        cudaStreamCreate(&stream);
    }

    namedStream::~namedStream() {
        cudaStreamDestroy(stream);
    }


}