#pragma once
#include <thread>
#include <vector>

class join_thread{
    std::vector<std::thread>& threads;
public:
    explicit join_thread(std::vector<std::thread>& other);
    ~join_thread();
};


join_thread::join_thread(std::vector<std::thread>& other):threads(other){}

join_thread::~join_thread() {
    for(unsigned long i=0;i<threads.size();++i)
    {
        if(threads[i].joinable())
            threads[i].join();
    }
}