#pragma once
#include <mutex>
#include <thread>
#include <condition_variable>
#include <vector>
#include <functional>
#include <atomic>
#include <future>

#include "ThreadJoin.h"
#include "ThreadSafeQueue.h"
#include "FunctionWrapper.h"
class ThreadPool {
    std::atomic_bool done;
    threadsafe_queue<function_wrapper> work_queue;
    std::vector<std::thread> threads;
    join_thread joiner;

    void worker_thread();
public:
    ThreadPool(const unsigned int);
    ~ThreadPool();
    int getNumThreads();
    template<typename FunctionType>
    std::future<typename std::result_of<FunctionType()>::type>
            submit(FunctionType f);
};

void ThreadPool::worker_thread()
{
    while(!done)
    {
        function_wrapper task;
        if(work_queue.try_pop(task))
        {
            task();
        }
        else
        {
            std::this_thread::yield();
        }
    }
}

ThreadPool::ThreadPool(const unsigned int thread_count):done(false),joiner(threads)
{
    //unsigned const thread_count=std::thread::hardware_concurrency()
    try
    {
        for(unsigned i=0;i<thread_count;++i)
        {
            threads.push_back(
                    std::thread(&ThreadPool::worker_thread,this));
        }
    }
    catch(...) //Catch all exceptions
    {
        done=true;
        throw;
    }
}

ThreadPool::~ThreadPool()
{
    done=true;
}

int ThreadPool::getNumThreads() {
    return this->threads.size();
}

template<typename FunctionType>
std::future<typename std::result_of<FunctionType()>::type> ThreadPool::submit(FunctionType f)
{
//        std::cout << "Work has been submitted\n";
    typedef typename std::result_of<FunctionType()>::type
            result_type;
    std::packaged_task<result_type()> task(std::move(f));
    std::future<result_type> res(task.get_future());
    work_queue.push(std::move(task));
    return res;
}


