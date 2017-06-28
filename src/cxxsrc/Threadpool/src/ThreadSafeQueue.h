#pragma once
#include <queue>
#include <mutex>
#include <condition_variable>
#include <memory>
#include "FunctionWrapper.h"
template<typename T>
class threadsafe_queue {
private:
    mutable std::mutex mut;
    std::queue<T> data_queue;
    std::condition_variable data_cond;

public:
    threadsafe_queue();
    void push(T new_value);
    void wait_and_pop(T& value);
    std::shared_ptr<T> wait_and_pop();
    bool try_pop(T& value);
    std::shared_ptr<T> try_pop();
    bool empty() const;
};



template<typename T>
threadsafe_queue<T>::threadsafe_queue(){};

template<typename T>
void threadsafe_queue<T>::push(T new_value){
    std::lock_guard<std::mutex> lk(mut);
    data_queue.push(std::move(new_value));
    data_cond.notify_one();
}

template<typename T>
void threadsafe_queue<T>::wait_and_pop(T& value){
    std::unique_lock<std::mutex> lk(mut);
    data_cond.wait(lk,[this]{return !data_queue.empty();});
    value=std::move(data_queue.front());
    data_queue.pop();
}

template<typename T>
std::shared_ptr<T> threadsafe_queue<T>::wait_and_pop(){
    std::unique_lock<std::mutex> lk(mut);
    data_cond.wait(lk,[this]{return !data_queue.empty();});
    std::shared_ptr<T> res(std::make_shared<T>(std::move(data_queue.front())));

    data_queue.pop();
    return res;
}
template<typename T>
bool threadsafe_queue<T>::try_pop(T& value)
{
    std::lock_guard<std::mutex> lk(mut);
    if(data_queue.empty())
        return false;
    value=std::move(data_queue.front());
    data_queue.pop();
    return true;
}
template<typename T>
std::shared_ptr<T> threadsafe_queue<T>::try_pop(){
    std::lock_guard<std::mutex> lk(mut);
    if(data_queue.empty())
        return std::shared_ptr<T>();
    std::shared_ptr<T> res(std::make_shared<T>(std::move(data_queue.front())));
    data_queue.pop();
    return res;
}
template<typename T>
bool threadsafe_queue<T>::empty() const {
    std::lock_guard<std::mutex> lk(mut);
    return data_queue.empty();
}
