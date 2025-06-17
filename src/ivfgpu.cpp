#include <iostream>
#include <cstdio>
#include "ivfgpu.h"
#include <sycl/sycl.hpp>

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>

using namespace std;
using namespace sycl;

queue* Q;

void init_gpu() {
    Q = new queue();
 
}

void* init_gpu_memory(void** P, int size) {    
    return malloc_device(size, *Q);
}

void free_gpu_memory(void* P) {
    free(P,*Q);
}

void copy_memory_to_cpu(void* T, void* F, int size) {
    Q->memcpy(T, F, size);
    Q->wait();
}

void copy_memory_to_gpu(void* T, void* F, int size) {
    Q->memcpy(T, F, size);
    Q->wait();
}

struct comparator {

    bool operator()(const sort_item& a, const sort_item& b) const {
        return a.distance < b.distance;
    }
  };

void sort_item_array_gpu(sort_item* P, int N) {
    auto policy = oneapi::dpl::execution::make_device_policy(*Q);
    
    oneapi::dpl::sort(policy, P, P+N, comparator());
    
    Q->wait();
}

void calc_squared_euclidean_distances(float* M, float* V, sort_item* C, int* p, int N, int L, int probe) {
    
    Q->parallel_for(range<1>(N),
    [=](id<1> k){ 
        int pos = *p;

        float tmp = 0;
        for(int i = 0; i < L; i++) {
            tmp += (M[L*k+i] - V[i])*(M[L*k+i] - V[i]);
        }
        
        C[pos+k].distance = tmp;
        C[pos+k].probe = probe;
        C[pos+k].pos = k;
       
    });

    Q->wait();
}


/*
    Squared euclidean distances with == filter
*/
void calc_squared_euclidean_distances_weqfilter(float* M, float* V, sort_item* C, const float f, int* p, int N, int L, int probe) {
    
    Q->parallel_for(range<1>(N),
    [=](id<1> k){ 
      
        float tmp = (M[L*k] - V[0])*(M[L*k] - V[0]);
        for(int i = 1; i < L; i++) {
            tmp += (M[L*k+i] - V[i])*(M[L*k+i] - V[i]);
        }
        
        if(tmp == f) {

            auto a = atomic_ref<int,sycl::memory_order_relaxed,memory_scope_device,access::address_space::global_space>(p[0]);
            
            int pos = a.fetch_add(1);
            
            C[pos].distance = tmp;
            C[pos].probe = probe;
            C[pos].pos = k;
        }
    });

    Q->wait();
}

/*
    Squared euclidean distances with < filter
*/
void calc_squared_euclidean_distances_wsfilter(float* M, float* V, sort_item* C, const float f, int* p, int N, int L, int probe) {
    
    Q->parallel_for(range<1>(N),
    [=](id<1> k){ 
      
        float tmp = (M[L*k] - V[0])*(M[L*k] - V[0]);
        for(int i = 1; i < L; i++) {
            tmp += (M[L*k+i] - V[i])*(M[L*k+i] - V[i]);
        }
        
        if(tmp < f) {

            auto a = atomic_ref<int,sycl::memory_order_relaxed,memory_scope_device,access::address_space::global_space>(p[0]);
            
            int pos = a.fetch_add(1);
            
            C[pos].distance = tmp;
            C[pos].probe = probe;
            C[pos].pos = k;
        }
    });

    Q->wait();
}

/*
    Squared euclidean distances with > filter
*/
void calc_squared_euclidean_distances_wlfilter(float* M, float* V, sort_item* C, const float f, int* p, int N, int L, int probe) {
    
    Q->parallel_for(range<1>(N),
    [=](id<1> k){ 
      
        float tmp = (M[L*k] - V[0])*(M[L*k] - V[0]);
        for(int i = 1; i < L; i++) {
            tmp += (M[L*k+i] - V[i])*(M[L*k+i] - V[i]);
        }
        
        if(tmp > f) {

            auto a = atomic_ref<int,sycl::memory_order_relaxed,memory_scope_device,access::address_space::global_space>(p[0]);
            
            int pos = a.fetch_add(1);
            
            C[pos].distance = tmp;
            C[pos].probe = probe;
            C[pos].pos = k;
        }
    });

    Q->wait();
}


/*
    Squared euclidean distances with <= filter
*/
void calc_squared_euclidean_distances_wseqfilter(float* M, float* V, sort_item* C, const float f, int* p, int N, int L, int probe) {
    
    Q->parallel_for(range<1>(N),
    [=](id<1> k){ 
      
        float tmp = (M[L*k] - V[0])*(M[L*k] - V[0]);
        for(int i = 1; i < L; i++) {
            tmp += (M[L*k+i] - V[i])*(M[L*k+i] - V[i]);
        }
        
        if(tmp <= f) {

            auto a = atomic_ref<int,sycl::memory_order_relaxed,memory_scope_device,access::address_space::global_space>(p[0]);
            
            int pos = a.fetch_add(1);
            
            C[pos].distance = tmp;
            C[pos].probe = probe;
            C[pos].pos = k;
        }
    });

    Q->wait();
}

/*
    Squared euclidean distances with >= filter
*/
void calc_squared_euclidean_distances_wleqfilter(float* M, float* V, sort_item* C, const float f, int* p, int N, int L, int probe) {
    
    Q->parallel_for(range<1>(N),
    [=](id<1> k){ 
      
        float tmp = (M[L*k] - V[0])*(M[L*k] - V[0]);
        for(int i = 1; i < L; i++) {
            tmp += (M[L*k+i] - V[i])*(M[L*k+i] - V[i]);
        }
        
        if(tmp >= f) {

            auto a = atomic_ref<int,sycl::memory_order_relaxed,memory_scope_device,access::address_space::global_space>(p[0]);
            
            int pos = a.fetch_add(1);
            
            C[pos].distance = tmp;
            C[pos].probe = probe;
            C[pos].pos = k;
        }
    });

    Q->wait();
}


/*
    Calc euclidean distances and apply < filter
*/
void calc_squared_distances_gpu_euclidean_wfilter(float* M, float* V, sort_item* C, const float f, int* p, int N, int L, int probe, int op) {
    
    Q->wait();
           
    // Calc distance + filter
    switch(op) {
        case 0:
            calc_squared_euclidean_distances_weqfilter(M, V, C, f, p, N, L, probe);
            break;
        case -1:
            calc_squared_euclidean_distances_wsfilter(M, V, C, f, p, N, L, probe);
            break;
        case 1:
            calc_squared_euclidean_distances_wlfilter(M, V, C, f, p, N, L, probe);
            break;
        case -2:
            calc_squared_euclidean_distances_wseqfilter(M, V, C, f, p, N, L, probe);
            break;
        case 2:
            calc_squared_euclidean_distances_wleqfilter(M, V, C, f, p, N, L, probe);
            break;
        default:
            calc_squared_euclidean_distances(M, V, C, p, N, L, probe);
            
            int pos;
            Q->memcpy(&pos,p,sizeof(int));
            Q->wait();
            pos += N;
            Q->memcpy(p,&pos,sizeof(int));
            Q->wait();
    }       
}
