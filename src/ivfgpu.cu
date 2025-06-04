#include "ivfgpu.h"
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

#define THREADS_PER_BLOCK 1024

void init_gpu_memory(void** P, int size) {
    // Initialize non-unified memory
    cudaMalloc(P, size);    
}

void free_gpu_memory(void* P) {
    cudaFree(P);
}

void copy_memory_to_cpu(void* T, void* F, int size) {
    cudaMemcpy(T, F, size, cudaMemcpyDeviceToHost);
}

void copy_memory_to_gpu(void* T, void* F, int size) {
    cudaMemcpy(T, F, size, cudaMemcpyHostToDevice);
}

struct cmp_item : public thrust::less<sort_item>
{
   __inline__
   __host__ __device__
   bool operator()(const sort_item& a, const sort_item& b) const {
      return a.distance < b.distance;
   }
};

void sort_item_array_gpu(sort_item* P, int N) {
    thrust::sort(thrust::device, P, P + N, cmp_item() );
}

/*
    Squared euclidean distances with < filter
    v0: N/stridex vec per thread
*/
__global__ void calc_squared_euclidean_distances_wsfilter_v0(float* M, float* V, sort_item* C, const float f, int* p, int N, int L, int probe) {
    unsigned int indexx = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int stridex = blockDim.x*gridDim.x;
    unsigned int k;

    __shared__ float VL[THREADS_PER_BLOCK];
    if(threadIdx.x < L)
        VL[threadIdx.x] = V[threadIdx.x];
    
    __syncthreads();
    
    for(k=indexx; k < N; k += stridex) {
        float tmp = 0;
        for(int i = 0; i < L; i++) {
            tmp += (M[L*k+i] - VL[i])*(M[L*k+i] - VL[i]);
        }
        if( tmp < f ) {
            int pos = atomicAdd(p,1);
            C[pos].distance = tmp;
            C[pos].probe = probe;
            C[pos].pos = k;
        }
    }
}

/*
    Squared euclidean distances with <= filter
    v0: N/stridex vec per thread
*/
__global__ void calc_squared_euclidean_distances_wseqfilter_v0(float* M, float* V, sort_item* C, const float f, int* p, int N, int L, int probe) {
    unsigned int indexx = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int stridex = blockDim.x*gridDim.x;
    unsigned int k;

    __shared__ float VL[THREADS_PER_BLOCK];
    if(threadIdx.x < L)
        VL[threadIdx.x] = V[threadIdx.x];
    
    __syncthreads();
    
    for(k=indexx; k < N; k += stridex) {
        float tmp = 0;
        for(int i = 0; i < L; i++) {
            tmp += (M[L*k+i] - VL[i])*(M[L*k+i] - VL[i]);
        }
        if( tmp <= f ) {
            int pos = atomicAdd(p,1);
            C[pos].distance = tmp;
            C[pos].probe = probe;
            C[pos].pos = k;
        }
    }
}

/*
    Squared euclidean distances with = filter
    v0: N/stridex vec per thread
*/
__global__ void calc_squared_euclidean_distances_weqfilter_v0(float* M, float* V, sort_item* C, const float f, int* p, int N, int L, int probe) {
    unsigned int indexx = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int stridex = blockDim.x*gridDim.x;
    unsigned int k;

    __shared__ float VL[THREADS_PER_BLOCK];
    if(threadIdx.x < L)
        VL[threadIdx.x] = V[threadIdx.x];
    
    __syncthreads();
    
    for(k=indexx; k < N; k += stridex) {
        float tmp = 0;
        for(int i = 0; i < L; i++) {
            tmp += (M[L*k+i] - VL[i])*(M[L*k+i] - VL[i]);
        }
        if( tmp == f ) {
            int pos = atomicAdd(p,1);
            C[pos].distance = tmp;
            C[pos].probe = probe;
            C[pos].pos = k;
        }
    }
}

/*
    Squared euclidean distances with > filter
    v0: N/stridex vec per thread
*/
__global__ void calc_squared_euclidean_distances_wlfilter_v0(float* M, float* V, sort_item* C, const float f, int* p, int N, int L, int probe) {
    unsigned int indexx = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int stridex = blockDim.x*gridDim.x;
    unsigned int k;

    __shared__ float VL[THREADS_PER_BLOCK];
    if(threadIdx.x < L)
        VL[threadIdx.x] = V[threadIdx.x];
    
    __syncthreads();
    
    for(k=indexx; k < N; k += stridex) {
        float tmp = 0;
        for(int i = 0; i < L; i++) {
            tmp += (M[L*k+i] - VL[i])*(M[L*k+i] - VL[i]);
        }
        if( tmp > f ) {
            int pos = atomicAdd(p,1);
            C[pos].distance = tmp;
            C[pos].probe = probe;
            C[pos].pos = k;
        }
    }
}

/*
    Squared euclidean distances with > filter
    v0: N/stridex vec per thread
*/
__global__ void calc_squared_euclidean_distances_wleqfilter_v0(float* M, float* V, sort_item* C, const float f, int* p, int N, int L, int probe) {
    unsigned int indexx = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int stridex = blockDim.x*gridDim.x;
    unsigned int k;

    __shared__ float VL[THREADS_PER_BLOCK];
    if(threadIdx.x < L)
        VL[threadIdx.x] = V[threadIdx.x];
    
    __syncthreads();
    
    for(k=indexx; k < N; k += stridex) {
        float tmp = 0;
        for(int i = 0; i < L; i++) {
            tmp += (M[L*k+i] - VL[i])*(M[L*k+i] - VL[i]);
        }
        if( tmp >= f ) {
            int pos = atomicAdd(p,1);
            C[pos].distance = tmp;
            C[pos].probe = probe;
            C[pos].pos = k; 
        }
    }
}

__global__ void calc_squared_euclidean_distances_v0d(float* M, float* V, sort_item* C, int* p, int N, int L, int probe) {
    unsigned int indexx = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int stridex = blockDim.x*gridDim.x;
    unsigned int k;
    int pos = *p;

    __shared__ float VL[THREADS_PER_BLOCK];
    if(threadIdx.x < L)
        VL[threadIdx.x] = V[threadIdx.x];
    
    __syncthreads();
    
    for(k=indexx; k < N; k += stridex) {
        float tmp = (M[L*k] - VL[0])*(M[L*k] - VL[0]);
        for(int i = 1; i < L; i++) {
            tmp += (M[L*k+i] - VL[i])*(M[L*k+i] - VL[i]);
        }
        C[pos+k].distance = tmp;
        C[pos+k].probe = probe;
        C[pos+k].pos = k;
    }
}


/*
    Calc euclidean distances and apply < filter
*/
void calc_squared_distances_gpu_euclidean_wfilter(float* M, float* V, sort_item* C, const float f, int* p, int N, int L, int probe, int op) {
    
    
    int NB = (N-1+THREADS_PER_BLOCK)/THREADS_PER_BLOCK;

    // Calc distance + filter
    switch(op) {
        case 0:
            calc_squared_euclidean_distances_weqfilter_v0<<<NB,THREADS_PER_BLOCK>>>(M, V, C, f, p, N, L, probe);
            break;
        case -1:
            calc_squared_euclidean_distances_wsfilter_v0<<<NB,THREADS_PER_BLOCK>>>(M, V, C, f, p, N, L, probe);
            break;
        case 1:
            calc_squared_euclidean_distances_wlfilter_v0<<<NB,THREADS_PER_BLOCK>>>(M, V, C, f, p, N, L, probe);
            break;
        case -2:
            calc_squared_euclidean_distances_wseqfilter_v0<<<NB,THREADS_PER_BLOCK>>>(M, V, C, f, p, N, L, probe);
            break;
        case 2:
            calc_squared_euclidean_distances_wleqfilter_v0<<<NB,THREADS_PER_BLOCK>>>(M, V, C, f, p, N, L, probe);
            break;
        default:
            calc_squared_euclidean_distances_v0d<<<NB,THREADS_PER_BLOCK>>>(M, V, C, p, N, L, probe);
            int pos;
            cudaMemcpy(&pos,p, sizeof(int), cudaMemcpyDeviceToHost);
            pos += N;
            cudaMemcpy(p,&pos, sizeof(int), cudaMemcpyHostToDevice);
    }       
}
