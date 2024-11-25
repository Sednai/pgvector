#include "ivfgpu.h"
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

#define THREADS_PER_BLOCK 1024

__global__ void calc_euclidean_distances_v0(float* M, float* V, float* C, int N, int L) {
    int row = blockIdx.x*blockDim.x+threadIdx.x;

    if(row < N) {
        float tmp = 0;
        float tmp2;
        for(int i = 0; i < L; i++) {
            tmp2 = M[L*row+i] - V[i];
            tmp += tmp2*tmp2;
        }
        C[row] = sqrt(tmp);
    }
}

__global__ void calc_euclidean_distances_v1(float* M, float* V, float* C, int N, int L) {
    unsigned int indexx = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int stridex = blockDim.x*gridDim.x;
    unsigned int k;

    for(k=indexx; k < N; k += stridex) {
        float tmp = (M[L*k] - V[0])*(M[L*k] - V[0]);
        for(int i = 1; i < L; i++) {
            tmp += (M[L*k+i] - V[i])*(M[L*k+i] - V[i]);
        }
        C[k] = sqrt(tmp);
    }
}

__global__ void calc_squared_euclidean_distances_v0(float* M, float* V, float* C, int N, int L) {
    unsigned int indexx = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int stridex = blockDim.x*gridDim.x;
    unsigned int k;

    for(k=indexx; k < N; k += stridex) {
        float tmp = (M[L*k] - V[0])*(M[L*k] - V[0]);
        for(int i = 1; i < L; i++) {
            tmp += (M[L*k+i] - V[i])*(M[L*k+i] - V[i]);
        }
        C[k] = tmp;
    }
}

__global__ void calc_squared_euclidean_distances_v0b(float* M, float* V, float V2s, float* C, int N, int L) {
    unsigned int indexx = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int stridex = blockDim.x*gridDim.x;
    unsigned int k;

    for(k=indexx; k < N; k += stridex) {
        float tmp = V2s + M[L*k]*M[L*k] - 2*V[0]*M[L*k];
        for(int i = 1; i < L; i++) {
            tmp += M[L*k+i]*M[L*k+i] - 2*V[i]*M[L*k+i];
        }
        C[k] = tmp;
    }
}


__global__ void calc_squared_euclidean_distances_v0c(float* M, float* V, float* C, int N, int L) {
    unsigned int indexx = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int stridex = blockDim.x*gridDim.x;
    unsigned int k;

    __shared__ float VL[THREADS_PER_BLOCK];
    if(threadIdx.x < L)
        VL[threadIdx.x] = V[threadIdx.x];
    
    __syncthreads();
    
    for(k=indexx; k < N; k += stridex) {
        float tmp = (M[L*k] - VL[0])*(M[L*k] - VL[0]);
        for(int i = 1; i < L; i++) {
            tmp += (M[L*k+i] - VL[i])*(M[L*k+i] - VL[i]);
        }
        C[k] = tmp;
    }
}

__global__ void calc_squared_euclidean_distances_wsfilter_v0(float* M, float* V, sort_item* C, const float f, int* p, int N, int L, int probe) {
    unsigned int indexx = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int stridex = blockDim.x*gridDim.x;
    unsigned int k;

    __shared__ float VL[THREADS_PER_BLOCK];
    if(threadIdx.x < L)
        VL[threadIdx.x] = V[threadIdx.x];
    
    __syncthreads();
    
    for(k=indexx; k < N; k += stridex) {
        float tmp = (M[L*k] - VL[0])*(M[L*k] - VL[0]);
        for(int i = 1; i < L; i++) {
            tmp += (M[L*k+i] - VL[i])*(M[L*k+i] - VL[i]);
        }
        if(tmp < f) {
            int pos = atomicAdd(p,1);
            C[pos].distance = tmp;
            C[pos].probe = probe;
            C[pos].pos = k;
        }
    }
}



__global__ void calc_squared_euclidean_distances_v1(float* M, float* V, float* C, int N, int L) {
    unsigned int indexx = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int stridex = blockDim.x*gridDim.x;
    
    unsigned int indexy = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int stridey = blockDim.y*gridDim.y;

    unsigned int x,y;

    for(x = indexx; x < N; x += stridex) {
        float tmp = 0;
        for(y = indexy; y < L; y += stridey) {
            tmp += (M[L*x+y] - V[y]) * (M[L*x+y] - V[y]);
        }

        atomicAdd(&C[x],tmp);
    }
}

__global__ void calc_squared_euclidean_distances_v2(float* M, float* V, float* C, int N, int L) {
    unsigned int indexx = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int stridex = blockDim.x*gridDim.x;
    
    unsigned int indexy = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int stridey = blockDim.y*gridDim.y;

    unsigned int x,y;
    
    __shared__ float tmp[32]; 
    
    for(x = indexx; x < N; x += stridex) {
        tmp[threadIdx.x] = 0;
        __syncthreads();
        
        float tmp2 = 0;
        for(y = indexy; y < L; y += stridey) {
            tmp2 += (M[L*x+y] - V[y]) * (M[L*x+y] - V[y]);
        }
        
        atomicAdd(&tmp[threadIdx.x], tmp2);
        
        __syncthreads();
        
        if(threadIdx.y < 1)
            atomicAdd(&C[x],tmp[threadIdx.x]);
    }
}

__global__ void apply_sqrt(float* C, int N) {
    unsigned int indexx = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int stridex = blockDim.x*gridDim.x;

    for(int x = indexx; x < N; x += stridex) {
        C[x] = sqrt(C[x]);
    }
}

__global__ void apply_sqrt_with_seq_filter(float* C, float filter, int N) {
    unsigned int indexx = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int stridex = blockDim.x*gridDim.x;

    for(int x = indexx; x < N; x += stridex) {
        float tmp = (C[x]);
        if( tmp <= filter) 
            C[x] = tmp;
        else
            C[x] = -1;
    }
}

__global__ void nullify(float* C, int N) {
    unsigned int indexx = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int stridex = blockDim.x*gridDim.x;

    for(int x = indexx; x < N; x += stridex) {
        C[x] = 0;
    }
}

void* init_shared_gpu_memory(int size) {
    // Initialize unified memory
    void* M;
    cudaMallocManaged(&M, size);
    
    return M;
}

void init_gpu_memory(void** P, int size) {
    // Initialize non-unified memory
    cudaMalloc(P, size);    
}

void free_gpu_memory(void* P) {
    cudaFree(P);
}

void prefetch_gpu_memory(void* P, int size, int device) {
    cudaMemPrefetchAsync(P, size, device);
}

void advise_memory_readonly(void* P, int size, int device ) {
       cudaMemAdvise(P, size, cudaMemAdviseSetReadMostly, device);
}

void copy_memory_to_gpu(void* T, void* F, int size) {
    cudaMemcpy(T, F, size, cudaMemcpyHostToDevice);
}

void copy_memory_async_to_gpu(void* T, void* F, int size) {
    cudaMemcpyAsync(T, F, size, cudaMemcpyHostToDevice);
}

void copy_memory_to_cpu(void* T, void* F, int size) {
    cudaMemcpy(T, F, size, cudaMemcpyDeviceToHost);
}

void copy_memory_async_to_cpu(void* T, void* F, int size) {
    cudaMemcpyAsync(T, F, size, cudaMemcpyDeviceToHost);
}

void synchronize_gpu() {
    cudaDeviceSynchronize();
}

void calc_distances_gpu_euclidean(float* M, float* V, float* C, int N, int L) {
    
    //calc_euclidean_distances_v0<<<(N+THREADS_PER_BLOCK+1)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(M, V, C, N, L);    
    //calc_euclidean_distances_v1<<<1024,THREADS_PER_BLOCK>>>(M, V, C, N, L);    
    //calc_euclidean_distances_v1<<<(N+THREADS_PER_BLOCK+1)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(M, V, C, N, L);   
    
    dim3 DimGrid(1024, 2); 
    dim3 DimBlock(32, 32); 
    
    nullify<<<(N+THREADS_PER_BLOCK+1)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(C, N);    
    calc_squared_euclidean_distances_v2<<<DimGrid,DimBlock>>>(M, V, C, N, L);    
    apply_sqrt<<<(N+THREADS_PER_BLOCK+1)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(C, N);    
    
    cudaMemPrefetchAsync(C, N*sizeof(float), cudaCpuDeviceId);
    cudaDeviceSynchronize();
   
}


void calc_squared_distances_gpu_euclidean_nosharedmem(float* M, float* V, float* C, int N, int L) {
    
    calc_squared_euclidean_distances_v0c<<<(N-1+THREADS_PER_BLOCK)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(M, V, C, N, L);    
}

/*
    Calc euclidean distances and apply < filter
*/
void calc_squared_distances_gpu_euclidean_wsfilter(float* M, float* V, sort_item* C, const float f, int* p, int N, int L, int probe) {

    // Calc distance + filter
    calc_squared_euclidean_distances_wsfilter_v0<<<(N-1+THREADS_PER_BLOCK)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(M, V, C, f, p, N, L, probe);
}



void calc_squared_distances_gpu_euclidean(float* M, float* V, float* C, int N, int L) {
    
    calc_squared_euclidean_distances_v0<<<(N+THREADS_PER_BLOCK+1)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(M, V, C, N, L);    
    /*
    dim3 DimGrid(1024, 2); 
    dim3 DimBlock(32, 32); 
    
    nullify<<<(N+THREADS_PER_BLOCK+1)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(C, N);    
    calc_squared_euclidean_distances_v2<<<DimGrid,DimBlock>>>(M, V, C, N, L);    
    */
    cudaMemPrefetchAsync(C, N*sizeof(float), cudaCpuDeviceId);
    cudaDeviceSynchronize();
}

void calc_squared_distances_gpu_euclidean_mod(float* M, float* V, float V2s, float* C, int N, int L) {
    
    calc_squared_euclidean_distances_v0b<<<(N+THREADS_PER_BLOCK+1)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(M, V, V2s, C, N, L);    
    /*
    dim3 DimGrid(1024, 2); 
    dim3 DimBlock(32, 32); 
    
    nullify<<<(N+THREADS_PER_BLOCK+1)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(C, N);    
    calc_squared_euclidean_distances_v2<<<DimGrid,DimBlock>>>(M, V, C, N, L);    
    */
    cudaMemPrefetchAsync(C, N*sizeof(float), cudaCpuDeviceId);
    cudaDeviceSynchronize();
}


struct cmp : public thrust::less<page_item>
{
   __inline__
   __host__ __device__
   bool operator()(const page_item& a, const page_item& b) const {
      return a.distance < b.distance;
   }
};

void sort_array_gpu(page_item* P, int N) {
    thrust::sort(thrust::device, P, P + N, cmp() );
    cudaMemPrefetchAsync(P, N*sizeof(page_item), cudaCpuDeviceId);
    cudaDeviceSynchronize();
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