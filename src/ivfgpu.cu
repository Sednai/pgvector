#include <stdio.h>

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

__global__ void calc_squared_euclidean_distances_v1(float* M, float* V, float* C, int N, int L) {
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

extern "C" void* init_shared_gpu_memory(int size) {
    // Initialize unified memory
    void* M;
    cudaMallocManaged(&M, size);
    
    return M;
}

extern "C" void init_gpu_memory(void** P, int size) {
    // Initialize non-unified memory
    cudaMalloc(P, size);    
}

extern "C" void free_gpu_memory(void* P) {
    cudaFree(P);
}

extern "C" void prefetch_gpu_memory(void* P, int size, int device) {
    cudaMemPrefetchAsync(P, size, device);
}

extern "C" void advise_memory_readonly(void* P, int size, int device ) {
       cudaMemAdvise(P, size, cudaMemAdviseSetReadMostly, device);
}

extern "C" void copy_memory_to_gpu(void* T, void* F, int size) {
    cudaMemcpy(T, F, size, cudaMemcpyHostToDevice);
}

extern "C" void copy_memory_async_to_gpu(void* T, void* F, int size) {
    cudaMemcpyAsync(T, F, size, cudaMemcpyHostToDevice);
}

extern "C" void copy_memory_to_cpu(void* T, void* F, int size) {
    cudaMemcpy(T, F, size, cudaMemcpyDeviceToHost);
}

extern "C" void copy_memory_async_to_cpu(void* T, void* F, int size) {
    cudaMemcpyAsync(T, F, size, cudaMemcpyDeviceToHost);
}

extern "C" void synchronize_gpu() {
    cudaDeviceSynchronize();
}

extern "C" void calc_distances_gpu_euclidean(float* M, float* V, float* C, int N, int L) {
    
    //calc_euclidean_distances_v0<<<(N+THREADS_PER_BLOCK+1)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(M, V, C, N, L);    
    //calc_euclidean_distances_v1<<<1024,THREADS_PER_BLOCK>>>(M, V, C, N, L);    
    //calc_euclidean_distances_v1<<<(N+THREADS_PER_BLOCK+1)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(M, V, C, N, L);   
    
    dim3 DimGrid(1024, 2); 
    dim3 DimBlock(32, 32); 
    
    nullify<<<(N+THREADS_PER_BLOCK+1)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(C, N);    
    calc_squared_euclidean_distances_v1<<<DimGrid,DimBlock>>>(M, V, C, N, L);    
    apply_sqrt<<<(N+THREADS_PER_BLOCK+1)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(C, N);    
    
    cudaMemPrefetchAsync(C, N*sizeof(float), cudaCpuDeviceId);
    cudaDeviceSynchronize();
   
}

