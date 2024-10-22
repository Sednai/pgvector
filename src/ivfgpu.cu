#include <stdio.h>

#define THREADS_PER_BLOCK 1024

// Note: First version just 1d parallelization over rows
__global__ void calc_euclidean_distances(float* M, float* V, float* C, int N, int L) {
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


extern "C" void* init_shared_gpu_memory(int size) {
    // Initialize unified memory
    void* M;
    cudaMallocManaged(&M, size);
    
    return M;
}

extern "C" void free_shared_gpu_memory(void* P) {
    cudaFree(P);
}

extern "C" void calc_distances_gpu_euclidean(float* M, float* V, float* C, int N, int L) {
    
    cudaDeviceSynchronize();
    
    calc_euclidean_distances<<<(N+THREADS_PER_BLOCK+1)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(M, V, C, N, L);    

    cudaDeviceSynchronize();

}

