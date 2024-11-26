#undef float4

#include <iostream>
#include <chrono>
#include <random>
#include "ivfgpu.h"

using namespace std;

random_device rd;
mt19937 mt(rd());

/*
    Generate array of N uniform random numbers in [minv,maxv)
*/
template <typename T> void gen_array_randunif(T* A, T minv, T maxv, int N) {
    uniform_real_distribution<T> dist(minv, maxv);

    for(int i = 0; i < N; i++) {
        A[i] = dist(mt);
    }
}

template <typename T> void squared_eucl_distances_cpu(T* V, T* C, T* R, int N, int dim) {

    for(int v = 0; v < N; v++) {
        float tmp = 0;
        for(int i = 0; i < dim; i++) {
            tmp += (V[v*dim + i] - C[i])*(V[v*dim + i] - C[i]);
        }
        R[v] = tmp;
    }
}

template <typename T> T calc_l1_norm(T* R_1, T* R_2, int N) {
    T res = 0;
    for(int i = 0; i < N; i++) {
        res += abs(R_1[i] - R_2[i]);
    }

    return res/N;
}

template <typename T> T calc_max_diff(T* R_1, T* R_2, int N) {
    T res = 0;

    for(int i = 0; i < N; i++) {
        T d = abs(R_1[i] - R_2[i]);
        if( d > res) 
            res = d;
    }

    return res;
}


template <typename T> void verify(T* V, T* C, T* R_gpu, int N, int dim) {
    T* R_cpu = (T*) malloc(N*sizeof(T));

    auto tic = std::chrono::system_clock::now();

    squared_eucl_distances_cpu(V, C, R_cpu, N, dim);

    auto toc = std::chrono::system_clock::now();

    cout << "[RT](cpu): " << " (" << std::chrono::duration_cast<std::chrono::milliseconds>(toc-tic).count() << "ms)" << endl;

    cout << "Avg L1: " << calc_l1_norm<T>(R_gpu, R_cpu, N) << endl;
    cout << "Max L1: " << calc_max_diff<T>(R_gpu, R_cpu, N) << endl;

    free(R_cpu);
}


int main() {

    int N = 1*1e7;
    int dim = 87;

    // CPU
    float* V = (float*) malloc(N*dim*sizeof(float));
    gen_array_randunif<float>(V, -3, 3, N*dim);

    float* C = (float*) malloc(dim*sizeof(float));
    gen_array_randunif<float>(C, -1, 1, dim);

    float* R = (float*) malloc(N*sizeof(float));

    auto tic = std::chrono::system_clock::now();

    // GPU
    float* d_V; 
    cudaMalloc((void**) &d_V, N*dim*sizeof(float));
    cudaMemcpy(d_V, V, N*dim*sizeof(float), cudaMemcpyHostToDevice );

    float* d_C; 
    cudaMalloc((void**) &d_C, dim*sizeof(float));
    cudaMemcpy(d_C, C, dim*sizeof(float), cudaMemcpyHostToDevice );
   
    float* d_R; 
    cudaMalloc((void**) &d_R, N*sizeof(float));

    auto toc = std::chrono::system_clock::now();

    cout << "[RT](memcpy): " << " (" << std::chrono::duration_cast<std::chrono::milliseconds>(toc-tic).count() << "ms)" << endl;

    float total = 0;
    int reps = 1;
    for(int c = 0; c < reps; c++) { 
        tic = std::chrono::system_clock::now();
        
        calc_squared_distances_gpu_euclidean_nosharedmem(d_V, d_C, d_R, N, dim); 

        toc = std::chrono::system_clock::now();
        float tmp = std::chrono::duration_cast<std::chrono::microseconds>(toc-tic).count();
        
        cout << "[RT](kernel): #" << c << " (" << tmp << "micros)" << endl;
        
        total += tmp;

    }
    cout << "[RT](Kernel)(avg): (" << total/reps << "micros)" << endl;
 
    tic = std::chrono::system_clock::now();
 
    cudaMemcpy(R, d_R, N*sizeof(float), cudaMemcpyDeviceToHost);

    toc = std::chrono::system_clock::now();
    
    cout << "[RT](memcpy): " << " (" << std::chrono::duration_cast<std::chrono::milliseconds>(toc-tic).count() << "ms)" << endl;

    verify<float>(V, C, R, N, dim);

    // Cleanup
    free(V);
    free(C);
    cudaFree(d_V);
    cudaFree(d_C);
    cudaFree(d_R);
}