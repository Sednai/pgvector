#include "ivfgpu.h"
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

#ifdef CUVS

#include <cstdio>

#include <dlpack/dlpack.h>
#include <cuvs/core/c_api.h>
#include <cuvs/core/exceptions.hpp>
#include <cuvs/core/interop.hpp>

#include <raft/core/resources.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <cuvs/distance/distance.hpp>
#include <raft/matrix/select_k.cuh>

#endif

#define THREADS_PER_BLOCK 1024


//#ifndef CUVS
void init_gpu() {}

void* init_gpu_memory(void** P, int size) {
    // Initialize non-unified memory
    cudaMalloc(P, size);   
    return *P; 
}

void free_gpu_memory(void* P) {
    cudaFree(P);
}
//#endif

void copy_memory_to_cpu(void* T, void* F, int size) {
    cudaMemcpy(T, F, size, cudaMemcpyDeviceToHost);
}

void copy_memory_to_gpu(void* T, void* F, int size) {
    cudaMemcpy(T, F, size, cudaMemcpyHostToDevice);
}

#ifdef CUVS
raft::device_resources handle;

//cuvsResources_t res;

/*
void init_gpu() {
    cuvsResourcesCreate(&res);
}

void init_gpu_memory(void** P, int size) {
    cuvsRMMAlloc(res,P,size);
}

void free_gpu_memory(void* P, int size) {
    cuvsRMMFree(res, P, size);
}
*/
void init_float_tensor(float* t_d, int64_t t_shape[2], DLManagedTensor* t_tensor)
{
  t_tensor->dl_tensor.data               = t_d;
  t_tensor->dl_tensor.device.device_type = kDLCUDA;
  t_tensor->dl_tensor.ndim               = 2;
  t_tensor->dl_tensor.dtype.code         = kDLFloat;
  t_tensor->dl_tensor.dtype.bits         = 32;
  t_tensor->dl_tensor.dtype.lanes        = 1;
  t_tensor->dl_tensor.shape              = t_shape;
  t_tensor->dl_tensor.strides            = NULL;
}

__global__ void set_sort_item(float* D, sort_item* C, int* p, int N, int probe) {
    unsigned int indexx = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int stridex = blockDim.x*gridDim.x;
    unsigned int k;
    int pos = *p;

    for(k=indexx; k < N; k += stridex) {
        C[pos+k].distance = D[k];
        C[pos+k].probe = probe;
        C[pos+k].pos = k;
    }
}

void calc_squared_euclidean_distances_cuvs(float* M, float* V, float* D, int* p, int N, int L, int probe) {
    raft::device_resources handle;

    auto metric = cuvs::distance::DistanceType::L2Expanded;
    
    /*
    DLManagedTensor i_tensor;
    int64_t i_shape[2]  = {N, L};
    DLManagedTensor v_tensor;
    int64_t v_shape[2]  = {1, L};
    DLManagedTensor o_tensor;
    int64_t o_shape[2]  = {N, 1};

    init_float_tensor(M, i_shape, &i_tensor);
    init_float_tensor(V, v_shape, &v_tensor);
    init_float_tensor((float*) C, o_shape, &o_tensor);

    auto queries   = v_tensor.dl_tensor;
    auto neighbors = i_tensor.dl_tensor;
    auto distances = o_tensor.dl_tensor;

    using queries_mdspan_type   = raft::device_matrix_view<float, int64_t, raft::row_major>;
    using neighbors_mdspan_type = raft::device_matrix_view<float, int64_t, raft::row_major>;
    using distances_mdspan_type = raft::device_matrix_view<float, int64_t, raft::row_major>;
    auto queries_mds            = cuvs::core::from_dlpack<queries_mdspan_type>(&v_tensor);
    auto neighbors_mds          = cuvs::core::from_dlpack<neighbors_mdspan_type>(&i_tensor);
    auto distances_mds          = cuvs::core::from_dlpack<distances_mdspan_type>(&o_tensor);
  
    //cuvs::distance::pairwise_distance(res, i_tensor, v_tensor, o_tensor, metric);
  
    */

   auto input_view = raft::make_device_matrix_view(M, N, L);
   auto query_view = raft::make_device_matrix_view(V, 1, L);

   //float* D;
   //init_gpu_memory((void**) &D, N);

   auto output_view = raft::make_device_matrix_view(D, N, 1);

   cuvs::distance::pairwise_distance(handle, input_view, query_view, output_view, metric);
   
   //set_sort_item<<<(N-1+THREADS_PER_BLOCK)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(D, C, p, N, probe);

   p[0] += N;

   //free_gpu_memory(D,0);
   
   /*
   float* cpu = (float*) malloc(N*sizeof(float));

   copy_memory_to_cpu(cpu, C, N*sizeof(float));
    
   std::cout << "[DEBUG]: " << cpu[0] << "," << cpu[1] << std::endl;

   free(cpu);
   */
}

void calc_squared_cosine_distances_cuvs(float* M, float* V, float* D, int* p, int N, int L, int probe) {

    auto metric = cuvs::distance::DistanceType::CosineExpanded;
    
    // Note p is not on device

    auto input_view = raft::make_device_matrix_view(M, N, L);
    auto query_view = raft::make_device_matrix_view(V, 1, L);
    auto output_view = raft::make_device_matrix_view(D+p[0], N, 1);

    cuvs::distance::pairwise_distance(handle, input_view, query_view, output_view, metric);
   
    p[0] += N;
}


#endif

struct cmp_item : public thrust::less<sort_item>
{
   __inline__
   __host__ __device__
   bool operator()(const sort_item& a, const sort_item& b) const {
      return a.distance < b.distance;
   }
};

#ifdef CUVS
void sort_item_array_nth_gpu(float* P, int* K, int N, int k) {
    
    auto input_view = raft::make_device_matrix_view(P, 1, N);
    auto out_idx_view = raft::make_device_matrix_view(K, 1, k);
    
    auto out_extents = raft::make_extents<int32_t>(input_view.extent(0), k);
    auto out_values  = raft::make_device_mdarray<float>(handle, out_extents);

    raft::matrix::select_k<float, int32_t>(handle, input_view, std::nullopt, out_values.view(), out_idx_view, true);
}

void sort_item_array_gpu(float* P, int* K, int N) {
    thrust::sort_by_key(thrust::device, P, P + N, K );
}
#else
void sort_item_array_nth_gpu(sort_item* P, int N, int k) {
    /* 
        At the time being, thrust does not have an nth element implementation. 
        Do full sort instead
    */
    thrust::sort(thrust::device, P, P + N, cmp_item() );
}

void sort_item_array_gpu(sort_item* P, int N) {
    thrust::sort(thrust::device, P, P + N, cmp_item() );
}
#endif

/*
    Squared euclidean distances with < filter
    v0: N/stridex vec per thread
*/
__global__ void calc_squared_euclidean_distances_wsfilter_v0(float* M, float* V, sort_item* C, const float f, int* p, int N, int L, int probe) {
    unsigned int indexx = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int stridex = blockDim.x*gridDim.x;
    unsigned int k;

    __shared__ float VL[2*THREADS_PER_BLOCK];
    if(threadIdx.x < L)
        VL[threadIdx.x] = V[threadIdx.x];
    if(THREADS_PER_BLOCK+threadIdx.x < L)
        VL[THREADS_PER_BLOCK+threadIdx.x] = V[THREADS_PER_BLOCK+threadIdx.x];
    
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

    __shared__ float VL[2*THREADS_PER_BLOCK];
    if(threadIdx.x < L)
        VL[threadIdx.x] = V[threadIdx.x];
    if(THREADS_PER_BLOCK+threadIdx.x < L)
        VL[THREADS_PER_BLOCK+threadIdx.x] = V[THREADS_PER_BLOCK+threadIdx.x];
    
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

    __shared__ float VL[2*THREADS_PER_BLOCK];
    if(threadIdx.x < L)
        VL[threadIdx.x] = V[threadIdx.x];
    if(THREADS_PER_BLOCK+threadIdx.x < L)
        VL[THREADS_PER_BLOCK+threadIdx.x] = V[THREADS_PER_BLOCK+threadIdx.x];

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

    __shared__ float VL[2*THREADS_PER_BLOCK];
    if(threadIdx.x < L)
        VL[threadIdx.x] = V[threadIdx.x];
    if(THREADS_PER_BLOCK+threadIdx.x < L)
        VL[THREADS_PER_BLOCK+threadIdx.x] = V[THREADS_PER_BLOCK+threadIdx.x];
    
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

    __shared__ float VL[2*THREADS_PER_BLOCK];
    if(threadIdx.x < L)
        VL[threadIdx.x] = V[threadIdx.x];
    if(THREADS_PER_BLOCK+threadIdx.x < L)
        VL[THREADS_PER_BLOCK+threadIdx.x] = V[THREADS_PER_BLOCK+threadIdx.x];

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

    __shared__ float VL[2*THREADS_PER_BLOCK];
    if(threadIdx.x < L)
        VL[threadIdx.x] = V[threadIdx.x];
    if(THREADS_PER_BLOCK+threadIdx.x < L)
        VL[THREADS_PER_BLOCK+threadIdx.x] = V[THREADS_PER_BLOCK+threadIdx.x];
    
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

__global__ void calc_squared_cosine_distances(float* M, float* V, sort_item* R, int* p, int N, int L, int probe) {
    unsigned int indexx = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int stridex = blockDim.x*gridDim.x;
    unsigned int k;
    int pos = *p;

    __shared__ float VL[2*THREADS_PER_BLOCK];
    if(threadIdx.x < L)
        VL[threadIdx.x] = V[threadIdx.x];
    if(THREADS_PER_BLOCK+threadIdx.x < L)
        VL[THREADS_PER_BLOCK+threadIdx.x] = V[THREADS_PER_BLOCK+threadIdx.x];
    
    __syncthreads();
    
    for(k=indexx; k < N; k += stridex) {
        float A = 0;
        float B = 0;
        float C = 0;

        for(int i = 0; i < L; i++) {
            A += M[L*k+i]*VL[i];
            B += M[L*k+i]*M[L*k+i];
            C += VL[i]*VL[i];
        }
        
        R[pos+k].distance = 1-A/sqrt(B)/sqrt(C);
        R[pos+k].probe = probe;
        R[pos+k].pos = k;
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
            int pos;
            calc_squared_euclidean_distances_v0d<<<NB,THREADS_PER_BLOCK>>>(M, V, C, p, N, L, probe);
            cudaMemcpy(&pos,p, sizeof(int), cudaMemcpyDeviceToHost);
            pos += N;
            cudaMemcpy(p,&pos, sizeof(int), cudaMemcpyHostToDevice); 
    }
}

/*
    Calc cosine distances
*/
void calc_squared_distances_gpu_cosine(float* M, float* V, sort_item* C, const float f, int* p, int N, int L, int probe, int op) {
    int pos;
    
    int NB = (N-1+THREADS_PER_BLOCK)/THREADS_PER_BLOCK;

    calc_squared_cosine_distances<<<NB,THREADS_PER_BLOCK>>>(M, V, C, p, N, L, probe);
    cudaMemcpy(&pos,p, sizeof(int), cudaMemcpyDeviceToHost);
    pos += N;
    cudaMemcpy(p,&pos, sizeof(int), cudaMemcpyHostToDevice); 
}