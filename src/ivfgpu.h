
typedef unsigned short uint16;    /* == 16 bits */
typedef unsigned int uint32;    /* == 32 bits */
typedef int int32;

#include "storage/itemptr.h"

typedef struct page_item {
	float distance;
	ItemPointerData ipd;
	int searchPage;
} page_item;


#ifdef __cplusplus
extern "C" void* init_shared_gpu_memory(int size);
extern "C" void init_gpu_memory(void** P, int size);
extern "C" void free_gpu_memory(void* P);
extern "C" void prefetch_gpu_memory(void* P, int size, int device);
extern "C" void advise_memory_readonly(void* P, int size, int device );
extern "C" void copy_memory_to_gpu(void* T, void* F, int size);
extern "C" void copy_memory_to_cpu(void* T, void* F, int size);
extern "C" void copy_memory_async_to_gpu(void* T, void* F, int size);
extern "C" void copy_memory_async_to_cpu(void* T, void* F, int size);
extern "C" void synchronize_gpu();
extern "C" void calc_distances_gpu_euclidean(float* M, float* V, float* C, int N, int L);
extern "C" void sort_array(page_item* P, int N);
#else
extern void* init_shared_gpu_memory(int size);
extern void init_gpu_memory(void** P, int size);
extern void free_gpu_memory(void* P);
extern void prefetch_gpu_memory(void* P, int size, int device);
extern void advise_memory_readonly(void* P, int size, int device );
extern void copy_memory_to_gpu(void* T, void* F, int size);
extern void copy_memory_to_cpu(void* T, void* F, int size);
extern void copy_memory_async_to_gpu(void* T, void* F, int size);
extern void copy_memory_async_to_cpu(void* T, void* F, int size);
extern void synchronize_gpu();
extern void calc_distances_gpu_euclidean(float* M, float* V, float* C, int N, int L);
extern void sort_array(page_item* P, int N);

#endif