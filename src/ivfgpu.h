
typedef unsigned short uint16;    /* == 16 bits */
typedef unsigned int uint32;    /* == 32 bits */
typedef int int32;

#include "storage/itemptr.h"

typedef struct page_item {
	float distance;
	ItemPointerData ipd;
	int searchPage;
} page_item;

typedef struct sort_item {
	float distance;
	int probe;
	int pos;
} sort_item;

typedef struct page_list {
	long length;
	long max_length;
	int pos;
	page_item* data;
} page_list;

static int compare_pi(const void* a, const void* b) {
	
	const page_item *elem1 = (page_item*) a;    
    const page_item *elem2 = (page_item*) b;

   if (elem1->distance < elem2->distance)
      return -1;
   else if (elem1->distance > elem2->distance)
      return 1;
   else
      return 0;
}


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
extern "C" void calc_squared_distances_gpu_euclidean(float* M, float* V, float* C, int N, int L);
extern "C" void calc_squared_distances_gpu_euclidean_mod(float* M, float* V, float V2s, float* C, int N, int L);    
extern "C" void calc_squared_distances_gpu_euclidean_nosharedmem(float* M, float* V, float* C, int N, int L);
extern "C" void calc_squared_distances_gpu_euclidean_nosharedmem_test(float* M, float* V, float* C, int N, int L);
extern "C" void calc_squared_distances_gpu_euclidean_wsfilter(float* M, float* V, sort_item* C, const float f, int* p, int N, int L, int probe);
extern "C" void sort_array_gpu(page_item* P, int N);
extern "C" void sort_item_array_gpu(sort_item* P, int N);
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
extern void calc_squared_distances_gpu_euclidean(float* M, float* V, float* C, int N, int L);
extern void calc_squared_distances_gpu_euclidean_mod(float* M, float* V, float V2s, float* C, int N, int L);
extern void calc_squared_distances_gpu_euclidean_nosharedmem(float* M, float* V, float* C, int N, int L);
extern void calc_squared_distances_gpu_euclidean_nosharedmem_test(float* M, float* V, float* C, int N, int L);
extern void calc_squared_distances_gpu_euclidean_wsfilter(float* M, float* V, sort_item* C, const float f, int* p, int N, int L, int probe);
extern void sort_array_gpu(page_item* P, int N);
extern void sort_item_array_gpu(sort_item* P, int N);

#endif