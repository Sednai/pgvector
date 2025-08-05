#ifndef IVFGPU_H
#define IVFGPU_H

typedef unsigned short uint16;    /* == 16 bits */
typedef unsigned int uint32;    /* == 32 bits */
typedef int int32;

#include "storage/itemptr.h"

#define MAX_DATA 2097152*1

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

#ifdef GPU
extern void init_gpu();
extern void* init_gpu_memory(void** P, int size);
extern void free_gpu_memory(void* P);
extern void copy_memory_to_gpu(void* T, void* F, int size);
extern void copy_memory_to_cpu(void* T, void* F, int size);
extern void calc_squared_distances_gpu_euclidean_wfilter(float* M, float* V, sort_item* C, const float f, int* p, int N, int L, int probe, int op);
extern void sort_item_array_gpu(sort_item* P, int N);
#endif

#endif