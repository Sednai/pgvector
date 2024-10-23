
extern void* init_shared_gpu_memory(int size);
extern void free_shared_gpu_memory(void* P);
extern void prefetch_gpu_memory(void* P, int size, int device);
extern void advise_memory_readonly(void* P, int size, int device );

extern void calc_distances_gpu_euclidean(float* M, float* V, float* C, int N, int L);

