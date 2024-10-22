
extern void* init_shared_gpu_memory(int size);
extern void free_shared_gpu_memory(void* P);

extern void calc_distances_gpu_euclidean(float* M, float* V, float* C, int N, int L);

