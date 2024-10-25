
extern void* init_shared_gpu_memory(int size);
extern void init_gpu_memory(void* P, int size);
extern void free_gpu_memory(void* P);
extern void prefetch_gpu_memory(void* P, int size, int device);
extern void advise_memory_readonly(void* P, int size, int device );
extern void copy_memory_to_gpu(void* T, void* F, int size);
extern void copy_memory_to_cpu(void* T, void* F, int size);
extern void copy_memory_async_to_gpu(void* T, void* F, int size);
extern void copy_memory_async_to_cpu(void* T, void* F, int size);
extern void synchronize_gpu();
extern void calc_distances_gpu_euclidean(float* M, float* V, float* C, int N, int L);

