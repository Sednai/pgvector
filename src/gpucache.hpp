#include "storage/relfilenode.h"
#include "gpuworker.h"

struct item {
	int page;
    ItemPointerData ipd;
};

#ifdef __cplusplus
extern "C" bool incache(RelFileNode node);
extern "C" int new_probe(RelFileNode node, Vector* c, bool use_triangle);
extern "C" void insert(RelFileNode node, int probenumber, Vector* c, int page, ItemPointerData ipd);
extern "C" void insert_wdistance(RelFileNode node, int probenumber, Vector* c, float distance, int page, ItemPointerData ipd);
extern "C" void logsize();
extern "C" int exec_query_cpu(worker_exec_entry* entry, worker_data_head* worker);
extern "C" int exec_query_gpu(worker_exec_entry* entry, worker_data_head* worker);
#else
extern bool incache(RelFileNode node);
extern int new_probe(RelFileNode node, Vector* c, bool use_triangle);
extern int exec_query_cpu(worker_exec_entry* entry, worker_data_head* worker);
extern int exec_query_gpu(worker_exec_entry* entry, worker_data_head* worker);
extern void insert(RelFileNode node, int probenumber, Vector* c, int page, ItemPointerData ipd);
extern void insert_wdistance(RelFileNode node, int probenumber, Vector* c, float distance, int page, ItemPointerData ipd);
extern void logsize();
#endif