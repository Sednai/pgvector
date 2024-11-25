#include "storage/relfilenode.h"

struct item {
	int page;
    ItemPointerData ipd;
};

#ifdef __cplusplus
extern "C" bool incache(RelFileNode node);
extern "C" int new_probe(RelFileNode node, Vector* c);
extern "C" void insert(RelFileNode node, int probenumber, Vector* c, int page, ItemPointerData ipd);
extern "C" void logsize();
extern "C" page_list exec_query_cpu(RelFileNode node, int Np, int op, float filter, float* q, int dim);
extern "C" page_list exec_query_gpu(RelFileNode node, int Np, int op, float filter, float* q, int dim);
#else
extern bool incache(RelFileNode node);
extern int new_probe(RelFileNode node, Vector* c);
extern page_list exec_query_cpu(RelFileNode node, int Np, int op, float filter, float* q, int dim);
extern page_list exec_query_gpu(RelFileNode node, int Np, int op, float filter, float* q, int dim);
extern void insert(RelFileNode node, int probenumber, Vector* c, int page, ItemPointerData ipd);
extern void logsize();
#endif