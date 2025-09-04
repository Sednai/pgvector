#ifndef GPUWORKER_H
#define GPUWORKER_H

#include "postgres.h"
#include "storage/latch.h"
#include "postmaster/bgworker.h"
#include "ivfgpu.h"
#include "lib/ilist.h"
#include "access/tupdesc.h"
#include "storage/s_lock.h"

#define MAX_QUEUE_LENGTH 128

typedef struct worker_exec_entry
{
    dlist_node node;
    int taskid;
    Latch *notify_latch;
    bool error;
    RelFileNode nodeid;
    TupleDesc tupdesc;
    bool usegpu;
    bool usetriangle;
    int probes;
    int op;
    float filter;
    int limit;
    float* vector;
    int vec_dim;
    int returns;
    int pos;
    struct worker_exec_entry* next;
    char data[MAX_DATA];
} worker_exec_entry;

typedef struct
{
	volatile slock_t lock;
    dlist_head exec_list;
    dlist_head free_list;
    dlist_head return_list;
    Latch *latch;
    pid_t pid;
    worker_exec_entry list_data[MAX_QUEUE_LENGTH];
} worker_data_head;

worker_data_head* launch_gpuworker(void);
worker_exec_entry* get_return_slot(worker_data_head* worker, int taskid);
#ifdef __cplusplus
extern "C" void free_slot(worker_data_head* worker, worker_exec_entry* entry);
extern "C" worker_exec_entry* get_free_slot(worker_data_head* worker);
#else
void free_slot(worker_data_head* worker, worker_exec_entry* entry);
worker_exec_entry* get_free_slot(worker_data_head* worker);
#endif
void put_slot(worker_data_head* worker, worker_exec_entry* entry);

void init_shared_mem(void);
void load_index_members(RelFileNode node, BlockNumber page, TupleDesc tupdesc, int probenumber, bool use_triangle);
void load_index(RelFileNode node, TupleDesc tupdesc, bool use_triangle);

#endif