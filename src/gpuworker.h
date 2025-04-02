#ifndef GPUWORKER_H
#define GPUWORKER_H

#include "postgres.h"
#include "storage/latch.h"
#include "postmaster/bgworker.h"
#include "ivfgpu.h"

#define MAX_QUEUE_LENGTH 32

typedef struct worker_exec_entry
{
    dlist_node node;
    int taskid;
    Latch *notify_latch;
    bool error;
    RelFileNode nodeid;
    TupleDesc tupdesc;
    bool usegpu;
    int probes;
    int op;
    float filter;
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

worker_data_head* launch_gpuworker();
worker_exec_entry* get_free_slot(worker_data_head* worker);
worker_exec_entry* get_return_slot(worker_data_head* worker, int taskid);
void put_slot(worker_data_head* worker, worker_exec_entry* entry);
void free_slot(worker_data_head* worker, worker_exec_entry* entry);

void init_shared_mem(void);
void load_index_members(RelFileNode node, BlockNumber page, TupleDesc tupdesc, int probenumber);
void load_index(RelFileNode node, TupleDesc tupdesc );

#endif