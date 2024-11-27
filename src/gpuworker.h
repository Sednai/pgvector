#include "postgres.h"
#include "storage/latch.h"
#include "postmaster/bgworker.h"

#define MAX_QUEUE_LENGTH 16

#define MAX_DATA 2097152*1
typedef struct 
{
    dlist_node node;
    int taskid;
    Latch *notify_latch;
    bool error;
    RelFileNode nodeid;
    TupleDesc tupdesc;
    int probes;
    int op;
    float filter;
    float* vector;
    int vec_dim;
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



worker_data_head* launch_dynamic_worker();
