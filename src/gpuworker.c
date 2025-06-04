
#include "postgres.h"
#include "postmaster/bgworker.h"
#include "miscadmin.h"

#include "fmgr.h"
#include "storage/latch.h"
#include "storage/spin.h"
#include "lib/ilist.h"
#include "utils/guc.h"
#include "pgstat.h"
#include "storage/bufmgr.h"

#include "ivfflat.h"
#include "gpuworker.h"
#include "gpucache.hpp"

bool got_signal = false;

static worker_data_head *worker_head = NULL;

/* shmem hook */
static shmem_request_hook_type prev_shmem_request_hook = NULL;
static void pgv_shmem_request(void);

/*
 * Init shared memory
 */
void
init_shared_mem(void)
{
	if (!process_shared_preload_libraries_in_progress)
			return;

	prev_shmem_request_hook = shmem_request_hook;
	shmem_request_hook = pgv_shmem_request;
}

/* Reserve shared memory */
static void
pgv_shmem_request(void)
{
	if (prev_shmem_request_hook)
		prev_shmem_request_hook();

	RequestAddinShmemSpace(sizeof(worker_data_head));
	RequestNamedLWLockTranche("pgv_background_worker", 1);
}

void
sigTermHandler(SIGNAL_ARGS)
{
    elog(WARNING,"pgv_gpuworker received sigterm");
	got_signal = true;
	SetLatch(MyLatch);
}

worker_data_head*
launch_gpuworker()
{	
    char buf[BGW_MAXLEN];
    snprintf(buf, BGW_MAXLEN, "pgv_gpuworker");

	/* initialize worker data header */
    bool found = false;
    
    worker_head = ShmemInitStruct(buf,
								   sizeof(worker_data_head),
								   &found);
	
    SpinLockAcquire(&worker_head->lock);

    if (found && worker_head->pid != 0) {
        SpinLockRelease(&worker_head->lock);
    	return worker_head;
    }
    
	/* initialize worker data header */
	memset(worker_head, 0, sizeof(worker_data_head));
    dlist_init(&worker_head->exec_list);
    dlist_init(&worker_head->free_list);
	dlist_init(&worker_head->return_list);
	
	// Init free list
	for(int i = 0; i < MAX_QUEUE_LENGTH; i++) {
		worker_head->list_data[i].taskid = i;
		dlist_push_tail(&worker_head->free_list,&worker_head->list_data[i].node);
	}

    BackgroundWorker worker;
    BackgroundWorkerHandle *handle;
    BgwHandleStatus status;
    pid_t		pid;
    
    memset(&worker, 0, sizeof(worker));
    worker.bgw_flags = BGWORKER_SHMEM_ACCESS | BGWORKER_BACKEND_DATABASE_CONNECTION;
    worker.bgw_start_time = BgWorkerStart_RecoveryFinished;
    worker.bgw_restart_time = BGW_NEVER_RESTART; 

    char* WORKER_LIB = "$libdir/vector.so";
    
    sprintf(worker.bgw_library_name, WORKER_LIB);
    sprintf(worker.bgw_function_name, "pgv_gpuworker_main");
    
    snprintf(worker.bgw_name, BGW_MAXLEN, "%s",buf);
         
    worker.bgw_notify_pid = MyProcPid;

    if (!RegisterDynamicBackgroundWorker(&worker, &handle))
        elog(ERROR,"Could not register background worker");

    status = WaitForBackgroundWorkerStartup(handle, &pid);

    if (status == BGWH_STOPPED)
        ereport(ERROR,
                (errcode(ERRCODE_INSUFFICIENT_RESOURCES),
                errmsg("could not start background process"),
                errhint("More details may be available in the server log.")));
    if (status == BGWH_POSTMASTER_DIED)
        ereport(ERROR,
                (errcode(ERRCODE_INSUFFICIENT_RESOURCES),
                errmsg("cannot start background processes without postmaster"),
                errhint("Kill all remaining database processes and restart the database.")));
    
    Assert(status == BGWH_STARTED);
    
    SpinLockRelease(&worker_head->lock);
	
    // Sleep a moment to wait for worker init
    pg_usleep(5000L);	

	return worker_head;
}

void load_index_members(RelFileNode node, BlockNumber page, TupleDesc tupdesc, int probenumber, bool use_triangle) {
    IndexTuple	itup;  
    bool isnull;
    OffsetNumber offno;
    OffsetNumber maxoffno;
    Buffer cbuf;
    Page cpage;

    while (BlockNumberIsValid(page))
    {
        cbuf = ReadBufferWithoutRelcache(node, MAIN_FORKNUM, page, RBM_NORMAL, NULL, true);
        LockBuffer(cbuf, BUFFER_LOCK_SHARE);
        cpage = BufferGetPage(cbuf);
        maxoffno = PageGetMaxOffsetNumber(cpage);
     
        for (offno = FirstOffsetNumber; offno <= maxoffno; offno = OffsetNumberNext(offno)) {
            itup = (IndexTuple) PageGetItem(cpage, PageGetItemId(cpage, offno));
            
            Vector *v = PointerGetDatum( index_getattr(itup, 1, tupdesc, &isnull) );
           
            // Store
            if(!use_triangle) {
                insert(node, probenumber, v, (int) page, itup->t_tid);
            } else {
            
                double dist = DatumGetFloat8( index_getattr(itup, 2, tupdesc, &isnull) );
                
                insert_wdistance(node, probenumber, v, (float) dist, (int) page, itup->t_tid);
            }
        }

        page = IvfflatPageGetOpaque(cpage)->nextblkno;
        
        UnlockReleaseBuffer(cbuf);
    }
}

void load_index(RelFileNode node, TupleDesc tupdesc, bool use_triangle ) {

    // Not found in cache -> Load data
    if(!incache(node)) {
    //if(true) {  
        BlockNumber nextblkno = IVFFLAT_HEAD_BLKNO;
	    Buffer cbuf;

        while (BlockNumberIsValid(nextblkno))
	    {
            cbuf = ReadBufferWithoutRelcache(node, MAIN_FORKNUM, nextblkno, RBM_NORMAL, NULL, true);
            LockBuffer(cbuf, BUFFER_LOCK_SHARE);
            Page cpage = BufferGetPage(cbuf);
            OffsetNumber maxoffno = PageGetMaxOffsetNumber(cpage);
            OffsetNumber offno;

            for (offno = FirstOffsetNumber; offno <= maxoffno; offno = OffsetNumberNext(offno))
		    {
                IvfflatList list = (IvfflatList) PageGetItem(cpage, PageGetItemId(cpage, offno));
                
                Vector *c = PointerGetDatum(&list->center);

                // Store as new probe
                int pn = new_probe(node, c, use_triangle); 

                BlockNumber spage = list->startPage;
                
                load_index_members(node, spage, tupdesc, pn, use_triangle);
            }

            nextblkno = IvfflatPageGetOpaque(cpage)->nextblkno;
            
            UnlockReleaseBuffer(cbuf);
        }
    }
}


void
pgv_gpuworker_main(Datum main_arg)
{
    
	char buf[BGW_MAXLEN];
	snprintf(buf, BGW_MAXLEN, "%s", MyBgworkerEntry->bgw_name); 

	// Attach to shared memory
	bool found;
	worker_head = ShmemInitStruct(MyBgworkerEntry->bgw_name,
								   sizeof(worker_data_head),
								   &found);
	if(!found) {
		elog(ERROR,"Shared memory for background worker has not been initialized");
	}
	
	SpinLockAcquire(&worker_head->lock); 
	worker_head->latch = MyLatch;
    worker_head->pid = MyProcPid;
    SpinLockRelease(&worker_head->lock);


    /* Need to be able to look into catalogs */
	CurrentResourceOwner = ResourceOwnerCreate(NULL, "ForPGVbackgroundWorker");

	/* Establish signal handlers before unblocking signals. */
	pqsignal(SIGTERM, sigTermHandler);
	
	/* We're now ready to receive signals */
	BackgroundWorkerUnblockSignals();
		
    //elog(WARNING,"[DEBUG] -> pid: %d",MyProcPid);
	//sleep(60);

    /*
	 * Main loop: do this until SIGTERM is received and processed by
	 * ProcessInterrupts.
	 */
	while(!got_signal)
	{
		int			ret;

        SpinLockAcquire(&worker_head->lock);
       
        if (dlist_is_empty(&worker_head->exec_list))
        {
            SpinLockRelease(&worker_head->lock);
		    int ev = WaitLatch(MyLatch,
                            WL_LATCH_SET | WL_TIMEOUT | WL_POSTMASTER_DEATH,
                            10 * 1000L,
                            PG_WAIT_EXTENSION);
            ResetLatch(MyLatch);
		    if (ev & WL_POSTMASTER_DEATH)
                elog(FATAL, "unexpected postmaster dead");
            
            CHECK_FOR_INTERRUPTS();
            continue;
        }
        
        /*
            Exec task
        */       
        dlist_node* dnode = dlist_pop_head_node(&worker_head->exec_list);
        worker_exec_entry* entry = dlist_container(worker_exec_entry, node, dnode);

     	SpinLockRelease(&worker_head->lock);

        load_index(entry->nodeid, entry->tupdesc, entry->usetriangle);

        // Compute
        if(!entry->usegpu) {
            entry->returns = exec_query_cpu(entry, worker_head);
        }
        else
#ifdef GPU
            entry->returns = exec_query_gpu(entry, worker_head);
#else
            entry->returns = exec_query_cpu(entry, worker_head);
#endif
        entry->pos = 0;

        // Return
        // ToDo: CUT into pieces if too long ...
        
        SpinLockAcquire(&worker_head->lock);
		dlist_push_tail(&worker_head->return_list,&entry->node);
  		SpinLockRelease(&worker_head->lock);
	
		/*
			Cleanup
		*/
	
		SetLatch( entry->notify_latch );
	}

    /* Release everything */
	ResourceOwnerRelease(CurrentResourceOwner, RESOURCE_RELEASE_BEFORE_LOCKS, true, true);
	ResourceOwnerRelease(CurrentResourceOwner, RESOURCE_RELEASE_LOCKS, true, true);
	ResourceOwnerRelease(CurrentResourceOwner, RESOURCE_RELEASE_AFTER_LOCKS, true, true);
	CurrentResourceOwner = NULL;

    elog(WARNING, "SIG RECEIVED");	
}

worker_exec_entry* get_free_slot(worker_data_head* worker) {
    worker_exec_entry* entry = NULL;
   
    SpinLockAcquire(&worker->lock);
    
    if(!dlist_is_empty(&worker->free_list)) {
        dlist_node* dnode = dlist_pop_head_node(&worker->free_list);
        entry = dlist_container(worker_exec_entry, node, dnode);
    } 

    SpinLockRelease(&worker->lock);
  
    return entry;
}

void put_slot(worker_data_head* worker, worker_exec_entry* entry) {
    SpinLockAcquire(&worker->lock);
  
    dlist_push_tail(&worker->exec_list,&entry->node);			
    SetLatch( worker->latch );
			
    SpinLockRelease(&worker->lock);
}

worker_exec_entry* get_return_slot(worker_data_head* worker, int taskid) {

    dlist_iter    iter;
    bool got_signal = false;
			
    while(!got_signal)
    {
        SpinLockAcquire(&worker->lock);
    
        if (dlist_is_empty(&worker->return_list))
        {
            SpinLockRelease(&worker->lock);
            int ev = WaitLatch(MyLatch,
                            WL_LATCH_SET | WL_TIMEOUT | WL_POSTMASTER_DEATH,
                            1 * 1000L,
                            PG_WAIT_EXTENSION);
            ResetLatch(MyLatch);
            if (ev & WL_POSTMASTER_DEATH)
                elog(FATAL, "unexpected postmaster dead");
            
            CHECK_FOR_INTERRUPTS();
            continue;
        }

        worker_exec_entry* ret;
        dlist_foreach(iter, &worker->return_list) {
            ret = dlist_container(worker_exec_entry, node, iter.cur);

            if(ret->taskid == taskid) {
                got_signal = true;
                dlist_delete(iter.cur);
                break;
            }
        }

        SpinLockRelease(&worker->lock);           
    
        if(got_signal) {
           return ret;
        }
    }

    return NULL;
}

void free_slot(worker_data_head* worker, worker_exec_entry* entry) {
    SpinLockAcquire(&worker->lock);
    dlist_push_tail(&worker->free_list,entry);           
    SpinLockRelease(&worker->lock);          
}