
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
#include "gpucache.h"

bool got_signal = false;

static worker_data_head *worker_head = NULL;


void
sigTermHandler(SIGNAL_ARGS)
{
    elog(WARNING,"pgv_gpuworker received sigterm");
	got_signal = true;
	SetLatch(MyLatch);
}


worker_data_head*
launch_dynamic_worker()
{	
    char buf[BGW_MAXLEN];
    snprintf(buf, BGW_MAXLEN, "pgv_gpuworker");


	/* initialize worker data header */
    bool found = false;
    worker_head = ShmemInitStruct(buf,
								   sizeof(worker_data_head),
								   &found);
	
    SpinLockAcquire(&worker_head->lock);

    if (found) {
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
    worker.bgw_restart_time = BGW_NEVER_RESTART; // Time in s to restart if crash. Use BGW_NEVER_RESTART for no restart;
    
    char* WORKER_LIB = GetConfigOption("ivfflat.lib",true,true);
    
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

void load_index_members(RelFileNode node, BlockNumber page, TupleDesc tupdesc, int probenumber) {
    IndexTuple	itup;  
    bool isnull;
    OffsetNumber offno;
    OffsetNumber maxoffno;
    Buffer cbuf;
    Page cpage;

    while (BlockNumberIsValid(page))
    {
        cbuf = ReadBufferWithoutRelcache(node,0,page,0,NULL);
        LockBuffer(cbuf, BUFFER_LOCK_SHARE);
        cpage = BufferGetPage(cbuf);
        maxoffno = PageGetMaxOffsetNumber(cpage);
     
        for (offno = FirstOffsetNumber; offno <= maxoffno; offno = OffsetNumberNext(offno)) {
            itup = (IndexTuple) PageGetItem(cpage, PageGetItemId(cpage, offno));
            
            Vector *v = PointerGetDatum( index_getattr(itup, 1, tupdesc, &isnull) );
            
            // Store
            insert(node, probenumber, v, (int) page, itup->t_tid);
        }

        page = IvfflatPageGetOpaque(cpage)->nextblkno;
        
        UnlockReleaseBuffer(cbuf);
    }
}

void load_index(RelFileNode node, TupleDesc tupdesc ) {

    // Not found in cache -> Load data
    if(!incache(node)) {
        
        BlockNumber nextblkno = IVFFLAT_HEAD_BLKNO;
	    Buffer cbuf;

        while (BlockNumberIsValid(nextblkno))
	    {
        
            cbuf = ReadBufferWithoutRelcache(node,0,nextblkno,0,NULL);
            LockBuffer(cbuf, BUFFER_LOCK_SHARE);
            Page cpage = BufferGetPage(cbuf);
            OffsetNumber maxoffno = PageGetMaxOffsetNumber(cpage);
            OffsetNumber offno;

            for (offno = FirstOffsetNumber; offno <= maxoffno; offno = OffsetNumberNext(offno))
		    {
                IvfflatList list = (IvfflatList) PageGetItem(cpage, PageGetItemId(cpage, offno));
                
                Vector *c = PointerGetDatum(&list->center);

                // Store as new probe
                int pn = new_probe(node, c); 

                BlockNumber spage = list->startPage;
                
                load_index_members(node, spage, tupdesc, pn);
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
    SpinLockRelease(&worker_head->lock);

	/* Establish signal handlers before unblocking signals. */
	pqsignal(SIGTERM, sigTermHandler);
	
	/* We're now ready to receive signals */
	BackgroundWorkerUnblockSignals();
		
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

        load_index(entry->nodeid, entry->tupdesc);

        logsize();
    
        // Compute
        //page_list L = exec_query_cpu(entry->nodeid, entry->probes, entry->op, entry->filter, entry->vector, entry->vec_dim);
        page_list L = exec_query_gpu(entry->nodeid, entry->probes, entry->op, entry->filter, entry->vector, entry->vec_dim);

        //elog(WARNING,"[DEBUG]: items before %ld",L.length);

        // Return
        // ToDo: CUT into pieces if too long ...
        entry->probes = (int) L.length;
      
        memcpy(entry->data,L.data,L.length*sizeof(page_item));
      
        free(L.data);
        
	    SpinLockAcquire(&worker_head->lock);
		dlist_push_tail(&worker_head->return_list,&entry->node);
  		SpinLockRelease(&worker_head->lock);
	
		/*
			Cleanup
		*/
	
		SetLatch( entry->notify_latch );
	}

    elog(WARNING, "SIG RECEIVED");	
}
