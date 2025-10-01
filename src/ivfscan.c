#include "postgres.h"

#include <float.h>

#include "access/relscan.h"
#include "catalog/pg_operator_d.h"
#include "catalog/pg_type_d.h"
#include "lib/pairingheap.h"
#include "ivfflat.h"
#include "miscadmin.h"
#include "pgstat.h"
#include "storage/bufmgr.h"
#include "utils/memutils.h"
#ifdef AERO
#include "executor/executor.h"
#include "gpuworker.h"
#endif
#define GetScanList(ptr) pairingheap_container(IvfflatScanList, ph_node, ptr)
#define GetScanListConst(ptr) pairingheap_const_container(IvfflatScanList, ph_node, ptr)

#ifdef AERO
worker_data_head* worker = NULL;
worker_exec_entry* ret = NULL;
#endif

/*
 * Compare list distances
 */
static int
CompareLists(const pairingheap_node *a, const pairingheap_node *b, void *arg)
{
	if (GetScanListConst(a)->distance > GetScanListConst(b)->distance)
		return 1;

	if (GetScanListConst(a)->distance < GetScanListConst(b)->distance)
		return -1;

	return 0;
}

/*
 * Get lists and sort by distance
 */
static void
GetScanLists(IndexScanDesc scan, Datum value)
{
	IvfflatScanOpaque so = (IvfflatScanOpaque) scan->opaque;
	BlockNumber nextblkno = IVFFLAT_HEAD_BLKNO;
	int			listCount = 0;
	double		maxDistance = DBL_MAX;

	/* Search all list pages */
	while (BlockNumberIsValid(nextblkno))
	{
		Buffer		cbuf;
		Page		cpage;
		OffsetNumber maxoffno;

		cbuf = ReadBuffer(scan->indexRelation, nextblkno);
		LockBuffer(cbuf, BUFFER_LOCK_SHARE);
		cpage = BufferGetPage(cbuf);

		maxoffno = PageGetMaxOffsetNumber(cpage);

		for (OffsetNumber offno = FirstOffsetNumber; offno <= maxoffno; offno = OffsetNumberNext(offno))
		{
			IvfflatList list = (IvfflatList) PageGetItem(cpage, PageGetItemId(cpage, offno));
			double		distance;

			/* Use procinfo from the index instead of scan key for performance */
			distance = DatumGetFloat8(so->distfunc(so->procinfo, so->collation, PointerGetDatum(&list->center), value));

			if (listCount < so->maxProbes)
			{
				IvfflatScanList *scanlist;

				scanlist = &so->lists[listCount];
				scanlist->startPage = list->startPage;
				scanlist->distance = distance;
				listCount++;

				/* Add to heap */
				pairingheap_add(so->listQueue, &scanlist->ph_node);

				/* Calculate max distance */
				if (listCount == so->maxProbes)
					maxDistance = GetScanList(pairingheap_first(so->listQueue))->distance;
			}
			else if (distance < maxDistance)
			{
				IvfflatScanList *scanlist;

				/* Remove */
				scanlist = GetScanList(pairingheap_remove_first(so->listQueue));

				/* Reuse */
				scanlist->startPage = list->startPage;
				scanlist->distance = distance;
				pairingheap_add(so->listQueue, &scanlist->ph_node);

				/* Update max distance */
				maxDistance = GetScanList(pairingheap_first(so->listQueue))->distance;
			}
		}

		nextblkno = IvfflatPageGetOpaque(cpage)->nextblkno;

		UnlockReleaseBuffer(cbuf);
	}

	for (int i = listCount - 1; i >= 0; i--)
		so->listPages[i] = GetScanList(pairingheap_remove_first(so->listQueue))->startPage;

	Assert(pairingheap_is_empty(so->listQueue));
}

/*
 * Get items
 */
static void
GetScanItems(IndexScanDesc scan, Datum value)
{
	IvfflatScanOpaque so = (IvfflatScanOpaque) scan->opaque;
	TupleDesc	tupdesc = RelationGetDescr(scan->indexRelation);
	TupleTableSlot *slot = so->vslot;
	int			batchProbes = 0;

	tuplesort_reset(so->sortstate);

	/* Search closest probes lists */
	while (so->listIndex < so->maxProbes && (++batchProbes) <= so->probes)
	{
		BlockNumber searchPage = so->listPages[so->listIndex++];

		/* Search all entry pages for list */
		while (BlockNumberIsValid(searchPage))
		{
			Buffer		buf;
			Page		page;
			OffsetNumber maxoffno;

			buf = ReadBufferExtended(scan->indexRelation, MAIN_FORKNUM, searchPage, RBM_NORMAL, so->bas);
			LockBuffer(buf, BUFFER_LOCK_SHARE);
			page = BufferGetPage(buf);
			maxoffno = PageGetMaxOffsetNumber(page);

			for (OffsetNumber offno = FirstOffsetNumber; offno <= maxoffno; offno = OffsetNumberNext(offno))
			{
				IndexTuple	itup;
				Datum		datum;
				bool		isnull;
				ItemId		itemid = PageGetItemId(page, offno);

				itup = (IndexTuple) PageGetItem(page, itemid);
				datum = index_getattr(itup, 1, tupdesc, &isnull);

				/*
				 * Add virtual tuple
				 *
				 * Use procinfo from the index instead of scan key for
				 * performance
				 */
				ExecClearTuple(slot);
				slot->tts_values[0] = so->distfunc(so->procinfo, so->collation, datum, value);
				slot->tts_isnull[0] = false;
				slot->tts_values[1] = PointerGetDatum(&itup->t_tid);
				slot->tts_isnull[1] = false;
				ExecStoreVirtualTuple(slot);

				tuplesort_puttupleslot(so->sortstate, slot);
			}

			searchPage = IvfflatPageGetOpaque(page)->nextblkno;

			UnlockReleaseBuffer(buf);
		}
	}

	tuplesort_performsort(so->sortstate);

#if defined(IVFFLAT_MEMORY)
	elog(INFO, "memory: %zu MB", MemoryContextMemAllocated(CurrentMemoryContext, true) / (1024 * 1024));
#endif
}

/*
 * Zero distance
 */
static Datum
ZeroDistance(FmgrInfo *flinfo, Oid collation, Datum arg1, Datum arg2)
{
	return Float8GetDatum(0.0);
}

/*
 * Get scan value
 */
static Datum
GetScanValue(IndexScanDesc scan)
{
	IvfflatScanOpaque so = (IvfflatScanOpaque) scan->opaque;
	Datum		value;

	if (scan->orderByData->sk_flags & SK_ISNULL)
	{
		value = PointerGetDatum(NULL);
		so->distfunc = ZeroDistance;
	}
	else
	{
		value = scan->orderByData->sk_argument;
		so->distfunc = FunctionCall2Coll;

		/* Value should not be compressed or toasted */
		Assert(!VARATT_IS_COMPRESSED(DatumGetPointer(value)));
		Assert(!VARATT_IS_EXTENDED(DatumGetPointer(value)));

		/* Normalize if needed */
		if (so->normprocinfo != NULL)
		{
			MemoryContext oldCtx = MemoryContextSwitchTo(so->tmpCtx);

			value = IvfflatNormValue(so->typeInfo, so->collation, value);

			MemoryContextSwitchTo(oldCtx);
		}
	}

	return value;
}

/*
 * Initialize scan sort state
 */
static Tuplesortstate *
InitScanSortState(TupleDesc tupdesc)
{
	AttrNumber	attNums[] = {1};
	Oid			sortOperators[] = {Float8LessOperator};
	Oid			sortCollations[] = {InvalidOid};
	bool		nullsFirstFlags[] = {false};

	return tuplesort_begin_heap(tupdesc, 1, attNums, sortOperators, sortCollations, nullsFirstFlags, work_mem, NULL, false);
}

/*
 * Prepare for an index scan
 */
IndexScanDesc
ivfflatbeginscan(Relation index, int nkeys, int norderbys)
{
	IndexScanDesc scan;
	IvfflatScanOpaque so;
	int			lists;
	int			dimensions;
	int			probes = ivfflat_probes;
	int			maxProbes;
	MemoryContext oldCtx;

	scan = RelationGetIndexScan(index, nkeys, norderbys);

	/* Get lists and dimensions from metapage */
	IvfflatGetMetaPageInfo(index, &lists, &dimensions);

	if (ivfflat_iterative_scan != IVFFLAT_ITERATIVE_SCAN_OFF)
		maxProbes = Max(ivfflat_max_probes, probes);
	else
		maxProbes = probes;

	if (probes > lists)
		probes = lists;

	if (maxProbes > lists)
		maxProbes = lists;

	so = (IvfflatScanOpaque) palloc(sizeof(IvfflatScanOpaqueData));
	so->typeInfo = IvfflatGetTypeInfo(index);
	so->first = true;
	so->probes = probes;
	so->maxProbes = maxProbes;
	so->dimensions = dimensions;

	/* Set support functions */
	so->procinfo = index_getprocinfo(index, 1, IVFFLAT_DISTANCE_PROC);
	so->normprocinfo = IvfflatOptionalProcInfo(index, IVFFLAT_NORM_PROC);
	so->collation = index->rd_indcollation[0];

	so->tmpCtx = AllocSetContextCreate(CurrentMemoryContext,
									   "Ivfflat scan temporary context",
									   ALLOCSET_DEFAULT_SIZES);

	oldCtx = MemoryContextSwitchTo(so->tmpCtx);

	/* Create tuple description for sorting */
	so->tupdesc = CreateTemplateTupleDesc(2);
	TupleDescInitEntry(so->tupdesc, (AttrNumber) 1, "distance", FLOAT8OID, -1, 0);
	TupleDescInitEntry(so->tupdesc, (AttrNumber) 2, "heaptid", TIDOID, -1, 0);

	/* Prep sort */
	so->sortstate = InitScanSortState(so->tupdesc);

	/* Need separate slots for puttuple and gettuple */
	so->vslot = MakeSingleTupleTableSlot(so->tupdesc, &TTSOpsVirtual);
	so->mslot = MakeSingleTupleTableSlot(so->tupdesc, &TTSOpsMinimalTuple);

	/*
	 * Reuse same set of shared buffers for scan
	 *
	 * See postgres/src/backend/storage/buffer/README for description
	 */
	so->bas = GetAccessStrategy(BAS_BULKREAD);

	so->listQueue = pairingheap_allocate(CompareLists, scan);
	so->listPages = palloc(maxProbes * sizeof(BlockNumber));
	so->listIndex = 0;
	so->lists = palloc(maxProbes * sizeof(IvfflatScanList));

	MemoryContextSwitchTo(oldCtx);

	scan->opaque = so;

	return scan;
}

/*
 * Start or restart an index scan
 */
void
ivfflatrescan(IndexScanDesc scan, ScanKey keys, int nkeys, ScanKey orderbys, int norderbys)
{
	IvfflatScanOpaque so = (IvfflatScanOpaque) scan->opaque;

	so->first = true;
	pairingheap_reset(so->listQueue);
	so->listIndex = 0;

	if (keys && scan->numberOfKeys > 0)
		memmove(scan->keyData, keys, scan->numberOfKeys * sizeof(ScanKeyData));

	if (orderbys && scan->numberOfOrderBys > 0)
		memmove(scan->orderByData, orderbys, scan->numberOfOrderBys * sizeof(ScanKeyData));
}

/*
 * Fetch the next tuple in the given scan
 */
bool
ivfflatgettuple(IndexScanDesc scan, ScanDirection dir)
{
	IvfflatScanOpaque so = (IvfflatScanOpaque) scan->opaque;
	ItemPointer heaptid;
	bool		isnull;

	/*
	 * Index can be used to scan backward, but Postgres doesn't support
	 * backward scan on operators
	 */
	Assert(ScanDirectionIsForward(dir));

	if (so->first)
	{
		Datum		value;

#ifdef AERO
		/* 
		 * Start background worker if not started yet
		 */
		if (ivfflat_bgw) {
			/* GPU background worker init */
			worker =  launch_gpuworker();
		}
#endif

		/* Count index scan for stats */
		pgstat_count_index_scan(scan->indexRelation);

		/* Safety check */
		if (scan->orderByData == NULL)
			elog(ERROR, "cannot scan ivfflat index without order");

		/* Requires MVCC-compliant snapshot as not able to pin during sorting */
		/* https://www.postgresql.org/docs/current/index-locking.html */
		if (!IsMVCCSnapshot(scan->xs_snapshot))
			elog(ERROR, "non-MVCC snapshots are not supported with ivfflat");

#ifdef AERO
	if(ivfflat_bgw) {
		char* pos;
		Vector* v;
		TupleDesc desc;

		/*
		 * Start job on background worker and wait for return
		 */
		worker_exec_entry* entry = get_free_slot(worker);
		if(entry == NULL)
			elog(ERROR,"No free slot available for pg_vector background worker");
			
		// Check for special operator
		if(scan->orderByData->sk_strategy == 2) {
			HeapTupleHeader t = DatumGetHeapTupleHeader(DatumGetPointer(scan->orderByData->sk_argument));
			bool isnull;	
			value = GetAttributeByNum(t, 1, &isnull);
			if(isnull)
				elog(ERROR,"Vector in advanced type can not be null !");	
		
			entry->op = DatumGetInt32( GetAttributeByNum(t, 2, &isnull) ); 
			entry->filter = DatumGetFloat4( GetAttributeByNum(t, 3, &isnull) );
			entry->filter = entry->filter*entry->filter; // Squared because we use squared distance for gpu functions
			entry->limit = DatumGetInt32( GetAttributeByNum(t, 4, &isnull) ); 
			entry->distfunc = 0;
		} else {
			char* mod;
			char* fn;

			value = GetScanValue(scan);
			entry->op = -100;
			entry->filter = 0;
			entry->limit = -1;

			// Infer dist op
			mod = palloc(128);
			fn = palloc(128);
			
			fmgr_symbol(scan->orderByData->sk_func.fn_oid, &mod, &fn);
			if(strcmp(fn,"l2_distance") == 0)
				entry->distfunc = 0;
			else if(strcmp(fn,"cosine_distance") == 0) {
				entry->distfunc = 1;
				if(ivfflat_triangle)
					elog(ERROR,"Triangle relation not supported for cosine distance");
			}
			else {
				pfree(mod);
				pfree(fn);
				elog(ERROR,"Distance function %s not supported yet in background worker",fn);
			}

			pfree(mod);
			pfree(fn);		
		}

		// Set job data
		entry->notify_latch = MyLatch;
		entry->nodeid = scan->indexRelation->rd_node;
		entry->probes = so->probes;
		entry->usegpu = ivfflat_gpu;
		entry->usetriangle = ivfflat_triangle;
		pos = entry->data;

		// Copy vec to data
		v = DatumGetVector(value);
		entry->vec_dim = v->dim;
		memcpy(pos, v->x, v->dim*sizeof(float));
		entry->vector = (float*) pos;
		pos += v->dim*sizeof(float);

		// ToDo: Read TupDesc directly from relation in gpuworker !!!

		// Copy tupledesc to data	
		entry->tupdesc = (TupleDesc) pos;

		desc = CreateTemplateTupleDesc(2);
		
		TupleDescCopyEntry(desc, (AttrNumber) 1, scan->indexRelation->rd_att, (AttrNumber) 1);
		TupleDescInitEntry(desc, (AttrNumber) 2, "distance", FLOAT8OID, -1, 0);

		memcpy(pos, desc, sizeof(*desc) + desc->natts*sizeof(FormData_pg_attribute) );		
		pos += sizeof(*desc) + desc->natts*sizeof(FormData_pg_attribute);
	
		//memcpy(pos,scan->indexRelation->rd_att, sizeof(*scan->indexRelation->rd_att) + scan->indexRelation->rd_att->natts*sizeof(FormData_pg_attribute) );		
		//pos += sizeof(*scan->indexRelation->rd_att) + scan->indexRelation->rd_att->natts*sizeof(FormData_pg_attribute);
	
		put_slot(worker, entry);

		// Get result
		ret = get_return_slot(worker,entry->taskid);

		if(ret->returns == -1) {
			elog(ERROR,"Too many returns. Increase MAX_DATA in ivfgpu.h");
		}	

	} else {
		if(scan->orderByData->sk_strategy == 2) 
			elog(ERROR,"<!> operator only supported with background worker. Set ivfflat.bgw = 1.");
#endif
		value = GetScanValue(scan);

		IvfflatBench("GetScanLists", GetScanLists(scan, value));
		IvfflatBench("GetScanItems", GetScanItems(scan, value));
#ifdef AERO
	}
#endif
		so->first = false;
		so->value = value;
	}

#ifdef AERO
	if(!ivfflat_bgw) {
	while (!tuplesort_gettupleslot(so->sortstate, true, false, so->mslot, NULL))
	{
		if (so->listIndex == so->maxProbes)
			return false;

		IvfflatBench("GetScanItems", GetScanItems(scan, so->value));
	}
	}

	if(ivfflat_bgw) {
		page_item* tmp;
		
		//elog(WARNING,"[DEBUG]: slot %d,%d",ret->pos,ret->returns);
		if (ret->pos == ret->returns || ret->returns == 0) {
			if(ret->next != NULL) {
				// Free and set next;
				worker_exec_entry* tmp = ret;
				ret = ret->next;
				free_slot(worker,tmp);
				//elog(WARNING,"[DEBUG]: slot freed");
			} else {
				return false;
			}
		}
		// ToDo: Do not return page_item but only ItemPointer
		tmp = (page_item*) &ret->data[ret->pos*sizeof(page_item)];
		heaptid = (ItemPointer) &tmp->ipd;
		ret->pos++;
	} else
#endif
	heaptid = (ItemPointer) DatumGetPointer(slot_getattr(so->mslot, 2, &isnull));

	scan->xs_heaptid = *heaptid;
	scan->xs_recheck = false;
	scan->xs_recheckorderby = false;
	return true;
}

/*
 * End a scan and release resources
 */
void
ivfflatendscan(IndexScanDesc scan)
{
	IvfflatScanOpaque so = (IvfflatScanOpaque) scan->opaque;

#ifdef AERO
	if(ivfflat_bgw) {
		if(ret != NULL && worker != NULL)
			// ToDo: Free all remaining slots in chain ! 	
			while(ret != NULL) {
				worker_exec_entry* tmp = ret;
				ret = ret->next;		
				free_slot(worker,tmp);
				//elog(WARNING,"[DEBUG]: end slot freed");
			}
	}
#endif

	/* Free any temporary files */
	tuplesort_end(so->sortstate);

	MemoryContextDelete(so->tmpCtx);

	pfree(so);
	scan->opaque = NULL;
}
