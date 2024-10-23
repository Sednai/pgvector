#include "postgres.h"

#include <float.h>

#include "access/relscan.h"
#include "ivfflat.h"
#include "miscadmin.h"
#include "storage/bufmgr.h"

#if PG_VERSION_NUM >= 110000
#include "catalog/pg_operator_d.h"
#include "catalog/pg_type_d.h"
#else
#include "catalog/pg_operator.h"
#include "catalog/pg_type.h"
#endif

#ifdef XZ
#include "ivfgpu.h"
#endif
/*
 * Compare list distances
 */
static int
CompareLists(const pairingheap_node *a, const pairingheap_node *b, void *arg)
{
	if (((const IvfflatScanList *) a)->distance > ((const IvfflatScanList *) b)->distance)
		return 1;

	if (((const IvfflatScanList *) a)->distance < ((const IvfflatScanList *) b)->distance)
		return -1;

	return 0;
}

/*
 * Get lists and sort by distance
 */
static void
GetScanLists(IndexScanDesc scan, Datum value)
{
	Buffer		cbuf;
	Page		cpage;
	IvfflatList list;
	OffsetNumber offno;
	OffsetNumber maxoffno;
	BlockNumber nextblkno = IVFFLAT_HEAD_BLKNO;
	int			listCount = 0;
	IvfflatScanOpaque so = (IvfflatScanOpaque) scan->opaque;
	double		distance;
	IvfflatScanList *scanlist;
	double		maxDistance = DBL_MAX;

	/* Search all list pages */
	while (BlockNumberIsValid(nextblkno))
	{
		cbuf = ReadBuffer(scan->indexRelation, nextblkno);
		LockBuffer(cbuf, BUFFER_LOCK_SHARE);
		cpage = BufferGetPage(cbuf);

		maxoffno = PageGetMaxOffsetNumber(cpage);

		for (offno = FirstOffsetNumber; offno <= maxoffno; offno = OffsetNumberNext(offno))
		{
			list = (IvfflatList) PageGetItem(cpage, PageGetItemId(cpage, offno));

			/* Use procinfo from the index instead of scan key for performance */
			distance = DatumGetFloat8(FunctionCall2Coll(so->procinfo, so->collation, PointerGetDatum(&list->center), value));

			if (listCount < so->probes)
			{
				scanlist = &so->lists[listCount];
				scanlist->startPage = list->startPage;
				scanlist->distance = distance;
				listCount++;

				/* Add to heap */
				pairingheap_add(so->listQueue, &scanlist->ph_node);

				/* Calculate max distance */
				if (listCount == so->probes)
					maxDistance = ((IvfflatScanList *) pairingheap_first(so->listQueue))->distance;
			}
			else if (distance < maxDistance)
			{
				/* Remove */
				scanlist = (IvfflatScanList *) pairingheap_remove_first(so->listQueue);

				/* Reuse */
				scanlist->startPage = list->startPage;
				scanlist->distance = distance;
				pairingheap_add(so->listQueue, &scanlist->ph_node);

				/* Update max distance */
				maxDistance = ((IvfflatScanList *) pairingheap_first(so->listQueue))->distance;
			}
		}

		nextblkno = IvfflatPageGetOpaque(cpage)->nextblkno;

		UnlockReleaseBuffer(cbuf);
	}
}

/*
 * Get items
 */
static void
GetScanItems(IndexScanDesc scan, Datum value)
{
	IvfflatScanOpaque so = (IvfflatScanOpaque) scan->opaque;
	Buffer		buf;
	Page		page;
	IndexTuple	itup;
	BlockNumber searchPage;
	OffsetNumber offno;
	OffsetNumber maxoffno;
	Datum		datum;
	bool		isnull;
	TupleDesc	tupdesc = RelationGetDescr(scan->indexRelation);

#if PG_VERSION_NUM >= 120000
	TupleTableSlot *slot = MakeSingleTupleTableSlot(so->tupdesc, &TTSOpsVirtual);
#else
	TupleTableSlot *slot = MakeSingleTupleTableSlot(so->tupdesc);
#endif

	/*
	 * Reuse same set of shared buffers for scan
	 *
	 * See postgres/src/backend/storage/buffer/README for description
	 */
	BufferAccessStrategy bas = GetAccessStrategy(BAS_BULKREAD);

#ifdef XZ
	int c = 0;
	ParallelIndexScanDesc parallel_scan = scan->parallel_scan;
	IvfflatScanParallel ivff_target;
	if(scan->parallel_scan)
		ivff_target = (IvfflatScanParallel) OffsetToPointer((void *) parallel_scan,parallel_scan->ps_offset);
#endif

#ifdef XZ
	int BATCH_SIZE;
	Vector	   *v; 
	float* M;
	float* V;
	float* C;
	
	
	ItemPointerData* tmp_tid; 
	Datum* tmp_page;
	int row;

	if(ivfflat_gpu) {
		BATCH_SIZE = ivfflat_gpu_batchsize;
		v = PointerGetDatum(value);
		M = (float*) init_shared_gpu_memory(v->dim*BATCH_SIZE*sizeof(float) );
		V = (float*) init_shared_gpu_memory(v->dim*sizeof(float) );
		C = (float*) init_shared_gpu_memory(BATCH_SIZE*sizeof(float) );
		memcpy(V,v->x,v->dim*sizeof(float));
		advise_memory_readonly(V, v->dim*sizeof(float), 0); 
		prefetch_gpu_memory(V, v->dim*sizeof(float), 0);

		tmp_tid = palloc(sizeof(ItemPointerData)*BATCH_SIZE); 
		tmp_page = palloc(sizeof(Datum) * BATCH_SIZE);

		row = 0;
	}

#endif

	/* Search closest probes lists */
	while (!pairingheap_is_empty(so->listQueue))
	{
		searchPage = ((IvfflatScanList *) pairingheap_remove_first(so->listQueue))->startPage;

#ifdef XZ
		if(scan->parallel_scan) {
			// Sync if already visited
			SpinLockAcquire(&ivff_target->lock);
			if( c < ivff_target->next) {
			    SpinLockRelease(&ivff_target->lock);			
				c++;
				continue;
			}
			ivff_target->next++;
			c++;
		    SpinLockRelease(&ivff_target->lock);
		}
#endif

		/* Search all entry pages for list */
		while (BlockNumberIsValid(searchPage))
		{
			buf = ReadBufferExtended(scan->indexRelation, MAIN_FORKNUM, searchPage, RBM_NORMAL, bas);
			LockBuffer(buf, BUFFER_LOCK_SHARE);
			page = BufferGetPage(buf);
			maxoffno = PageGetMaxOffsetNumber(page);

			for (offno = FirstOffsetNumber; offno <= maxoffno; offno = OffsetNumberNext(offno))
			{
				itup = (IndexTuple) PageGetItem(page, PageGetItemId(page, offno));
				datum = index_getattr(itup, 1, tupdesc, &isnull);

#ifdef XZ
				if(ivfflat_gpu) {
					if (row == BATCH_SIZE) {

						calc_distances_gpu_euclidean(M, V, C, row, v->dim);
						
						for(int r = 0; r < row; r++) {
							ExecClearTuple(slot);
							slot->tts_values[0] = Float4GetDatum( C[r] );
							slot->tts_isnull[0] = false;
							slot->tts_values[1] = Int16GetDatum( tmp_tid[r].ip_blkid.bi_hi);
							slot->tts_isnull[1] = false;
							slot->tts_values[2] = Int16GetDatum( tmp_tid[r].ip_blkid.bi_lo);
							slot->tts_isnull[2] = false;
							slot->tts_values[3] = Int16GetDatum( tmp_tid[r].ip_posid);
							slot->tts_isnull[3] = false;
							slot->tts_values[4] = tmp_page[r];
							slot->tts_isnull[4] = false;
							ExecStoreVirtualTuple(slot);
							
							tuplesort_puttupleslot(so->sortstate, slot);
						}			
						row = 0;
					}

					Vector	   *a = PointerGetDatum(datum);
					memcpy(&M[row*a->dim],a->x,a->dim*sizeof(float));

					tmp_tid[row] = itup->t_tid;
					tmp_page[row] = Int32GetDatum((int) searchPage);
					
					row++;

					if(row % 100000 == 0) {
						prefetch_gpu_memory(&M[row-100000*v->dim*sizeof(float)], 100000*v->dim*sizeof(float), 0);
					}
				} else {
						 
					ExecClearTuple(slot);
					slot->tts_values[0] = FunctionCall2Coll(so->procinfo, so->collation, datum, value);
					slot->tts_isnull[0] = false;
					slot->tts_values[1] = Int16GetDatum( itup->t_tid.ip_blkid.bi_hi);
					slot->tts_isnull[1] = false;
					slot->tts_values[2] = Int16GetDatum( itup->t_tid.ip_blkid.bi_lo);
					slot->tts_isnull[2] = false;
					slot->tts_values[3] = Int16GetDatum( itup->t_tid.ip_posid);
					slot->tts_isnull[3] = false;
					slot->tts_values[4] = Int32GetDatum((int) searchPage);
					slot->tts_isnull[4] = false;
					ExecStoreVirtualTuple(slot);

					tuplesort_puttupleslot(so->sortstate, slot);
				}

#else	

				/*
				 * Add virtual tuple
				 *
				 * Use procinfo from the index instead of scan key for
				 * performance
				 */
				ExecClearTuple(slot);
				slot->tts_values[0] = FunctionCall2Coll(so->procinfo, so->collation, datum, value);
				slot->tts_isnull[0] = false;
				slot->tts_values[1] = PointerGetDatum(&itup->t_tid);
				slot->tts_isnull[1] = false;
				slot->tts_values[2] = Int32GetDatum((int) searchPage);
				slot->tts_isnull[2] = false;
				ExecStoreVirtualTuple(slot);

				tuplesort_puttupleslot(so->sortstate, slot);
#endif
			}

			searchPage = IvfflatPageGetOpaque(page)->nextblkno;

			UnlockReleaseBuffer(buf);
		}

	}

#ifdef XZ
	if(ivfflat_gpu) {
		if(row > 0) {	
			calc_distances_gpu_euclidean(M, V, C, row, v->dim);
			
			for(int r = 0; r < row; r++) {
				ExecClearTuple(slot);
				slot->tts_values[0] = Float4GetDatum( C[r] );
				slot->tts_isnull[0] = false;
				slot->tts_values[1] = Int16GetDatum( tmp_tid[r].ip_blkid.bi_hi);
				slot->tts_isnull[1] = false;
				slot->tts_values[2] = Int16GetDatum( tmp_tid[r].ip_blkid.bi_lo);
				slot->tts_isnull[2] = false;
				slot->tts_values[3] = Int16GetDatum( tmp_tid[r].ip_posid);
				slot->tts_isnull[3] = false;
				slot->tts_values[4] = tmp_page[r];
				slot->tts_isnull[4] = false;
				ExecStoreVirtualTuple(slot);
				
				tuplesort_puttupleslot(so->sortstate, slot);
			}
		}

		free_shared_gpu_memory(M);
		free_shared_gpu_memory(V);
		free_shared_gpu_memory(C);
		pfree(tmp_page);
		pfree(tmp_tid);
	}
#endif

	tuplesort_performsort(so->sortstate);
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
	AttrNumber	attNums[] = {1};
	Oid			sortOperators[] = {Float8LessOperator};
	Oid			sortCollations[] = {InvalidOid};
	bool		nullsFirstFlags[] = {false};
	int			probes = ivfflat_probes;

	scan = RelationGetIndexScan(index, nkeys, norderbys);
	lists = IvfflatGetLists(scan->indexRelation);

	if (probes > lists)
		probes = lists;

	so = (IvfflatScanOpaque) palloc(offsetof(IvfflatScanOpaqueData, lists) + probes * sizeof(IvfflatScanList));
	so->buf = InvalidBuffer;
	so->first = true;
	so->probes = probes;

	/* Set support functions */
	so->procinfo = index_getprocinfo(index, 1, IVFFLAT_DISTANCE_PROC);
	so->normprocinfo = IvfflatOptionalProcInfo(index, IVFFLAT_NORM_PROC);
	so->collation = index->rd_indcollation[0];

	/* Create tuple description for sorting */
#ifndef XZ

#if PG_VERSION_NUM >= 120000
	so->tupdesc = CreateTemplateTupleDesc(3);
#else
	so->tupdesc = CreateTemplateTupleDesc(3, false);
#endif

	TupleDescInitEntry(so->tupdesc, (AttrNumber) 1, "distance", FLOAT8OID, -1, 0);

#endif


#ifdef XZ
	so->tupdesc = CreateTemplateTupleDesc(5, false);

	TupleDescInitEntry(so->tupdesc, (AttrNumber) 1, "distance", FLOAT8OID, -1, 0);
	TupleDescInitEntry(so->tupdesc, (AttrNumber) 2, "tid_a", INT2OID, -1, 0);
	TupleDescInitEntry(so->tupdesc, (AttrNumber) 3, "tid_b", INT2OID, -1, 0);
	TupleDescInitEntry(so->tupdesc, (AttrNumber) 4, "tid_c", INT2OID, -1, 0);
	TupleDescInitEntry(so->tupdesc, (AttrNumber) 5, "indexblkno", INT4OID, -1, 0);

#else
	TupleDescInitEntry(so->tupdesc, (AttrNumber) 2, "tid", TIDOID, -1, 0);
	TupleDescInitEntry(so->tupdesc, (AttrNumber) 3, "indexblkno", INT4OID, -1, 0);
#endif
	/* Prep sort */
#if PG_VERSION_NUM >= 110000
	so->sortstate = tuplesort_begin_heap(so->tupdesc, 1, attNums, sortOperators, sortCollations, nullsFirstFlags, work_mem, NULL, false);
#else
	so->sortstate = tuplesort_begin_heap(so->tupdesc, 1, attNums, sortOperators, sortCollations, nullsFirstFlags, work_mem, false);
#endif

#if PG_VERSION_NUM >= 120000
	so->slot = MakeSingleTupleTableSlot(so->tupdesc, &TTSOpsMinimalTuple);
#else
	so->slot = MakeSingleTupleTableSlot(so->tupdesc);
#endif

	so->listQueue = pairingheap_allocate(CompareLists, scan);

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

#if PG_VERSION_NUM >= 130000
	if (!so->first)
		tuplesort_reset(so->sortstate);
#endif

	so->first = true;
	pairingheap_reset(so->listQueue);

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

	/*
	 * Index can be used to scan backward, but Postgres doesn't support
	 * backward scan on operators
	 */
	Assert(ScanDirectionIsForward(dir));

	if (so->first)
	{
		Datum		value;

		/* Safety check */
		if (scan->orderByData == NULL)
			elog(ERROR, "cannot scan ivfflat index without order");

		/* No items will match if null */
		if (scan->orderByData->sk_flags & SK_ISNULL)
			return false;

		value = scan->orderByData->sk_argument;

		if (so->normprocinfo != NULL)
		{
			/* No items will match if normalization fails */
			if (!IvfflatNormValue(so->normprocinfo, so->collation, &value, NULL))
				return false;
		}

		IvfflatBench("GetScanLists", GetScanLists(scan, value));
		IvfflatBench("GetScanItems", GetScanItems(scan, value));
		so->first = false;

		/* Clean up if we allocated a new value */
		if (value != scan->orderByData->sk_argument)
			pfree(DatumGetPointer(value));
	}

#if PG_VERSION_NUM >= 100000
	if (tuplesort_gettupleslot(so->sortstate, true, false, so->slot, NULL))
#else
	if (tuplesort_gettupleslot(so->sortstate, true, so->slot, NULL))
#endif
	{
#ifdef XZ
		ItemPointerData tid;
		tid.ip_blkid.bi_hi = DatumGetInt16(slot_getattr(so->slot, 2, &so->isnull));
		tid.ip_blkid.bi_lo = DatumGetInt16(slot_getattr(so->slot, 3, &so->isnull));
		tid.ip_posid = DatumGetInt16(slot_getattr(so->slot, 4, &so->isnull));

		scan->xs_ctup.t_self = tid;

		BlockNumber indexblkno = DatumGetInt32(slot_getattr(so->slot, 5, &so->isnull));

#else
		ItemPointer tid = (ItemPointer) DatumGetPointer(slot_getattr(so->slot, 2, &so->isnull));
		BlockNumber indexblkno = DatumGetInt32(slot_getattr(so->slot, 3, &so->isnull));

#if PG_VERSION_NUM >= 120000
		scan->xs_heaptid = *tid;
#else
		scan->xs_ctup.t_self = *tid;
#endif

#endif
		if (BufferIsValid(so->buf))
			ReleaseBuffer(so->buf);

		/*
		 * An index scan must maintain a pin on the index page holding the
		 * item last returned by amgettuple
		 *
		 * https://www.postgresql.org/docs/current/index-locking.html
		 */
		so->buf = ReadBuffer(scan->indexRelation, indexblkno);

		scan->xs_recheckorderby = false;
		return true;
	}

	return false;
}

/*
 * End a scan and release resources
 */
void
ivfflatendscan(IndexScanDesc scan)
{
	IvfflatScanOpaque so = (IvfflatScanOpaque) scan->opaque;

	/* Release pin */
	if (BufferIsValid(so->buf))
		ReleaseBuffer(so->buf);

	pairingheap_free(so->listQueue);
	tuplesort_end(so->sortstate);

	pfree(so);
	scan->opaque = NULL;
}


#ifdef XZ
/*
 * Estimate storage
*/
Size
ivffestimateparallelscan(void)
{
	//elog(WARNING,"[DEBUG](ivffestimateparallelscan)");
	return sizeof(IvfflatScanParallelData);
}

/*
*	initialize parallel scan
*/
void
ivffinitparallelscan(void *target)
{
#ifdef XZ_DEBUG
	elog(WARNING,"[DEBUG](ivffinitparallelscan)");
#endif
    IvfflatScanParallel ivff_target = (IvfflatScanParallel) target;

    SpinLockInit(&ivff_target->lock);

	SpinLockAcquire(&ivff_target->lock);
	ivff_target->next = 0;
	SpinLockRelease(&ivff_target->lock);
}

/*
 *    reset parallel scan
 */
void
ivffparallelrescan(IndexScanDesc scan)
{
    ParallelIndexScanDesc parallel_scan = scan->parallel_scan;

    Assert(parallel_scan);
	IvfflatScanParallel ivff_target = (IvfflatScanParallel) OffsetToPointer((void *) parallel_scan,parallel_scan->ps_offset);

    SpinLockAcquire(&ivff_target->lock);
	ivff_target->next = 0;
	SpinLockRelease(&ivff_target->lock);
}

#endif