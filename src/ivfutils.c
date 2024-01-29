#include "postgres.h"

#include "ivfflat.h"
#include "storage/bufmgr.h"
#include "vector.h"

#ifdef XZ
#include "executor/spi.h" 
#include "catalog/pg_type.h"
#endif

/*
 * Allocate a vector array
 */
VectorArray
VectorArrayInit(int maxlen, int dimensions)
{
	VectorArray res = palloc_extended(VECTOR_ARRAY_SIZE(maxlen, dimensions), MCXT_ALLOC_ZERO | MCXT_ALLOC_HUGE);

	res->length = 0;
	res->maxlen = maxlen;
	res->dim = dimensions;
	return res;
}

/*
 * Print vector array - useful for debugging
 */
void
PrintVectorArray(char *msg, VectorArray arr)
{
	int			i;

	for (i = 0; i < arr->length; i++)
		PrintVector(msg, VectorArrayGet(arr, i));
}

/*
 * Get the number of lists in the index
 */
int
IvfflatGetLists(Relation index)
{
	IvfflatOptions *opts = (IvfflatOptions *) index->rd_options;

	if (opts)
		return opts->lists;

	return IVFFLAT_DEFAULT_LISTS;
}

#ifdef XZ
/*
 * Get supplied centroids
 */
char*
IvfflatGetCentroids(Relation index)
{
	IvfflatOptions *opts = (IvfflatOptions *) index->rd_options;

	if (opts)
		return (char *) opts + opts->centroidsOffset;
	
	return IVFFLAT_DEFAULT_CENTROIDS;
}

/*
 * Get supplied centroid table
 */
char*
IvfflatGetCentroidsTable(Relation index)
{
	IvfflatOptions *opts = (IvfflatOptions *) index->rd_options;

	if (opts)
		return (char *) opts + opts->centroidsTableOffset;
	
	return IVFFLAT_DEFAULT_CENTROIDSTABLE;
}

/*
 * Get supplied centroid column
 */
char*
IvfflatGetCentroidsCol(Relation index)
{
	IvfflatOptions *opts = (IvfflatOptions *) index->rd_options;

	if (opts)
		return (char *) opts + opts->centroidsColOffset;
	
	return IVFFLAT_DEFAULT_CENTROIDSCOL;
}

/*
 * Get supplied centroid schema
 */
char*
IvfflatGetCentroidsSchema(Relation index)
{
	IvfflatOptions *opts = (IvfflatOptions *) index->rd_options;

	if (opts)
		return (char *) opts + opts->centroidsSchemaOffset;
	
	return IVFFLAT_DEFAULT_CENTROIDSSCHEMA;
}

#endif




/*
 * Get proc
 */
FmgrInfo *
IvfflatOptionalProcInfo(Relation rel, uint16 procnum)
{
	if (!OidIsValid(index_getprocid(rel, 1, procnum)))
		return NULL;

	return index_getprocinfo(rel, 1, procnum);
}

/*
 * Divide by the norm
 *
 * Returns false if value should not be indexed
 *
 * The caller needs to free the pointer stored in value
 * if it's different than the original value
 */
bool
IvfflatNormValue(FmgrInfo *procinfo, Oid collation, Datum *value, Vector * result)
{
	Vector	   *v;
	int			i;
	double		norm;

	norm = DatumGetFloat8(FunctionCall1Coll(procinfo, collation, *value));

	if (norm > 0)
	{
		v = (Vector *) DatumGetPointer(*value);

		if (result == NULL)
			result = InitVector(v->dim);

		for (i = 0; i < v->dim; i++)
			result->x[i] = v->x[i] / norm;

		*value = PointerGetDatum(result);

		return true;
	}

	return false;
}

/*
 * New buffer
 */
Buffer
IvfflatNewBuffer(Relation index, ForkNumber forkNum)
{
	Buffer		buf = ReadBufferExtended(index, forkNum, P_NEW, RBM_NORMAL, NULL);

	LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
	return buf;
}

/*
 * Init page
 */
void
IvfflatInitPage(Buffer buf, Page page)
{
	PageInit(page, BufferGetPageSize(buf), sizeof(IvfflatPageOpaqueData));
	IvfflatPageGetOpaque(page)->nextblkno = InvalidBlockNumber;
	IvfflatPageGetOpaque(page)->page_id = IVFFLAT_PAGE_ID;
}

/*
 * Init and register page
 */
void
IvfflatInitRegisterPage(Relation index, Buffer *buf, Page *page, GenericXLogState **state)
{
	*state = GenericXLogStart(index);
	*page = GenericXLogRegisterBuffer(*state, *buf, GENERIC_XLOG_FULL_IMAGE);
	IvfflatInitPage(*buf, *page);
}

/*
 * Commit buffer
 */
void
IvfflatCommitBuffer(Buffer buf, GenericXLogState *state)
{
	MarkBufferDirty(buf);
	GenericXLogFinish(state);
	UnlockReleaseBuffer(buf);
}

/*
 * Add a new page
 *
 * The order is very important!!
 */
void
IvfflatAppendPage(Relation index, Buffer *buf, Page *page, GenericXLogState **state, ForkNumber forkNum)
{
	/* Get new buffer */
	Buffer		newbuf = IvfflatNewBuffer(index, forkNum);
	Page		newpage = GenericXLogRegisterBuffer(*state, newbuf, GENERIC_XLOG_FULL_IMAGE);

	/* Update the previous buffer */
	IvfflatPageGetOpaque(*page)->nextblkno = BufferGetBlockNumber(newbuf);

	/* Init new page */
	IvfflatInitPage(newbuf, newpage);

	/* Commit */
	MarkBufferDirty(*buf);
	MarkBufferDirty(newbuf);
	GenericXLogFinish(*state);

	/* Unlock */
	UnlockReleaseBuffer(*buf);

	*state = GenericXLogStart(index);
	*page = GenericXLogRegisterBuffer(*state, newbuf, GENERIC_XLOG_FULL_IMAGE);
	*buf = newbuf;
}

/*
 * Update the start or insert page of a list
 */
void
IvfflatUpdateList(Relation index, GenericXLogState *state, ListInfo listInfo,
				  BlockNumber insertPage, BlockNumber originalInsertPage,
				  BlockNumber startPage, ForkNumber forkNum)
{
	Buffer		buf;
	Page		page;
	IvfflatList list;
	bool		changed = false;

	buf = ReadBufferExtended(index, forkNum, listInfo.blkno, RBM_NORMAL, NULL);
	LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
	state = GenericXLogStart(index);
	page = GenericXLogRegisterBuffer(state, buf, 0);
	list = (IvfflatList) PageGetItem(page, PageGetItemId(page, listInfo.offno));

	if (BlockNumberIsValid(insertPage) && insertPage != list->insertPage)
	{
		/* Skip update if insert page is lower than original insert page  */
		/* This is needed to prevent insert from overwriting vacuum */
		if (!BlockNumberIsValid(originalInsertPage) || insertPage >= originalInsertPage)
		{
			list->insertPage = insertPage;
			changed = true;
		}
	}

	if (BlockNumberIsValid(startPage) && startPage != list->startPage)
	{
		list->startPage = startPage;
		changed = true;
	}

	/* Only commit if changed */
	if (changed)
		IvfflatCommitBuffer(buf, state);
	else
	{
		GenericXLogAbort(state);
		UnlockReleaseBuffer(buf);
	}
}

#ifdef XZ
void getCentroidsFromTable(char* schemaname, char* tabname, char* colname,int N, int dim, VectorArray centroids) {

	// SPI connect to server
    SPI_connect();
    
	char* query_cmd_1 = "select exists (select 1 from information_schema.columns where table_name='";
	char* query_cmd_2 = "' and column_name='";
	char* query_cmd_3 = "' and table_schema='";
	int len;
	
	if(schemaname == NULL)
		len = strlen(query_cmd_1)+strlen(query_cmd_2)+strlen(tabname)+strlen(colname)+1;
	else
		len = strlen(query_cmd_1)+strlen(query_cmd_2)+strlen(query_cmd_3)+strlen(tabname)+strlen(colname)+strlen(schemaname)+1;

	char query[len];
	
	if(schemaname == NULL) {
		strcpy(query,query_cmd_1);
		strcat(query,tabname);
		strcat(query,query_cmd_2);
		strcat(query,colname);
		strcat(query,"')");
	} else {
		strcpy(query,query_cmd_1);
		strcat(query,tabname);
		strcat(query,query_cmd_2);
		strcat(query,colname);
		strcat(query,query_cmd_3);
		strcat(query,schemaname);
		strcat(query,"')");		
	}

	//elog(WARNING,"[DEBUG](query): %s",query);

	// 1. Check if table exists
  	SPIPlanPtr plan = SPI_prepare_cursor(query, 0, NULL, 0);
            
    Portal prtl = SPI_cursor_open(NULL, plan, NULL, NULL, true);
	
	SPI_cursor_fetch(prtl, true, 1);
	bool table_found = false;

	TupleDesc tupdesc;
	SPITupleTable *tuptable;

	if (SPI_processed > 0) {
		tupdesc = SPI_tuptable->tupdesc;
		tuptable = SPI_tuptable;
	 	HeapTuple row = tuptable->vals[0];
		bool isnull;
		Datum col = SPI_getbinval(row, tupdesc, 1, &isnull);
        table_found =  DatumGetBool(col);        
	}

	SPI_cursor_close(prtl);

	if(table_found) {
		// 2. Query for centroids
		
		int tablen;
		if(schemaname != NULL) {
			tablen = strlen(tabname)+strlen(schemaname)+1;
		}
		else {
			tablen = strlen(tabname);
		}
		char tabn[tablen];
		strcpy(tabn,tabname);
		strcat(tabn,".");

		if(schemaname != NULL) {
			strcpy(tabn,schemaname);
		}

		char* query_cmd_2_1 = "select ";
		char* query_cmd_2_2 = " from ";
		
		char query2[strlen(query_cmd_2_1)+strlen(query_cmd_2_2)+strlen(tabn)+strlen(colname)];
	
		strcpy(query2,query_cmd_2_1);
		strcat(query2,colname);
		strcat(query2,query_cmd_2_2);
		strcat(query2,tabn);

		plan = SPI_prepare_cursor(query2, 0, NULL, 0);
		prtl = SPI_cursor_open(NULL, plan, NULL, NULL, true);

		SPI_cursor_fetch(prtl, true, N);
		bool isnull = false;
		bool sizemismatch = false;
		bool typemismatch = false;

		if (SPI_processed > 0) {
			tupdesc = SPI_tuptable->tupdesc;
			tuptable = SPI_tuptable;
		
			// Build vector array
			for(int i = 0; i < SPI_processed; i++) {
				HeapTuple row = tuptable->vals[i];
				Datum col = SPI_getbinval(row, tupdesc, 1, &isnull);
				if(!isnull) {
					// Convert array 
					ArrayType* arr = DatumGetArrayTypeP( col );
					
					int size = (int) ArrayGetNItems(ARR_NDIM(arr), ARR_DIMS(arr));
					
					int type = (int) ARR_ELEMTYPE(arr);
					
					if(type!=FLOAT4OID) {
						typemismatch = true;
						break;
					}

					if(size != dim) {
						sizemismatch = true;
						break;
					}
				
					float* data = (float*) ARR_DATA_PTR(arr); 
					Vector* result = InitVector(dim);
					for (int j = 0; j < dim; j++)
						result->x[j] = data[j];
					
					VectorArraySet(centroids, i, result);
					centroids->length++;
	
				} else {
					break;
				}
			}
		}

		SPI_cursor_close(prtl);
		SPI_finish();

		if(isnull) {
			elog(ERROR,"Centroid array with NULLs detected");
		}

		if(sizemismatch) {
			elog(ERROR,"Centroid with non fitting dimension detected");
		}
	
		if(typemismatch) {
			elog(ERROR,"Centroid with non fitting typeoid detected. Float4[] is required");
		}


	} else {
		SPI_finish();
		if(schemaname!=NULL) {
			elog(ERROR,"Centroid table %s in schema %s with column %s not found", tabname, schemaname, colname);
		}
		else
			elog(ERROR,"Centroid table %s with column %s not found", tabname, colname);
		
	}

	if(centroids->length!=N) {
		elog(ERROR,"Too few centroids found in table (%d of %d)",centroids->length,N);
	}

}




#endif