#define INIT_STORE_SIZE 100000

#include <cstdio>
#include <iostream>
#include <list>
#include <algorithm>
#include <bits/stdc++.h>
#include "vector.h"
#include "storage/relfilenode.h"
#include "storage/itemptr.h"

#include "ivfgpu.h"
#include "gpucache.hpp"

#include "gpuworker.h"

#undef qsort
#include <stdlib.h>

using namespace std;

static void adjust_buffer(page_list* L, long n_new_elements) {
	// Adjust buffer
	if(L->length+n_new_elements >= L->max_length) {
        long n = (L->max_length+n_new_elements)+1;
        L->data = (page_item*) realloc(L->data,n*sizeof(page_item));        
        L->max_length = n;
    } 
}				

class Relation {
    public:
        RelFileNode node;

        bool operator==(const Relation &r) const {
            return node.spcNode == r.node.spcNode && node.dbNode == r.node.dbNode && node.relNode == r.node.relNode;
        }
};

class RelHashFunc {
    public:
        size_t operator()(const Relation& r) const {
            return (hash<uint>()(r.node.spcNode)) ^ 
                (hash<uint>()(r.node.dbNode)) ^
                (hash<uint>()(r.node.relNode));
        }
};

__inline__ float squared_eucl_dist(const float* X, const float* Y, int N) {
    float D = (X[0] - Y[0])*(X[0] - Y[0]);
    for(int i = 1; i < N; i++) {
        D += (X[i] - Y[i])*(X[i] - Y[i]);
    }
    return D;
}

static int compare_pi(const void* a, const void* b) {
	
	const page_item *elem1 = (page_item*) a;    
    const page_item *elem2 = (page_item*) b;

   if (elem1->distance < elem2->distance)
      return -1;
   else if (elem1->distance > elem2->distance)
      return 1;
   else
      return 0;
}

static inline bool filter_func(float val, float cond, int mode) {
	switch(mode) {
		case 0:
			if( val == cond) 
				return true;
			break;
		case 1:
			if( val > cond)
				return true;
			break;
		case -1:
			if( val < cond)
				return true;
			break;
		case 2:
			if (val >= cond)
				return true;
			break;
		case -2:
			if (val <= cond)
				return true;
			break;
		case -100:
			return true;
	}

	return false;
}

static inline bool triangle_filter(float qcdist, float pcdist, float cond, int mode) {
    //cout << "qc: " << qcdist << " pc: " << pcdist << " -> " << abs(qcdist - pcdist) << " cond: " << cond << " mode: " << mode << endl;

    switch(mode) {
        case -1: // OP: <
            if ( abs(qcdist - pcdist) >= cond)
                return false;
            break;
    }

    return true;    
}

/*
static void adjust_buffer(page_list* L, long n_new_elements) {
	// Adjust buffer
	if(L->length+n_new_elements >= L->max_length) {
        long n = (L->max_length+n_new_elements)+1;
        L->data = (page_item*) realloc(L->data,n*sizeof(page_item));
        //if(L->data == NULL) 
        //    elog(ERROR,"Fatal error occured in re-allocating buffer memory.");
        
        L->max_length = n;

    } 
}				
*/

class probe_entry {
    // Data
    float* vectors_gpu = nullptr;   
    float* vectors_cpu;
    float* centroid_distance = nullptr;

    int* pages;
    ItemPointerData* itdata;

    long maxlength = INIT_STORE_SIZE;
    long length = 0;
    
    public:
        int dim;
        float* probe;
        
        probe_entry(Vector* v, bool use_triangle) {
            // Init
            dim = v->dim;
            probe = new float[v->dim];
            memcpy(probe,v->x,v->dim*sizeof(float));

            vectors_cpu = (float*) malloc(INIT_STORE_SIZE*dim*sizeof(float));
            pages = (int*) malloc(INIT_STORE_SIZE*sizeof(int));
            itdata = (ItemPointerData*) malloc(INIT_STORE_SIZE*sizeof(ItemPointerData));

            if(use_triangle)
                centroid_distance = (float*) malloc(INIT_STORE_SIZE*sizeof(float));
        } 
        
        void insert_vector(Vector* v, int page, ItemPointerData ipd) {
            if(maxlength - length == 0)  {
                // Enlarge storage 
                maxlength *= 1.5;
                vectors_cpu = (float*) realloc(vectors_cpu, maxlength*dim*sizeof(float));
                pages = (int*) realloc(pages, maxlength*sizeof(int));
                itdata = (ItemPointerData*) realloc(itdata, maxlength*sizeof(ItemPointerData));
            } 

            memcpy(&vectors_cpu[dim*length], v->x, v->dim*sizeof(float));
            pages[length] = page;
            itdata[length] = ipd;

            length++;
        }

        void insert_vector(Vector* v, float distance, int page, ItemPointerData ipd) {
            if(maxlength - length == 0)  {
                // Enlarge storage 
                maxlength *= 1.5;
                vectors_cpu = (float*) realloc(vectors_cpu, maxlength*dim*sizeof(float));
                pages = (int*) realloc(pages, maxlength*sizeof(int));
                itdata = (ItemPointerData*) realloc(itdata, maxlength*sizeof(ItemPointerData));
                centroid_distance = (float*) realloc(centroid_distance, maxlength*sizeof(float));        
            } 

            memcpy(&vectors_cpu[dim*length], v->x, v->dim*sizeof(float));
            pages[length] = page;
            itdata[length] = ipd;
            centroid_distance[length] = sqrt(distance); // Note: its stored squared
       
            length++;
        }
 
        long size() {
            return length;
        }
        
        const float* getVectorCPU(long idx) {
            return &vectors_cpu[idx*dim];
        }

/*
        void printVectorCPU(long idx) {
            cout << "[ ";
            for(int i = 0; i < dim; i++) {
                cout << vectors_cpu[idx*dim+i] << " ";
            }
            cout << "]" << endl;
        }

        void printVectorRaw(float *f) {
            cout << "[ ";
            for(int i = 0; i < dim; i++) {
                cout << f[i] << " ";
            }
            cout << "]" << endl;
        }
*/
        float getCentroidDistanceCPU(long idx) {
            return centroid_distance[idx];
        }

        float* getAllVectorsCPU() {
            return vectors_cpu;
        }
        
        const int getPage(long idx) {
            return pages[idx];
        }

        const ItemPointerData getItemPointerData(long idx) {
            return itdata[idx];
        }

        float* getAllVectorsGPU() {
            return vectors_gpu; 
        }
#ifdef GPU
        void storeOnGPU() {
            
            if(vectors_gpu == nullptr) {
                init_gpu();
                
                // Init ordinary cuda memory
                vectors_gpu = (float*) init_gpu_memory((void**) &vectors_gpu, length * dim * sizeof(float) );
                
                // Copy
                copy_memory_to_gpu(vectors_gpu, vectors_cpu, length*dim*sizeof(float));
            }
        }
#endif
        ~probe_entry() {
            delete[] probe;
            free(vectors_cpu);
            free(pages);
            free(itdata);
/*
            if(vectors_gpu != nullptr) {
                free_gpu_memory(vectors_gpu);
            }
*/
        }   
};

class probes {
    vector<probe_entry*> PROBES;
    public:
          
        int insert(Vector* in, bool use_triangle) {
            probe_entry *PE = new probe_entry(in,use_triangle);
            
            PROBES.push_back( PE );    

            return PROBES.size()-1;
        }

        void insert_vector(int probenumber, Vector* x, int page, ItemPointerData ipd) {
            PROBES[probenumber]->insert_vector(x, page, ipd);
        }

        void insert_vector(int probenumber, Vector* x, float distance, int page, ItemPointerData ipd) {
            PROBES[probenumber]->insert_vector(x, distance, page, ipd);
        }

        probe_entry* get(int p) {
            return PROBES[p];
        }

        vector<int> get_ordered_probes_idx(float* q) {
            
            // Pre-calculate q <-> probe distances
            float* dist = (float*) malloc(PROBES.size()*sizeof(float));
            
            for(long unsigned int i = 0; i < PROBES.size(); i++) {
                dist[i] = squared_eucl_dist( PROBES[i]->probe, q, PROBES[i]->dim);
            }

            // Build sorted index
            vector<int> idx(PROBES.size());
            iota(idx.begin(), idx.end(), 0);
            
            sort(idx.begin(), idx.end(),[&dist,&q](int i1, int i2) { return dist[i1] < dist[i2]; });
            
            free(dist);

            return idx;
        }

        int size() {
            return PROBES.size();
        }

        long numvectors() {
            long N = 0;
            for(long unsigned i = 0; i < PROBES.size(); i++) {
                N += PROBES[i]->size();
            }
            return N;
        }
};

class cpucache {
    
    unordered_map<Relation, probes*, RelHashFunc> MAP;

    public:
    
        bool contains(Relation node) { 
            if (MAP.find(node) == MAP.end())
                return false;
            else 
                return true;
        }

        int insert(Relation node, Vector* c, bool use_triangle) {
        
            probes *P;
            if(!contains(node)) {
                P = new probes();
                MAP[node] = P;
            } else {
                P = MAP[node];
            }

            return P->insert(c, use_triangle);
        }
        
        probes* get(Relation node) {
            return MAP[node];
        }

        void insert_vector(Relation node, int probenumber, Vector* v, int page, ItemPointerData ipd ) {
            probes *P = MAP[node];
            P->insert_vector(probenumber, v, page, ipd);
        }

        void insert_vector(Relation node, int probenumber, Vector* v, float dist, int page, ItemPointerData ipd ) {
            probes *P = MAP[node];
            P->insert_vector(probenumber, v, dist, page, ipd);
        }

        void logsize() {
            int probes = 0;
            long vectors = 0;
            for(auto &it : MAP) {
                probes += MAP[it.first]->size();
                vectors += MAP[it.first]->numvectors();
            }

           cout << "GPUCACHE: # Relations: " << MAP.size() << " # probes: " << probes << " # vectors: " << vectors << endl;
        }

        ~cpucache() {
            for(auto &it : MAP) {
                delete MAP[it.first];
            }
        }
};



cpucache* CACHE = new cpucache();


bool incache(RelFileNode node) {

    Relation R = {node};

    return CACHE->contains(R);
}

int new_probe(RelFileNode node, Vector* c, bool use_triangle) {
    Relation R = {node};

    return CACHE->insert(R, c, use_triangle);

}

void insert(RelFileNode node, int probenumber, Vector* c, int page, ItemPointerData ipd) {
    Relation R = {node};

    return CACHE->insert_vector(R, probenumber, c, page, ipd);
}

void insert_wdistance(RelFileNode node, int probenumber, Vector* c, float distance, int page, ItemPointerData ipd) {
    Relation R = {node};

    return CACHE->insert_vector(R, probenumber, c, distance, page, ipd);
}


void logsize() {
    CACHE->logsize();
}

int exec_query_cpu(worker_exec_entry* entry, worker_data_head* worker) {
    RelFileNode node = entry->nodeid;
    int Np = entry->probes;
    int op = entry->op;
    float filter = entry->filter;
    float sfilter = sqrt(entry->filter);
    float* q = entry->vector;
    int dim = entry->vec_dim;
    char* return_data = entry->data;

    Relation R = {node};
    
    // Get probes for relation
    probes *P = CACHE->get(R);

    vector<int> idx = P->get_ordered_probes_idx(q);
    
    page_list RET;
    RET.data = (page_item*) malloc(sizeof(page_item) * INIT_STORE_SIZE);
    RET.length = 0;
    RET.max_length = INIT_STORE_SIZE;
    
    int pcount = 0;

    for(long unsigned i = 0; i < min(idx.size(), (size_t) Np); i++) {
        // Get entry
        probe_entry* E = P->get(idx[i]);
        long L = E->size();
        
        adjust_buffer(&RET, L);
        
        // Calc distance q to centroid
        float qdist;
        if(entry->usetriangle) {
            qdist = sqrt( squared_eucl_dist(q, E->probe, dim) );
            //cout << i << ": " << qdist << endl;
        }

        // Loop over vectors    
        for(long j = 0; j < L; j++) {
            // Pre-filter with triangular inequalities
            if(entry->usetriangle && !triangle_filter(qdist, E->getCentroidDistanceCPU(j), sfilter, op)) {
                //cout << j << ": prefiltered" << endl;
                pcount++;
                continue;
            }
            float dist = squared_eucl_dist(q, E->getVectorCPU(j), dim);
            //E->printVectorCPU(j);
            //cout << j << ": true dist: " << sqrt(dist) << endl;
            // Filter
            if(!filter_func(dist,filter,op))
                continue;

            // Build return item
            page_item* I = &RET.data[RET.length];
			I->distance = dist;
            I->ipd = E->getItemPointerData(j);
			I->searchPage = E->getPage(j);
			RET.length++;
        }
    }

    //cout << "prefiltered: " << pcount << endl;

    if(entry->limit > 0) {
        if(entry->limit > RET.length) 
            entry->limit = RET.length;

        nth_element(RET.data, RET.data + entry->limit, RET.data+RET.length,  [](const page_item& a, const page_item& b) { return a.distance < b.distance; }  );
        qsort(RET.data, entry->limit, sizeof(page_item), compare_pi);
        
        if(entry->limit < RET.length) 
            RET.length = entry->limit;
    } else {
        // Sort
        qsort(RET.data, RET.length, sizeof(page_item), compare_pi);
    }

    if(RET.length*sizeof(page_item) <= MAX_DATA ) {
        // Copy to return
        memcpy(return_data,RET.data,RET.length*sizeof(page_item));
        entry->next = NULL;
        entry->returns = RET.length;
        entry->pos = 0;
    } else {
        // Split into parts
        int Np = MAX_DATA/sizeof(page_item);
//cout << "[DEBUG] page_items / slot: " << Np << endl;
        int N = RET.length/Np;
        if (RET.length % Np != 0)
            N++;

//cout << "[DEBUG] slots needed: " << N << " (" << RET.length << ")" << endl;
        
        // Request N-1 additional slots
        worker_exec_entry* slots[N-1];
        bool fail = false;
        for(int i = 0; i < N-1; i++) {
            worker_exec_entry* tmp = get_free_slot(worker);
            if(tmp != NULL)
                slots[i] = tmp;
            else {
                slots[i] = NULL;
                fail = true;
                break;
            }
        }
        
        if(fail) {
            // Cleanup and return
            for(int i = 0; i < N-1; i++) {
                if(slots[i] != NULL)
                    free_slot(worker, slots[i]);
                else 
                    break;
            }
            free(RET.data);
            return -1;
        }

        // Copy 1.
        memcpy(return_data,RET.data,Np*sizeof(page_item));
        entry->next = slots[0];
        entry->returns = Np;
        entry->pos = 0;
        
        // Copy remaining
        for(int i = 0; i < N-2; i++) {
            memcpy(slots[i]->data, RET.data+(i+1)*Np,Np*sizeof(page_item) );
            slots[i]->returns = Np;
            slots[i]->next = slots[i+1];
            slots[i]->pos = 0;
        }

        // Copy last
        slots[N-2]->returns = RET.length % Np;
        if(slots[N-2]->returns == 0)
            slots[N-2]->returns = Np;

        slots[N-2]->next = NULL;
        slots[N-2]->pos = 0;
        memcpy(slots[N-2]->data, RET.data+(N-1)*Np,slots[N-2]->returns*sizeof(page_item) );
    }

    // Free
    free(RET.data);

    return entry->returns;
}

int exec_query_gpu(worker_exec_entry* entry, worker_data_head* worker) {
#ifdef GPU
    RelFileNode node = entry->nodeid;
    int Np = entry->probes;
    int op = entry->op;
    float filter = entry->filter;
    float* q = entry->vector;
    int dim = entry->vec_dim;
    char* return_data = entry->data;
    Relation R = {node};
    
    // Get probes for relation
    probes *P = CACHE->get(R);

    vector<int> idx = P->get_ordered_probes_idx(q);
    
    Np = min(idx.size(), (size_t) Np);

    page_list RET;
    RET.data = (page_item*) malloc(sizeof(page_item) * INIT_STORE_SIZE);
    RET.length = 0;
    RET.max_length = INIT_STORE_SIZE;

    int L = 0;
    // Prepare for all probes at once
    for(int i = 0; i < Np; i++) {
        // Get entry
        probe_entry* E = P->get(idx[i]);
    
        // Store data on gpu if not stored yet
        E->storeOnGPU();

        L += E->size();
    }
  
    //cout << "[DEBUG](GPU): " << L << " (" << Np << "," << dim << "," << op << ")" << endl;

    // Store query vector on GPU
    float* d_q;
    d_q = (float*) init_gpu_memory((void**) &d_q, dim*sizeof(float) );       
    copy_memory_to_gpu(d_q, q, dim*sizeof(float));
    
    // pointer to on device distance results
    sort_item* d_r;
    d_r = (sort_item*) init_gpu_memory((void**) &d_r, L*sizeof(sort_item) );
    
    // pointer to active position
    int a = 0;
    int* d_a; 
    d_a = (int*) init_gpu_memory((void**) &d_a, sizeof(int) );
    copy_memory_to_gpu(d_a, &a, sizeof(int));

    L = 0;
    for(int i = 0; i < Np; i++) {
        // Get entry
        probe_entry* E = P->get(idx[i]);
        int Ll = E->size();
       
        // Calc distances + filter
        calc_squared_distances_gpu_euclidean_wfilter(E->getAllVectorsGPU(), d_q, d_r, filter, d_a, Ll, dim, i, op); 

        L += Ll;
    }

    // Copy back  
    // pos index
    copy_memory_to_cpu(&a, d_a, sizeof(int));
  
   
    if(entry->limit > 0) {
        if(entry->limit > a) 
            entry->limit = a;

        // nth-element sort
        sort_item_array_nth_gpu(d_r,a,entry->limit);    

        if(entry->limit < a) 
            a = entry->limit;
    } else {
        // Sort on GPU
        sort_item_array_gpu(d_r, a); 
    }

    sort_item* d_r_cpu = (sort_item*) malloc(a*sizeof(sort_item));
    copy_memory_to_cpu(d_r_cpu, d_r, a*sizeof(sort_item));

    if(a*sizeof(page_item) <= MAX_DATA ) {
        for(int i = 0; i < a; i++) {
            probe_entry* E = P->get( idx[ d_r_cpu[i].probe]  );
            page_item* I = &((page_item*) return_data)[i];

            I->distance = d_r_cpu[i].distance;
            I->ipd = E->getItemPointerData( d_r_cpu[i].pos );
            I->searchPage = E->getPage( d_r_cpu[i].pos );
        }
        entry->next = NULL;
        entry->returns = a;
        entry->pos = 0;
    }
    else {
        // Split into parts
        int Np = MAX_DATA/sizeof(page_item);
        int N = a/Np;
        if (a % Np != 0)
            N++;

        //cout << "[DEBUG] slots needed: " << N << " (" << a << ","<< Np << ")" << endl;

        // Request N slots
        worker_exec_entry* slots[N-1];
        bool fail = false;
        for(int i = 0; i < N-1; i++) {
            worker_exec_entry* tmp = get_free_slot(worker);
            if(tmp != NULL)
                slots[i] = tmp;
            else {
                slots[i] = NULL;
                fail = true;
                break;
            }
        }
        
        if(fail) {
            //cout << "[DEBUG] slots: FAIL" << endl;

            // Cleanup and return
            for(int i = 0; i < N-1; i++) {
                if(slots[i] != NULL)
                    free_slot(worker, slots[i]);
                else 
                    break;
            }
            
            // Cleanup
            free_gpu_memory(d_a);
            free(d_r_cpu);
            free_gpu_memory(d_q);
            free_gpu_memory(d_r);

            return -1;
        }

        // Copy 1.
        for(int i = 0; i < Np; i++) {
            probe_entry* E = P->get( idx[ d_r_cpu[i].probe]  );
            page_item* I = &((page_item*) return_data)[i];

            I->distance = d_r_cpu[i].distance;
            I->ipd = E->getItemPointerData( (long) d_r_cpu[i].pos );
            I->searchPage = E->getPage( d_r_cpu[i].pos );
        }
        entry->next = slots[0];
        entry->returns = Np;
        entry->pos = 0;
        
        // Copy
        for(int i = 0; i < N-2; i++) {
            slots[i]->returns = Np;
            slots[i]->next = slots[i+1];
            slots[i]->pos = 0;
            
            for(int n = 0; n < Np; n++) {
                probe_entry* E = P->get( idx[ d_r_cpu[ (i+1)*Np + n].probe]  );
                page_item* I = &((page_item*) slots[i]->data)[n];

                I->distance = d_r_cpu[(i+1)*Np +n].distance;
                I->ipd = E->getItemPointerData( d_r_cpu[(i+1)*Np +n].pos );
                I->searchPage = E->getPage( d_r_cpu[(i+1)*Np +n].pos );
            }
        }

        // Copy last
        slots[N-2]->returns = a % Np;
        if(slots[N-2]->returns == 0)
            slots[N-2]->returns = Np;

        slots[N-2]->next = NULL;
        slots[N-2]->pos = 0;

        for(int n = 0; n < slots[N-2]->returns; n++) {
            probe_entry* E = P->get( idx[ d_r_cpu[ (N-1)*Np+n].probe]  );
            page_item* I = &((page_item*) slots[N-2]->data)[n];

            I->distance = d_r_cpu[(N-1)*Np+n].distance;
            I->ipd = E->getItemPointerData( d_r_cpu[(N-1)*Np+n].pos );
            I->searchPage = E->getPage( d_r_cpu[(N-1)*Np+n].pos );
        }
    }

    free_gpu_memory(d_a);

    // Cleanup
    free(d_r_cpu);
    free_gpu_memory(d_q);
    free_gpu_memory(d_r);

    return entry->returns;
#else
    return exec_query_cpu(entry, worker);
#endif
}

