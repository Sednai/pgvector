
#include <cstdio>
#include <iostream>
#include <list>
#include <algorithm>
#include <bits/stdc++.h>
#include "vector.h"
#include "storage/relfilenode.h"
#include "storage/itemptr.h"

#include "ivfgpu.h"
#include "gpucache.h"

#undef qsort
#include <stdlib.h>

using namespace std;

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

__inline__ float squared_eucl_dist(float* X, float* Y, int N) {
    float D = (X[0] - Y[0])*(X[0] - Y[0]);
    for(int i = 1; i < N; i++) {
        D += (X[i] - Y[i])*(X[i] - Y[i]);
    }
    return D;
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


class probe_entry {
    float* vectors_gpu;   
    unordered_map<long, item> vectors_cpu;
    
    public:
        int dim;
        float* probe;

        void insert_vector(Vector* v, int page, ItemPointerData ipd) {
            item I;
            I.vector = new float[v->dim];
            I.page = page;
            I.ipd = ipd;

            memcpy(I.vector, v->x, v->dim*sizeof(float));
            
            vectors_cpu[vectors_cpu.size()] = I;
        }
 
        long size() {
            return vectors_cpu.size();
        }
        
        const item* getItemData(long idx) {
            return &vectors_cpu[idx];
        }

        float* getGPUdata() {
            return vectors_gpu; 
        }

        void storeOnGPU() {
            if(vectors_gpu == nullptr) {
                // Init ordinary cuda memory
                init_gpu_memory((void**) &vectors_gpu, size() * dim * sizeof(float) );
                
                // Copy
                for(long i = 0; i < size(); i++) {
                    item I = vectors_cpu[i];
                    copy_memory_to_gpu(&vectors_gpu[i*dim], I.vector, dim*sizeof(float));
                }
            }
        }

        ~probe_entry() {
            delete[] probe;
            if(vectors_gpu != nullptr) {
                free_gpu_memory(vectors_gpu);
            }
            
            for(auto &it : vectors_cpu) {
                delete vectors_cpu[it.first].vector;
            }
            
        }   
};

class probes {
    vector<probe_entry*> PROBES;
    public:
          
        int insert(Vector* in) {
            probe_entry *PE = new probe_entry();
            PE->dim = in->dim;
            PE->probe = new float[in->dim];
            memcpy(PE->probe,in->x,in->dim*sizeof(float));

            PROBES.push_back( PE );    

            return PROBES.size()-1;
        }

        void insert_vector(int probenumber, Vector* x, int page, ItemPointerData ipd) {
            PROBES[probenumber]->insert_vector(x, page, ipd);
        }

        probe_entry* get(int p) {
            return PROBES[p];
        }

        vector<int> get_ordered_probes_idx(float* q) {
            
            vector<int> idx(PROBES.size());
            iota(idx.begin(), idx.end(), 0);
            
            vector<probe_entry*> *P = &PROBES;
            sort(idx.begin(), idx.end(),[&P,&q](int i1, int i2) { return squared_eucl_dist(P[0][i1]->probe,q,P[0][i1]->dim) < squared_eucl_dist(P[0][i2]->probe,q,P[0][i2]->dim); });
            
            return idx;
        }


        int size() {
            return PROBES.size();
        }

        long numvectors() {
            long N = 0;
            for(int i = 0; i < PROBES.size(); i++) {
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

        int insert(Relation node, Vector* c) {
        
            probes *P;
            if(!contains(node)) {
                P = new probes();
                MAP[node] = P;
            } else {
                P = MAP[node];
            }

            return P->insert(c);
        }
        
        probes* get(Relation node) {
            return MAP[node];
        }

        void insert_vector(Relation node, int probenumber, Vector* v, int page, ItemPointerData ipd ) {
            probes *P = MAP[node];
            P->insert_vector(probenumber, v, page, ipd);
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

int new_probe(RelFileNode node, Vector* c) {
    Relation R = {node};

    return CACHE->insert(R, c);

}

void insert(RelFileNode node, int probenumber, Vector* c, int page, ItemPointerData ipd) {
    Relation R = {node};

    return CACHE->insert_vector(R, probenumber, c, page, ipd);
}

void logsize() {
    CACHE->logsize();
}

page_list exec_query_cpu(RelFileNode node, int Np, int op, float filter, float* q, int dim) {
    Relation R = {node};
    
    // Get probes for relation
    probes *P = CACHE->get(R);

    vector<int> idx = P->get_ordered_probes_idx(q);
    
    page_list RET;
    RET.data = (page_item*) malloc(sizeof(page_item) * 100000);
    RET.length = 0;
    RET.max_length = 100000;

    for(int i = 0; i < min(idx.size(), (size_t) Np); i++) {
        // Get entry
        probe_entry* E = P->get(idx[i]);
        long L = E->size();
        
        adjust_buffer(&RET, L);
        
        // Loop over vectors    
        for(long j = 0; j < L; j++) {
            const item *D_I = E->getItemData(j);
            float dist = squared_eucl_dist(q,D_I->vector,dim);

            // Filter
            if(!filter_func(dist,filter,op))
                continue;

            // Build return item
            page_item* I = &RET.data[RET.length];
			I->distance = dist;
            I->ipd = D_I->ipd;
			I->searchPage = D_I->page;
			RET.length++;
        }
    }

    // Sort
    qsort(RET.data, RET.length, sizeof(page_item), compare_pi);

    return RET;
}


page_list exec_query_gpu(RelFileNode node, int Np, int op, float filter, float* q, int dim) {
    Relation R = {node};
    
    // Get probes for relation
    probes *P = CACHE->get(R);

    vector<int> idx = P->get_ordered_probes_idx(q);
    
    page_list RET;
    RET.data = (page_item*) malloc(sizeof(page_item) * 100000);
    RET.length = 0;
    RET.max_length = 100000;

    // Store query vector on GPU
    float* d_q;
    init_gpu_memory((void**) &d_q, dim*sizeof(float) );       
    copy_memory_to_gpu(d_q, q, dim*sizeof(float));

    for(int i = 0; i < min(idx.size(), (size_t) Np); i++) {
        // Get entry
        probe_entry* E = P->get(idx[i]);
        
        // Store data on gpu if not stored yet
        E->storeOnGPU();

        long L = E->size();
          
        float* d_r;
        init_gpu_memory((void**) &d_r, L*sizeof(float) );
   
        // Calc distances
        calc_squared_distances_gpu_euclidean_nosharedmem(E->getGPUdata(), d_q, d_r, L, dim);
    
        // Copy back
        float* d_r_cpu = (float*) malloc(L*sizeof(float));
        copy_memory_to_cpu(d_r_cpu, d_r, L*sizeof(float));
        free_gpu_memory(d_r);

        adjust_buffer(&RET, L);

        // Build
        for(long j = 0; j < L; j++) {
            const item *D_I = E->getItemData(j);
        
            // Filter
            if(!filter_func(d_r_cpu[j],filter,op))
                continue;

            // Build return item
            page_item* I = &RET.data[RET.length];
			I->distance = d_r_cpu[j];
            I->ipd = D_I->ipd;
			I->searchPage = D_I->page;
			RET.length++;
        }

        free(d_r_cpu);
    }
    
    // Cleanup
    free_gpu_memory(d_q);
    
    // Sort
    qsort(RET.data, RET.length, sizeof(page_item), compare_pi);

    return RET;
}
