EXTENSION = vector
EXTVERSION = 0.8.0

MODULE_big = vector
DATA = $(wildcard sql/*--*--*.sql)
DATA_built = sql/$(EXTENSION)--$(EXTVERSION).sql
# AERO change
OBJS = src/bitutils.o src/bitvec.o src/halfutils.o src/halfvec.o src/hnsw.o src/hnswbuild.o src/hnswinsert.o src/hnswscan.o src/hnswutils.o src/hnswvacuum.o src/ivfbuild.o src/ivfflat.o src/ivfinsert.o src/ivfkmeans.o src/ivfscan.o src/ivfutils.o src/ivfvacuum.o src/sparsevec.o src/vector.o src/gpuworker.o src/gpucache.o
HEADERS = src/halfvec.h src/sparsevec.h src/vector.h

TESTS = $(wildcard test/sql/*.sql)
REGRESS = $(patsubst test/sql/%.sql,%,$(TESTS))
REGRESS_OPTS = --inputdir=test --load-extension=$(EXTENSION)

# To compile for portability, run: make OPTFLAGS=""
OPTFLAGS = -march=native

# Mac ARM doesn't always support -march=native
ifeq ($(shell uname -s), Darwin)
	ifeq ($(shell uname -p), arm)
		# no difference with -march=armv8.5-a
		OPTFLAGS =
	endif
endif

# PowerPC doesn't support -march=native
ifneq ($(filter ppc64%, $(shell uname -m)), )
	OPTFLAGS =
endif

# For auto-vectorization:
# - GCC (needs -ftree-vectorize OR -O3) - https://gcc.gnu.org/projects/tree-ssa/vectorization.html
# - Clang (could use pragma instead) - https://llvm.org/docs/Vectorizers.html


# AERO change
PG_CFLAGS += $(OPTFLAGS) -ftree-vectorize -fassociative-math -fno-signed-zeros -fno-trapping-math -DAERO -g -march=native -O3
PG_CXXFLAGS += -std=c++11 -DAERO -g -O3

# Debug GCC auto-vectorization
# PG_CFLAGS += -fopt-info-vec

# Debug Clang auto-vectorization
# PG_CFLAGS += -Rpass=loop-vectorize -Rpass-analysis=loop-vectorize

all: sql/$(EXTENSION)--$(EXTVERSION).sql

sql/$(EXTENSION)--$(EXTVERSION).sql: sql/$(EXTENSION).sql
	cp $< $@

PG_CONFIG ?= pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)

# for Mac
ifeq ($(PROVE),)
	PROVE = prove
endif

# for Postgres < 15
PROVE_FLAGS += -I ./test/perl

# AERO
aero:	all
	g++ $(PG_CXXFLAGS) -march=native -shared -o vector.so src/bitutils.o src/bitvec.o src/halfutils.o src/halfvec.o src/hnsw.o src/hnswbuild.o src/hnswinsert.o src/hnswscan.o src/hnswutils.o src/hnswvacuum.o src/ivfbuild.o src/ivfflat.o src/ivfinsert.o src/ivfkmeans.o src/ivfscan.o src/ivfutils.o src/ivfvacuum.o src/sparsevec.o src/vector.o src/gpuworker.o src/gpucache.o
cuda:	all
	g++ $(PG_CXXFLAGS) -I$(includedir_server) -DGPU -fPIC -c -o src/gpucache.o src/gpucache.cpp
	nvcc $(PG_CXXFLAGS) -I$(includedir_server) --compiler-options '-fPIC -march=native -shared' -c src/ivfgpu.cu -o src/ivfgpu.o
	nvcc $(PG_CXXFLAGS) -Xcompiler="-march=native" -DGPU -shared -o vector.so src/bitutils.o src/bitvec.o src/halfutils.o src/halfvec.o src/hnsw.o src/hnswbuild.o src/hnswinsert.o src/hnswscan.o src/hnswutils.o src/hnswvacuum.o src/ivfbuild.o src/ivfflat.o src/ivfinsert.o src/ivfkmeans.o src/ivfscan.o src/ivfutils.o src/ivfvacuum.o src/sparsevec.o src/vector.o src/gpuworker.o src/ivfgpu.o src/gpucache.o	
sycl:	all
	g++ $(PG_CXXFLAGS)  -I$(includedir_server) -DGPU -fPIC -c -o src/gpucache.o src/gpucache.cpp
	icpx -std=c++17 -DAERO -g -O3 -march=native -fsycl -fsycl-targets=nvptx64-nvidia-cuda -I$(includedir_server) -fPIC -c src/ivfgpu.cpp -o src/ivfgpu.o
	icpx -std=c++17 -DAERO -g -O3 -march=native -fsycl -fsycl-targets=nvptx64-nvidia-cuda -DGPU -shared -o vector.so src/bitutils.o src/bitvec.o src/halfutils.o src/halfvec.o src/hnsw.o src/hnswbuild.o src/hnswinsert.o src/hnswscan.o src/hnswutils.o src/hnswvacuum.o src/ivfbuild.o src/ivfflat.o src/ivfinsert.o src/ivfkmeans.o src/ivfscan.o src/ivfutils.o src/ivfvacuum.o src/sparsevec.o src/vector.o src/gpuworker.o src/ivfgpu.o src/gpucache.o 
opencl:	all
	g++ $(PG_CXXFLAGS) -I$(includedir_server) -DGPU -fPIC -c -o src/gpucache.o src/gpucache.cpp
	icpx -std=c++17 -DAERO -g -O3 -march=native -fsycl -I$(includedir_server) -fPIC -c src/ivfgpu.cpp -o src/ivfgpu.o
	icpx -std=c++17 -DAERO -g -O3 -march=native -fsycl -DGPU -shared -o vector.so src/bitutils.o src/bitvec.o src/halfutils.o src/halfvec.o src/hnsw.o src/hnswbuild.o src/hnswinsert.o src/hnswscan.o src/hnswutils.o src/hnswvacuum.o src/ivfbuild.o src/ivfflat.o src/ivfinsert.o src/ivfkmeans.o src/ivfscan.o src/ivfutils.o src/ivfvacuum.o src/sparsevec.o src/vector.o src/gpuworker.o src/ivfgpu.o src/gpucache.o 
cuvs:	all
	g++ $(PG_CXXFLAGS) -I$(includedir_server) -DGPU -DCUVS -fPIC -c -o src/gpucache.o src/gpucache.cpp
	nvcc -DAERO -g -O3 --expt-relaxed-constexpr -I$(includedir_server) -DCUVS --compiler-options '-fPIC -march=native -shared' -c src/ivfgpu.cu -o src/ivfgpu.o -I/rhea/git/dlpack/include -I/rhea/git/raft/cpp/build/install/include -I/rhea/git/cuvs/cpp/build/install/include -DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE
	nvcc -DAERO -g -O3 --expt-relaxed-constexpr -Xcompiler="-march=native" -DGPU -DCUVS -shared -L/rhea/git/raft/cpp/build -L/rhea/git/cuvs/cpp/build -lraft -lcuvs_c -o vector.so src/bitutils.o src/bitvec.o src/halfutils.o src/halfvec.o src/hnsw.o src/hnswbuild.o src/hnswinsert.o src/hnswscan.o src/hnswutils.o src/hnswvacuum.o src/ivfbuild.o src/ivfflat.o src/ivfinsert.o src/ivfkmeans.o src/ivfscan.o src/ivfutils.o src/ivfvacuum.o src/sparsevec.o src/vector.o src/gpuworker.o src/ivfgpu.o src/gpucache.o
prove_installcheck:
	rm -rf $(CURDIR)/tmp_check
	cd $(srcdir) && TESTDIR='$(CURDIR)' PATH="$(bindir):$$PATH" PGPORT='6$(DEF_PGPORT)' PG_REGRESS='$(top_builddir)/src/test/regress/pg_regress' $(PROVE) $(PG_PROVE_FLAGS) $(PROVE_FLAGS) $(if $(PROVE_TESTS),$(PROVE_TESTS),test/t/*.pl)

.PHONY: dist

dist:
	mkdir -p dist
	git archive --format zip --prefix=$(EXTENSION)-$(EXTVERSION)/ --output dist/$(EXTENSION)-$(EXTVERSION).zip master

# for Docker
PG_MAJOR ?= 17

.PHONY: docker

docker:
	docker build --pull --no-cache --build-arg PG_MAJOR=$(PG_MAJOR) -t pgvector/pgvector:pg$(PG_MAJOR) -t pgvector/pgvector:$(EXTVERSION)-pg$(PG_MAJOR) .

.PHONY: docker-release

docker-release:
	docker buildx build --push --pull --no-cache --platform linux/amd64,linux/arm64 --build-arg PG_MAJOR=$(PG_MAJOR) -t pgvector/pgvector:pg$(PG_MAJOR) -t pgvector/pgvector:$(EXTVERSION)-pg$(PG_MAJOR) .
