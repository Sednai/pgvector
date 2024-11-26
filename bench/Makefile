
all:
	nvcc --compiler-options '-D_PG_ORCL_ -DPGXC -DXCP -g -fuse-linker-plugin -fno-omit-frame-pointer -march=native' bench.cu -o bench ../src/ivfgpu.o -L/data/TBase/install/TBase_bin_v2.0/lib -I/data/pgvector/src -I/data/TBase/install/TBase_bin_v2.0/include/postgresql/server
