NVCC        = nvcc
NVCC_FLAGS  = -O3 -I/usr/local/cuda/include  
LD_FLAGS    = -lcudart -L/usr/local/cuda/lib64
EXE	    = gpu_mining_problem1 gpu_mining_starter
OBJ	    = gpu_mining_problem1.o gpu_mining_starter.o support.o

default: gpu_mining_problem1

gpu_mining_starter: gpu_mining_starter.o support.o
	$(NVCC) -o $@ $^ $(LD_FLAGS)

gpu_mining_problem1: gpu_mining_problem1.o support.o
	$(NVCC) -o $@ $^ $(LD_FLAGS)

# ------------------------------------------------------------------------------- #

# input file and size
FILE     = debug_1k.csv
SIZE	 = 20000

# number of trials to run
TRIALS_A = 5000000
TRIALS_B = 10000000

# output file suffix
OUT_A    = 20k_5m
OUT_B    = 20k_10m


run:
	make gpu_mining_starter
	./gpu_mining_starter $(FILE) $(SIZE) $(TRIALS_A) _out_$(OUT_A).csv _time_$(OUT_A).csv
	./gpu_mining_starter $(FILE) $(SIZE) $(TRIALS_B) _out_$(OUT_B).csv _time_$(OUT_B).csv


# ------------------------------------------------------------------------------- #

gpu_mining_starter.o: gpu_mining_starter.cu nonce_kernel.cu support.h
	$(NVCC) -c -o $@ gpu_mining_starter.cu $(NVCC_FLAGS) $(LD_FLAGS)

gpu_mining_problem1.o: gpu_mining_problem1.cu nonce_kernel.cu hash_kernel.cu support.h
	$(NVCC) -c -o $@ gpu_mining_problem1.cu $(NVCC_FLAGS) $(LD_FLAGS)

support.o: support.cu support.h
	$(NVCC) -c -o $@ support.cu $(NVCC_FLAGS) $(LD_FLAGS)

clean:
	rm -rf *.o $(EXE)
