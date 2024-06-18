all: sgemm

sgemm: main.o libsgemm_multigpu.so
	nvcc main.o -Xlinker -rpath,. libsgemm_multigpu.so -Xcompiler -fopenmp -o sgemm

main.o: main.cpp sgemm.h
	nvcc -O3 -c main.cpp -Xcompiler -fopenmp

libsgemm_multigpu.so: sgemm.cpp sgemm.h
	nvcc -O3 -shared -Xcompiler -fPIC sgemm.cpp -Xlinker -lcublas -lcudart -lnvToolsExt -o libsgemm_multigpu.so

clean:
	rm -f sgemm libsgemm_multigpu.so *.o
