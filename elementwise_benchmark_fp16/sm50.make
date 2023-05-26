

CC:=nvcc
exe:=main
obj:=main.o cast.o add.o util.o
#flag:=-g -G -O0
flag:=-O3 -arch=sm_50

all:$(obj)
	$(CC) -o $(exe) $(obj)  
%.o:%.cu
	$(CC) -c -std=c++11 $(flag) $^ -o $@
%.o:%.cpp
	$(CC) -c -std=c++11 $(flag) $^ -o $@
.PHONY:clean
clean:
	rm -rf $(obj) $(exe)



