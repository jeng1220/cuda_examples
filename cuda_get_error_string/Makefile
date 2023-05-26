out:=main

all: main.cpp
	nvcc -std=c++11 main.cpp -lcudart -o $(out)

.PHONY: clean
clean:
	rm -f $(out)
