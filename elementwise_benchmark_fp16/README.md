## Elementwise Benchmark with float16 ##

### build ###
```sh
$ make -j4 # for Volta GPU
```

### run ###
```sh
$ ./main
$ ./main <number of elements> <iterations>
$ ./main 1000000 10 # run 1M elements 10 times
```
