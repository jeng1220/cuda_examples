# CUDA Unified Memory Oversubscription #

## Build ##
```sh
$ nvcc foo.cu -o foo
```

## Run ##
```sh
$ foo <itr, integer> <size, integer> <use 1)cudaMemset or 0) memset, integer>
$ foo 85 1000 1
```
In parallel, open another terminal to observe system memory footprint.
```sh
$ htop
# or
$ free
```

## Log ##
```sh
$ ./foo 85 1000 1
85 times allocations
allocate 1048576000 B at a time
allocate 89128960000 B totally
use cudaMemset to access the buffers
free: 24991170560, total: 25229197312
free: 23917428736, total: 25229197312
free: 22843686912, total: 25229197312
...
free: 2843148288, total: 25229197312
free: 1903624192, total: 25229197312
free: 829882368, total: 25229197312
# start oversubscription
free: 1507328, total: 25229197312
free: 1507328, total: 25229197312
free: 1507328, total: 25229197312
...
```

The system memory footprint:
```sh
$ free

               total        used        free      shared  buff/cache   available
Mem:        65500104    64305780      606620       10124      587704      427724
Swap:              0           0           0
```