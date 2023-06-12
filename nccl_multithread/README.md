## NCCL with Single Progress, Multi-thread, 1 Device per Thread ##

### Requirement
* 8 GPU

### Build ###
```sh
$ make
```

### Run ###
```sh
$ ./main
```

### Log ###
```sh
myRank 5: 28, 28, 28, 28, 28, 28, 28, 28, 28, 28,
myRank 4: 28, 28, 28, 28, 28, 28, 28, 28, 28, 28,
myRank 1: 28, 28, 28, 28, 28, 28, 28, 28, 28, 28,
myRank 3: 28, 28, 28, 28, 28, 28, 28, 28, 28, 28,
myRank 2: 28, 28, 28, 28, 28, 28, 28, 28, 28, 28,
myRank 0: 28, 28, 28, 28, 28, 28, 28, 28, 28, 28,
myRank 6: 28, 28, 28, 28, 28, 28, 28, 28, 28, 28,
myRank 7: 28, 28, 28, 28, 28, 28, 28, 28, 28, 28,
Success
```