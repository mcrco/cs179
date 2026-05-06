# Part 1

Took me about 2 hours. Had to brush up on FFT + why we use for convolution, and then ran into some problems implementing since I was forgetting to do things like

- scale by 1/padded_length
- allocate enough memory for complex, not float
- memset with size of complex, not float

# Part 2

## Max Kernel

- shared memory accumulation per block
- first, go through all the values a thread is "responsible" for in the case that # threads < padded length
  - coalesced reads by using stride = gridDim.x * blockDim.x
- then, take the max of each of the first N/2 values with the value that's N/2 to the right
  - sequential threads activate => reduces bank conflicts when accessing shared memory
  - calculating/storing max using shared memory instead of global reads/writes => much faster
- take atomic max of max_abs_val and each of the values in the first element of all blocks' shared memory

## Divide Kernel

- literally just do coalesced reads of each value, divide it by max, then do coalesced write.

# Part 3

All relevant images are in `img/`.

## ProdScale Kernel

The prod scale kernel should be dominated by memory since we are loading a bunch of device memory and then performing individual multiplication. This is apparent in both the speed of light chart (memory dominates) and the compute workload analysis chart, where we see LSU dominate in pipe utilization. We should also see high cache hit rates since we are doing coalesced reads/writes.

## Max Kernel

The main thing I was looking for with the max kernel was low bank conflicts and warp divergence since the key speedup of my max kernel was doing sequential accesses in shared memory, and bank conflicts/warp divergence would be the main slowdowns. In the max kernel memory analysis chart, the number of requests to and from the shared memory are roughly equal, which should show that the number of sequential accesses is low (otherwise responses >> requests). In the warp statistics, it said 26.31 average not predicated off threads per warp, which should mean that ~26.31/32 threads were active, which seems like a good thing to me?

## Divide Kernel

Once again I just looked to check that memory was the main overhead here since all we do is global memory read + divide + global memory write. And the speed of light graph did look like that.
