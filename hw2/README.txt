1.1. 

FMA is two operations. Given that the RTX A5000 has a theoretical performance 
of 27.77 TFLOPS for fp32 operations, it should be able to perform 27.77 * 
10^12 / 2 = 13.885 * 10^12 FMAs per second.

1.2.a.

The warp consists of chunks of 32 threads with consecutive flattened indices.
Flattened indices can be calculated by threadIdx.x + threadIdx.y + blockDim.x.
If we calculate `idx = threadIdx.y + blockSize.y * threadIdx.x`, we can notice
that for the first consecutive 32 threads in terms of flattened indices, we 
have threadIdx.y = 0 and threadIdx.x = 0, 1, 2, ..., 31. Then, idx = 0, 32, 64,
..., 992. Since they are all the same mod 32, it doesn't diverge for the first 
warp. The same applies for all subsequent warps because we just increase 
threadIdx.y by 1, which leads to increasing everything by 1, and since we add a 
multiple of 32, all indices in every warp are just threadIdx.y mod 32.

Thus, it doesn't diverge.

1.2.b. 

Since every thread only executes threadIdx.x times, threads complete at 
different times, but have to wait for the last to finish. Thus, it does diverge.

1.3.a.

With blockSize.x = 32, every warp has thread 32k + i for i = 0, 1, 2, ..., 32
and accesses indices x + 32 * y for x = 0, 1, 2, ..., 32.

Since data is 128 byte aligned, and each float is 4 bytes, each warp accesses a
contiguous 128 bytes/1 cache line. So yes, the write is coalesced and accesses a 
total of 32 cache lines since there are 32 warps.

1.3.b. 

This write is not coalesced. The warp now accesses indices 32i + k for some k and 
i = 0, 1, 2, ..., 32. That means that each thread accesses a different cache line 
since 32 floats in between each data point => 128 bytes in between, so each warp
accesses 32 different cache lines. But since each warp is accessing the same 32 
cache lines, the total number of cache lines is still 32.

1.3.c.

This write is not fully coalesced since we shift everything by 1 index/4 bytes. 
It's pretty much like part a, but the last thread of each warp ends up indexing 
into a different cache line since it's 1 over the next multiple of 32 compared to 
the first thread. Thus, it's 2 cache lines for each warp, but since adjacent warps 
have adjacent cache lines, we can reuse the new cache line from the last element of
the last warp, so the total is 33.

1.4.a.

Adjacent threads have same y and different x.

There are 3 memory access indices in the code:

1. i + 32 * j = x + 32 * y
2. i + 32 * k = x + 32 * k
3. k + 128 * j = k + 128 * y
4. i + 32 * (k + 1) = x + 32 * (k + 1)
5. (k + 1) + 128 * j = k + 128 * y

For 1, since the indices are consecutive, and consecutive indices in shared memory 
are sent to different banks mod 32, and x and y both range from 0 to 31, there are
no bank conflicts.

For 2, it's similar since we multiply k by 32, and since each thread in a warp has
a different x % 32, the banks, which are also x % 32, will be different for each 
thread. Thus, there are no bank conflicts.

For 3, since k and j are the same for all threads in a warp at every step, the value
in the shared bank gets broadcast to all 32 threads, so no conflict again.

4 and 5 have the same logic as 2 and 3: no bank conflicts.

Since none of the memory accesses have bank conflicts, the code overall does not have
bank conflits.

1.4.b.

1. LDS R0, [i + 32 * j]
2. LDS R1, [i + 32 * k]
3. LDS R2, [k + 128 * j]
4. FMA R0, R1, R2, R0
5. STS [i + 32 * j], R0

6. LDS R0, [i + 32 * j]
7. LDS R1, [i + 32 * (k + 1)]
8. LDS R2, [(k + 1) + 128 * j]
9. FMA R0, R1, R2, R0
10. STS [i + 32 * j], R0

1.4.c. 

4 must come after 1, 2, 3 since otherwise there is no data.
Same thing with 9 coming after 6, 7, 8
5 must come after 4 otherwise the value of R0 isn't updated for the write.
Same for 10 and 9.

Instruction 6 depends on instruction 5 since we want to use a stale value for output[i + 32 * j]
(before it got added in the previous line).

1.4.d

```cuda
int i = threadIdx.x;
int j = threadIdx.y;
float orig_val = output[i + 32 * j];
for (int k = 0; k < 128; k += 2) {
    orig_val += lhs[i + 32 * k] * rhs[k + 128 * j];
    orig_val += lhs[i + 32 * (k + 1)] * rhs[(k + 1) + 128 * j];
}
output[i + 32 * j] = orig_val;
```

By storing the output[i + 32 * j] into an intermediate register, I remove the need
to store and reload it back into memory in L6. 

1.4.e.

I think if we load all of the indexed shared memory values into floats at the start, then 
the GPU requests them all in parallel, which would parallelize the long time taken to get
each value. Then, if we do all the computation at the end, we essentially save the time it
takes to load something from shared memory?

2.

Optimal kernel strategies:

- unrolling of iteration over 4 floats per thread
- parallelize global memory access for input by loading all 4 input floats into register.

Output:

```
Time limit for this program set to 10 seconds
Size 512 naive CPU: 1.494208 ms
Size 512 GPU memcpy: 0.017792 ms
Size 512 naive GPU: 0.124928 ms
Size 512 shmem GPU: 0.036864 ms
Size 512 optimal GPU: 0.030720 ms

Size 1024 naive CPU: 7.228416 ms
Size 1024 GPU memcpy: 0.022080 ms
Size 1024 naive GPU: 0.049152 ms
Size 1024 shmem GPU: 0.026624 ms
Size 1024 optimal GPU: 0.027648 ms

Size 2048 naive CPU: 47.113186 ms
Size 2048 GPU memcpy: 0.080512 ms
Size 2048 naive GPU: 0.174080 ms
Size 2048 shmem GPU: 0.057344 ms
Size 2048 optimal GPU: 0.057344 ms

Size 4096 naive CPU: 312.953186 ms
Size 4096 GPU memcpy: 0.312160 ms
Size 4096 naive GPU: 0.667648 ms
Size 4096 shmem GPU: 0.195584 ms
Size 4096 optimal GPU: 0.195584 ms
```

3.1.

Images in `img/` folder.

3.2.

Global memory bandwith:

- naive: 142.38 GB/s, 18.56%
- shmem: 690.13 GB/s, 89.97%
- optimal: 658.78 GB/s, 89.77%

3.3

Based on the output, I see the global memory reads being clumped together, 
so I'm going to assume that the compiler unrolled the for loop for me.

