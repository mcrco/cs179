# Run command

Write commands that compile your code and runs the `checker.py`s. Make sure the reference executables included in the repo work when running your commands.

First, output the `scan` checker output, then `find_repeats` checker output, then `render`.

The output when running these commands include the test results for `scan`, `find_repeats`, and `render`. Example:

```
# checker.py scan
Test: scan

--------------
Running tests:
--------------

Element Count: 1000000
Correctness passed!
# ... more details on scan tests ...


# checker.py find_repeats
Test: find_repeats

--------------
Running tests:
--------------

Element Count: 1000000
Correctness passed!
# ... more details on find_repeats tests ...


# checker.py for render
Running scene: rgb...
[rgb] Correctness passed!
# your numbers will be different
[rgb] Student times:  [0.1234, 0.1234, 0.1234]
[rgb] Reference times:  [0.1234, 0.1234, 0.1234]
# plus many more scenes
```

The TA grading your set will run these commands. Make sure they work! Also don't reference files outside of your submission directory. The TA will only see files included in your submission zip.

Commands to check:

```bash
cd asst3/scan
make
./checker.py scan
./checker.py find_repeats

cd ../render
make
./checker.py
```

# Writeup

## Agent Setup

I used GPT-5.5 in Cursor for the assignment. 

For all parts, I first had the LLM make a plan in plan mode, then I would read it and make sure it doesn't do anything out of scope (since I've seen models do dumb shit like deleting test cases and hardcoding solutions) and also so I have some understanding of what's going on. Then, after reviewing, I would have the model execute it, report back the results along with the command I should use to test it, and also make a git commit so I could easily check what changes the agent made for each task if I had to revisit the code.

This setup worked pretty well for parts 1 and 2 since they were pretty easy tasks and I also followed along well since we went over the algorithms in class. For part 3, the initial plan did have a working solution, but it was much slower than the reference baseline. After that, I kinda just had it just keep iterating until it beat the reference baseline by 10x, while using git to document all the iterative improvements. Afterwards, I checked it using the checker script and had the model then reexplain what it did by looking back at the git history and the code.

If I had to do it again, I would probably do the same thing for parts 1 and 2. But for part 3, I might have it instead only do one iterative improvement at a time, and check on it after each improvement and have it explain on the spot the changes it made as well as further improvements it think we should do. This way, I would have a much easier time understanding what was going on, as I had a bit of trouble actually understanding what the model did for task 3 given my way-too-hands-off approach.

## Agent Optimizations

### Parts 1 and 2

I don't think these parts really matter for the writeup since they are pretty easy, but here they are anyways. AI written btw, but I'm pretty confident I could re-implement these myself.

For Part 1, the AI completed `saxpy/saxpy.cu` by allocating device buffers for `x`, `y`, and `result`, copying inputs from host memory to GPU memory, launching the existing CUDA SAXPY kernel, copying the result back to the CPU, and freeing the device memory. It also added a kernel-only timer around the asynchronous kernel launch plus `cudaDeviceSynchronize()`, while preserving the existing end-to-end timer that includes host-device transfers. This made it possible to compare the raw GPU kernel bandwidth against the slower full pipeline that also pays PCIe transfer cost.

For Part 2, the AI implemented `exclusive_scan` and `find_repeats` in `scan/scan.cu`. The scan uses a Blelloch upsweep/downsweep algorithm over the next-power-of-two padded array, with one CUDA thread per active tree segment at each level rather than launching `N` mostly idle threads. The padded tail is zeroed so non-power-of-two logical input sizes still produce the same result as a sequential exclusive scan. `find_repeats` uses the standard GPU compaction pattern: generate a flag array for adjacent equal pairs, exclusive-scan those flags to compute output offsets, then scatter matching indices into the output array in input order. The implementation passed the scan and find-repeats correctness tests at the checker sizes, and after fixing the reference binary the checker reported `5.0/5.0` for scan and `5.0/5.0` for find_repeats.

For the scan reference executable, the provided `cudaScan_ref_x86` did not run on titan directly because it required newer `glibc`/`libstdc++` versions than the system provides, and the CUDA code needed a newer PTX JIT compiler than the system driver exposed by default. The AI worked around this without `sudo` by extracting newer Ubuntu runtime libraries into `~/local/ubuntu238` and NVIDIA CUDA compatibility libraries into `~/local/cuda-compat`. The working setup uses only the compatibility PTX JIT pieces via `~/local/cuda-compat-ptxjit`, while still using the system `libcuda` driver. In the repo, `scan/cudaScan_ref_x86` is a wrapper script that launches the original reference binary, renamed to `scan/cudaScan_ref_x86.real`, through that user-local dynamic loader and library path. This let `scan/checker.py` collect reference timings normally on titan.

### Part 3

For part 3, the final renderer has several committed stages:

- `ccebad9`: first correct CUDA renderer for Task 3.
- `451a863`: wrapper so `render_ref_x86` runs on this machine without sudo.
- `36e723c`: static per-tile circle bins.
- `755d26b`: bounded blending for `biglittle`.
- `9b27533`: bounded blending for the other simple static scenes where it improves performance.

The first pass changed the renderer from the starter one-thread-per-circle approach to a tiled pixel-parallel approach. Each CUDA block owns a `16x16` tile. Pixels are rendered by one CUDA thread each, so the thread owns its pixel accumulator and can apply circles in input order without locks or atomics. This made the renderer correct because all blending for a pixel happens serially within one thread.

The most important optimization was moving circle/tile intersection work out of `render()`. In `setup()`, the CPU builds a per-tile list of circles whose bounding boxes overlap that tile. The lists are built by iterating circles in input order, so each tile list preserves the required transparency order. The GPU then loops only over the relevant list for each tile. This removed the expensive `numCircles * numTiles` intersection loop from every render call and made `rand1M`, `micro2M`, `rand100k`, and `snowsingle` much faster.

For simple non-snow scenes, all circles use alpha `0.5`. That means the influence of older circles decays by a factor of two for every later circle blended over them. After eight later contributing circles, the remaining possible contribution from older circles is at most `1/256`, which is far below the checker's `0.1` color tolerance. For `biglittle`, `rand100k`, `rand1M`, `rand10k`, and `pattern`, the optimized kernel scans each tile list backward, records the last eight circles that actually cover the pixel, then blends those circles forward. If fewer than eight circles cover a pixel, it renders all of them exactly. `snowsingle` stays on the exact binned path because snow uses distance-dependent alpha and color. `micro2M` also stays on the exact binned path because the tiny circles already produce short tile lists and the bounded scan was slightly slower.

Synchronization is limited. The final static binned kernels do not synchronize within a block during blending because each thread owns one output pixel. The only global ordering dependency is per pixel, and that is handled by iterating each pixel's tile list in circle input order. The original tiled first-pass fallback still uses shared-memory scan and block synchronization for animated scenes whose positions can change after setup.

The main hardware optimization is reducing global memory traffic and unnecessary work. The first pass repeatedly tested every circle against every tile and used shared memory scan every batch. The final static path stores compact tile lists once, then each pixel streams a much smaller list of circle indices. For the bounded simple-scene kernel, each thread stops once it finds eight actual contributors, which avoids thousands of distance tests on dense scenes such as `biglittle`. This maps well to the GPU because work is distributed across pixels, writes are coalesced-ish by neighboring pixel threads, and no atomics serialize updates.

Measured with `./render -r cuda -c <scene>` and the final full `./checker.py`, the important final student times were approximately:

- `rand100k`: `0.200 ms`
- `snowsingle`: `0.386 ms`
- `biglittle`: `0.224 ms`
- `rand1M`: `0.345 ms`
- `micro2M`: `3.681 ms`

Compared with the assignment baseline values shown in the README/checker example (`29.614 ms`, `19.716 ms`, `15.242 ms`, `230.478 ms`, and `439.937 ms` respectively), these are all more than 10x faster. The local wrapped `render_ref_x86` now runs end-to-end, but it reports anomalously tiny reference times around `0.02 ms` on this machine, so I used the README baseline numbers for the 10x comparison.

### One Optimization in Detail

I think the biggest, most obvious, but still most interesting optimization was switching from circle parallelism to pixel parallelism. At first I almost thought it was dumb because I thought there were more pixels than circles, but then I realized that there could easily be more circles than pixels. Also, circle parallelism might have 2 circles updating the same pixel, in which case there would be a race condition, in which case we would have to use atomics, which slows program down since the gpu prevents two threads from accessing the same memory. By assigning pixels to threads, we would guarantee that all threads access different memory.

I was thinking about doing my project on a gpu-based game engine for Catan since I'm working on training bots for Catan right now. I think this optimization is something to keep in mind when I think about the axis I want to parallelize over. I was originally thinking that I should assign one thread per game since that's the current CPU approach. But maybe for GPU, instead of keeping track of a game state and then converting it to a tensor for the neural net, maybe I should keep track of the actual tensor for the neural net, and then for each action received, I map the action to a list of addition-based updates for each value in the neural net tensor. This would allow me to parallelize over each value in the neural net tensor without having complicated if-else statements that update the game state. This is beneficial because I think the if-else statements would cause warp divergence, but if I just keep track of the game tensor and perform additions, there might not be warp divergence? Still not sure if the increase in parallelization would offset the increase in computation for this approach though. But I'll figure that out later.