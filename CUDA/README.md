# Parallel CUDA code for Sloth Encode Implementation

## What Does This Library Do?

As Subspace Labs, we are utilizing [Sloth](https://eprint.iacr.org/2015/366.pdf) as our encoding/decoding scheme in our protocol.

Encoding of Sloth (and the most other alternatives) is intentionally slow, because of security measures. And because of this property, `plotting` phase requires a considerable amount of time to finish (required time is linear with the size of the space being dedicated to the protocol). To enhance user-experience, we decided to take advantage of the commonly found, massively parallel hardwares (GPUs).
This repository is the parallelized version of the Sloth's Encoding scheme, specifically for the NVidia GPUs.

CUDA being the programming language for NVidia GPUs, and PTX being the low-level (similar to assembly) code for CUDA. This repository includes optimized PTX code (performing 3x better than the high-level CUDA), that can run on NVidia GPUs, and providing on average 100-300x speed up (depending on the GPU model).

In other words: Reducing the duration of the `plotting` phase from possibly ***1 week*** to ***30 minutes***.

Below code is just the Proof of Concept for NVidia GPUs, and will be integrated to our Protocol as the next step.

### Okay, but how are we achieving this remarkable speed-up?

There are 2 things:
1. GPUs provide thousands of cores, compared to 4-32 cores of CPUs. However, GPU cores are much weaker than CPU ones, so it is not a direct comparison like: "if GPU has thousands of cores, the program on the GPU should be thousands of time faster". Unfortunately not, usually the expectation is 50-100x speed up, and this assumption is only valid if the algorithm is highly parallelizable.
2. Math. Yes, that's right, by optimizing the complex and computation-wise really heavy operations (thanks to mathematics), we were able to achieve this speed up. And actually this part is responsible from the 30x speed up.

### The Most Important Mathematical Tricks For This Repo
1. We are using fast-reduction for Mersenne Primes (https://eprint.iacr.org/2017/437.pdf), instead of Montgomery Reduction.
2. We are not checking if the square root exists for that number with Legendre (originally Jacobi).

## How to Build

### Linux, Windows (maybe possible with eGPU setting on MacOS as well)
0. Download this repo
1. Open your favorite terminal
2. `cd` into this directory (this repo/CUDA)
3. Run this command for compiling the project: `nvcc -gencode arch=compute_XX,code=sm_XX -use_fast_math -O2 ptx_test.cu -o yeahBOI`
4. Run this command for running the executable: `./yeahBOI`

Explanation of the above command:
- `nvcc` is for the compiler
- `-gencode arch=compute_XX,code=sm_XX`, this is an optimization, and it is suggested. You have to replace `XX`'s with your compute capability. For example: if your compute capability is 8.6, you should replace `XX`'s with `86`.
- `-use_fast_math` another suggested optional optimization.
- `-O2` yet another optional optimization
- `ptx_test.cu` the actual file we are compiling
- `-o yeahBOI` is optional but beneficial to one's sanity.

### Windows with Visual Studio
0. Download the repo
1. Create an CUDA project in Visual Studio (tested with version 2017 and 2019)
2. include the `encode_ptx.h` and `ptx_test.cu` to your project.
3. Run the program :)
