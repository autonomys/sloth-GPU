# Parallel CUDA code for Sloth Encode Implementation

## How to Build

### Linux (maybe possible with eGPU setting on MacOS as well)
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
