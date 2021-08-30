# c-code

*Note: The file `sloth256_189.c` is heavily commented under `Documentation.md`. The reason for these comments are made inline is: it is a very low-level code and explaining the concepts are taking too much space, and rendering the code file a bit chaotic in the end.

## How To Build

### Linux (maybe possible with eGPU setting on MacOS as well)
0. Download this repo
1. Open your favorite terminal
2. `cd` into this directory (this repo/CUDA/c-code)
3. Run this command for compiling the project: `nvcc -gencode arch=compute_XX,code=sm_XX -use_fast_math -O2 sloth256_189.cu -o yeahBOI`
4. Run this command for running the executable: `./yeahBOI`

Explanation of the above command:
- `nvcc` is for the compiler
- `-gencode arch=compute_XX,code=sm_XX`, this is an optimization, and it is suggested. You have to replace `XX`'s with your compute capability. For example: if your compute capability is 8.6, you should replace `XX`'s with `86`.
- `-use_fast_math` another suggested optional optimization.
- `-O2` yet another optional optimization
- `sloth256_189.cu` the actual file we are compiling
- `-o yeahBOI` is optional but beneficial to one's sanity.

### Windows with Visual Studio
0. Download the repo
1. Create an CUDA project in Visual Studio (tested with version 2017 and 2019)
2. include the `sloth256_189.cu` in your project, but DO NOT INCLUDE `sloth256_189.c` in your project (it creates problem for Visual Studio). These two files just need to be in the same directory in your file system.
3. Run the program :)
