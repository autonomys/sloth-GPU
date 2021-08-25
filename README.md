# sloth-GPU
GPU implementation of sloth (CUDA and OpenCL)


## CUDA

- Old implementation (high-level implementation using Montgomery and Legendre) can be found in `CUDA/High-Level(old)`
- `Experimentation-ptx` includes test files for experimenting with the current ptx implementation.
- The files `encode_ptx.h` and `ptx_text.cu` are the current PTX implementation of Sloth Encoding.

*For building the CUDA code in your machine, please refer to the `README.md` in the `CUDA` folder.*

## OpenCL
- Includes the necessary OpenCL drivers (these may already be present in your machine, but for building purposes, the location of these files matter, thus they are included in the repo) under the folder `openCL/OpenCL`.
- There are currently 2 implementations, one for testing the correctness, one for throughput benchmarking:
    - Correctness (1 thread):
        - `main.cpp`
        - `hello.cl`
    - Throughput (1024 * 256 threads)
        - `throughput_main.cpp`
        - `throughput_kernel.cl`
- `sloth256_189.c` is the file where actual Sloth-Encoding's code resides. Basically, `.cpp` file is the boilerplate OpenCL code, that calls the kernel (`.cl` file), and the kernel is including the code inside `sloth256_189.c`.

*For building the OpenCL code in your machine, please refer to the `README.md` in the `OpenCL` folder.*


## Research

1. Removal of Jacobi (check the result by squaring it, if it is incorrect, negate it)
2. Perform modular multiplication via Addition Chain (Montgomery not required)
3. Using fast-reduction for modulo operations
4. Other required operations are in the `Research/operations-tree.md`

## Research (old)

*Required Operations for the sqrt-permuatation (high-level only):*
1. Jacobi (legendre can be a better fit for our needs)
2. Modular Exponentiation (choose the best Montgomery for this)
3. IsOdd (to be implemented in uint_t libraries directly)
4. IsEven (to be implemented in uint_t libraries directly)
5. Subtract (to be implemented in uint_t libraries directly)