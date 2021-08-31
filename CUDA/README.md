# sloth-GPU
GPU implementation of Sloth's internal permutation (CUDA)


## CUDA

- Old implementation (high-level implementation using Montgomery and Legendre) can be found in by rolling back to previous commits.
- `Experimentation-ptx` includes test files for experimenting with the current ptx implementation (can be found in by rolling back to previous commits).
- The files `encode_ptx.h` and `ptx_text.cu` are the current PTX implementation of Sloth Encoding.

*For building the CUDA code in your machine, please refer to the `README.md` in the root folder.*


## Research

1. Removal of Jacobi (check the result by squaring it, if it is incorrect, negate it)
2. Perform modular multiplication via Addition Chain (Montgomery not required)
3. Using fast-reduction for modulo operations
4. Other required operations are in the `Research/operations-tree.md`
