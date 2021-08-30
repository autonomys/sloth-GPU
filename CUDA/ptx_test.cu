#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <bitset>
#include <string>

#include "encode_ptx.h"

using namespace std;


#define CUDA_FATAL(expr) do {				\
    cudaError_t code = expr;				\
    if (code != cudaSuccess) {				\
        cerr << #expr << "@" << __LINE__ << " failed: "	\
             << cudaGetErrorString(code) << endl;	\
	exit(1);					\
    }							\
} while(0)



int main()
{
// creating problem with nsight, can remove this part - BEGIN
    cudaDeviceProp prop;
    CUDA_FATAL(cudaGetDeviceProperties(&prop, 0));
    cout << prop.name << endl;
    cout << "Capability: " << prop.major << "." << prop.minor << endl;
    cout << "Clock rate: " << prop.clockRate << "kHz" << endl;
    cout << "Memory clock rate: " << prop.memoryClockRate << "kHz" << endl;
    cout << "L2 cache size: " << prop.l2CacheSize << endl;
    cout << "Shared Memory: " << prop.sharedMemPerBlock << endl;
    // creating problem with nsight, can remove this part - END


    // instead of below, we can give any number to blockSize and minGridSize like this:
    /*
    int blockSize = 256;
    int minGridSize = 30;
    */

    int blockSize;      // The launch configurator returned block size
    int minGridSize;    // The minimum grid size needed to achieve the
                        // maximum occupancy for a full device launch
    CUDA_FATAL(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
                                                  encode_ptx_test));  // creating problem with nsight, can remove this part also

    cout << "kernel<<<" << minGridSize << ", " << blockSize << ">>>, \n";  // shows the parameters for max-occupancy

	u32* piece = (u32*)malloc(sizeof(u32) * 8 * 128 * minGridSize * blockSize);  // allocates memory on the CPU for the piece
	u32* d_piece_ptx, * d_expanded_iv_ptx;  // creating device pointers

    cudaMalloc(&d_piece_ptx, sizeof(u32) * 8 * 128 * minGridSize * blockSize);  // allocates memory on the GPU for the piece
    cudaMalloc(&d_expanded_iv_ptx, sizeof(u32) * 8 * minGridSize * blockSize);  // allocates memory on the GPU for the expanded_iv
	// since expanded_iv will be static for a farmer, this does not need to be copied from CPU everytime, it can be hardcoded to GPU

    cudaMemset(d_piece_ptx, 5u, sizeof(u32) * 8 * 128 * minGridSize * blockSize);  // setting all values inside piece as 5
    cudaMemset(d_expanded_iv_ptx, 3u, sizeof(u32) * 8 * minGridSize * blockSize);  // setting all values inside expanded_iv as 3

    encode_ptx_test<<<minGridSize, blockSize >>>(d_piece_ptx, d_expanded_iv_ptx);  // calling the kernel

    cudaMemcpy(piece, d_piece_ptx, sizeof(u32) * 8 * 128 * minGridSize * blockSize, cudaMemcpyDeviceToHost);  // copying the result back to CPU

	cudaDeviceSynchronize();  // wait for GPU operations to finish

    cout << "Operation successful!\n";

	// FOR DEBUGGING THE OUTPUT (prints the piece in hexadecimal)
	/*unsigned char* piece_byte_ptr = (unsigned char*)piece;
	for (int i = 0; i < 128 * 32; i++)
	{
		unsigned number = (unsigned)piece_byte_ptr[i];

		if (number == 0)
		{
			cout << "00";
		}
		else if (number < 16)
		{
			cout << "0";
			cout << hex << number;
		}
		else
		{
			cout << hex << number;
		}

		if (i % 32 == 31)
			cout << endl;
	}*/

    return 0;
}
