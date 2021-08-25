#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <bitset>
#include <string>

#include "test_parallel.h"

using namespace std;

int main()
{
	int blockSize;      // The launch configurator returned block size
	int minGridSize;    // The minimum grid size needed to achieve the
						// maximum occupancy for a full device launch
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, encode_ptx_test);

	//minGridSize = 1; blockSize = 1;

	cout << "kernel<<<" << minGridSize << ", " << blockSize << ">>>" << endl;

	u32* piece = (u32*)malloc((sizeof(u32) * 8 + 64)* 128 * minGridSize * blockSize);

	u32* d_piece_ptx;

    cudaMalloc(&d_piece_ptx, (sizeof(u32) * 8 + 64) * 128 * minGridSize * blockSize);

    cudaMemset(d_piece_ptx, 5u, (sizeof(u32) * 8 + 64) * 128 * minGridSize * blockSize);
    cudaMemset(d_piece_ptx, 3u, (sizeof(u32)) * 8);

    encode_ptx_test<<<minGridSize, blockSize >>>(d_piece_ptx, d_expanded_iv_ptx);

    cudaMemcpy(piece, d_piece_ptx, (sizeof(u32) * 8 + 64) * 128 * minGridSize * blockSize, cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();

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
