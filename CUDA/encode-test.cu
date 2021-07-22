// C++ IMPORTS 
#include <iostream>
#include <bitset>
#include <random>

// CUSTOM IMPORTS
#include "encode_library.h"

using namespace std;

std::random_device dev;
std::mt19937_64 rng(dev());

#define num_piece 1024 * 32

uint256_t random_256()
{
	uint256_t res;

	std::uniform_int_distribution<unsigned long long> randnum(0, UINT64_MAX);

	res.high.high = randnum(rng);
	res.high.low = randnum(rng);
	res.low.high = randnum(rng);
	res.low.low = randnum(rng);

	return res;
}

void random_array_256(uint256_t a[], unsigned n)
{
	std::uniform_int_distribution<unsigned long long> randnum(0, UINT64_MAX);

	for (int i = 0; i < n; i++)
	{
		a[i] = random_256();
	}
}

int main()
{

	uint256_t* piece, * d_piece, * d_nonce;
	cudaMallocHost(&piece, 4 * sizeof(unsigned long long) * 128);

	unsigned char* byte_piece = (unsigned char*)piece;
	for (int i = 0; i < 32 * 128; i++)
	{
		byte_piece[i] = 5u;
	}

	uint256_t expanded_iv;

	unsigned char* byte_expanded_iv = (unsigned char*)&expanded_iv;
	for (int i = 0; i < 32; i++)
	{
		byte_expanded_iv[i] = 3u;
	}

	cudaMalloc(&d_piece, 4 * sizeof(unsigned long long) * 128);
	cudaMemcpyAsync(d_piece, piece, 4 * sizeof(unsigned long long) * 128, cudaMemcpyHostToDevice, 0);

	encode_test << <1, 1 >> > (d_piece, expanded_iv);

	cudaMemcpyAsync(piece, d_piece, 4 * sizeof(unsigned long long) * 128, cudaMemcpyDeviceToHost, 0);
	cudaDeviceSynchronize();


	unsigned char* piece_byte_ptr = (unsigned char*)piece;
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
	}

	return 0;
}