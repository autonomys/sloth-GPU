// C++ IMPORTS 
#include <iostream>
#include <bitset>
#include <random>

// CUSTOM IMPORTS
#include "encode_library.h"

using namespace std;

std::random_device dev;
std::mt19937_64 rng(dev());

#define num_piece 1024 * 64

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
	uint256_t* piece, * d_piece, * nonce, * d_nonce, farmer_id;

	cudaMallocHost(&piece, 4 * sizeof(unsigned long long) * 128 * num_piece);
	cudaMallocHost(&nonce, 4 * sizeof(unsigned long long) * num_piece);

	cudaMalloc(&d_piece, 4 * sizeof(unsigned long long) * 128 * num_piece);
	cudaMalloc(&d_nonce, 4 * sizeof(unsigned long long) * num_piece);

	random_array_256(piece, 128 * num_piece);
	random_array_256(nonce, num_piece);
	farmer_id = random_256();

	cudaMemcpyAsync(d_piece, piece, 4 * sizeof(unsigned long long) * 128 * num_piece, cudaMemcpyHostToDevice, 0);
	cudaMemcpyAsync(d_nonce, nonce, 4 * sizeof(unsigned long long) * num_piece, cudaMemcpyHostToDevice, 0);

	encode_new << <num_piece / num_threads, num_threads >> > (d_piece, d_nonce, farmer_id);

	cudaMemcpyAsync(d_piece, piece, 4 * sizeof(unsigned long long) * 128 * num_piece, cudaMemcpyHostToDevice, 0);

	encode_mont << <num_piece / num_threads, num_threads >> > (d_piece, d_nonce, farmer_id);

	cudaMemcpyAsync(d_piece, piece, 4 * sizeof(unsigned long long) * 128 * num_piece, cudaMemcpyHostToDevice, 0);

	encode_old << <num_piece / num_threads, num_threads >> > (d_piece, d_nonce, farmer_id);

	cudaMemcpyAsync(piece, d_piece, 4 * sizeof(unsigned long long) * 128 * num_piece, cudaMemcpyDeviceToHost, 0);
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
	}
	*/
	return 0;
}