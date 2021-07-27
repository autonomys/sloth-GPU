// C++ IMPORTS 
#include <iostream>
#include <bitset>
#include <random>

// CUSTOM IMPORTS
#include "encode_library.h"

using namespace std;

#define num_piece 1024 * 16
#define num_threads 128

std::random_device dev;
std::mt19937_64 rng(dev());

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

void random_array_256(uint256_t a[], unsigned iter_amount)
{
	std::uniform_int_distribution<unsigned long long> randnum(0, UINT64_MAX);

	for (int i = 0; i < iter_amount; i++)
	{
		a[i] = random_256();
	}
}

int main()
{
	uint256_t* piece, * d_piece, * d_nonce;
	cudaMallocHost(&piece, 4 * sizeof(unsigned long long) * 128 * num_piece);
	random_array_256(piece, 128 * num_piece);

	uint256_t* nonce;
	cudaMallocHost(&nonce, 4 * sizeof(unsigned long long) * num_piece);
	random_array_256(nonce, num_piece);

	uint256_t farmer_id = random_256();

	cudaMalloc(&d_piece, 4 * sizeof(unsigned long long) * 128 * num_piece);
	cudaMalloc(&d_nonce, 4 * sizeof(unsigned long long) * num_piece);
	cudaMemcpyAsync(d_piece, piece, 4 * sizeof(unsigned long long) * 128 * num_piece, cudaMemcpyHostToDevice, 0);
	cudaMemcpyAsync(d_nonce, nonce, 4 * sizeof(unsigned long long) * num_piece, cudaMemcpyHostToDevice, 0);

	encode << <num_piece / num_threads, num_threads >> > (d_piece, d_nonce, farmer_id);

	cudaMemcpyAsync(d_piece, piece, 4 * sizeof(unsigned long long) * 128 * num_piece, cudaMemcpyHostToDevice, 0);

	empty_encode << <num_piece / num_threads, num_threads >> > (d_piece, d_nonce, farmer_id);

	cudaMemcpyAsync(d_piece, piece, 4 * sizeof(unsigned long long) * 128 * num_piece, cudaMemcpyHostToDevice, 0);

	encode_coalesced << <num_piece / num_threads, num_threads >> > (d_piece, d_nonce, farmer_id, num_piece);

	cudaMemcpyAsync(d_piece, piece, 4 * sizeof(unsigned long long) * 128 * num_piece, cudaMemcpyHostToDevice, 0);

	empty_encode_coalesced << <num_piece / num_threads, num_threads >> > (d_piece, d_nonce, farmer_id, num_piece);

	cudaMemcpy(piece, d_piece, 4 * sizeof(unsigned long long) * 128 * num_piece, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();


	return 0;
}
