// safe include
#pragma once

// CUSTOM IMPORTS
#include "uint512.h"

using namespace std;

#define PRIME uint256_t p(18446744073709551615, 18446744073709551615, 18446744073709551615, 18446744073709551427)
#define MU uint512_t mu(0, 0, 0, 1, 0, 0, 0, 189)
#define EXP uint256_t expo(4611686018427387903, 18446744073709551615, 18446744073709551615, 18446744073709551569)
#define k 256

#define num_threads 128

__device__ __forceinline__ uint256_t shared_modular_multiplication(uint256_t x, uint256_t y, uint256_t s256[], uint512_t s512[])
{
	PRIME;
	MU;

	/*uint512_t t = mul256x2(x, y);
	uint256_t th = (t >> k).low; //since k is 256 we know that high 256 bits will be zero
	uint512_t t1 = mul257_256(mu, th);
	uint256_t t1h = (t1 >> k).low;
	uint512_t t2 = mul256x2(t1h, p);
	uint512_t cbar = t - t2;
	
	
	if (cbar < p) {  // if cbar does not require reduction
		return cbar.low;
	}
	if (cbar - p < p)  // if cbar requires only 1 reduction
	{
		uint512_t t3 = cbar - p;
		return t3.low;
	}
	else  // if cbar requires 2 reduction (cbar - p - p < p), there is no other condition possible
	{
		uint512_t t4 = cbar - p - p;
		return t4.low;
	}*/
	s512[0] = mul256x2(x, y);
	s256[2] = (s512[0] >> k).low; //since k is 256 we know that high 256 bits will be zero
	s512[1] = mul257_256(mu, s256[2]);
	s256[4] = (s512[0] >> k).low;
	s512[2] = mul256x2(s256[4], p);
	s512[3] = s512[0] - s512[2];
	s512[4] = s512[3] - p;
	s512[5] = s512[3] - p - p;

	if (s512[5] < p)
		return s512[5].low;
	else if (s512[4] < p)
		return s512[4].low;
	else
		return s512[3].low;
}

__device__ __forceinline__ uint256_t shared_montgomery_exponentiation(uint256_t a, uint256_t expo, uint256_t s256[], uint512_t s512[])
{

	s256[0].low.low = 1ull;
	s256[1] = a;

	for (int i = k - 1; i >= 0; i--)
	{

		if (!((expo >> i).low.low & 1)) //returns true if number is even
		{
			s256[1] = shared_modular_multiplication(s256[0], s256[1], s256, s512);
			s256[0] = shared_modular_multiplication(s256[0], s256[0], s256, s512);
		}
		else
		{
			s256[0] = shared_modular_multiplication(s256[0], s256[1], s256, s512);
			s256[1] = shared_modular_multiplication(s256[1], s256[1], s256, s512);
		}
	}

	return s256[0];
}

__global__ void montgomery_caller(uint256_t *a, uint256_t expo) 
{
	*a = montgomery_exponentiation(*a, expo);
	
}



__device__ __forceinline__ uint256_t sqrt_permutation(uint256_t a, uint256_t s256[], uint512_t s512[]) {

	PRIME;
	EXP;

	uint256_t square_root = montgomery_exponentiation(a, expo, s256, s512);
	if (square_root.isOdd()) {
		square_root = p - square_root;
	}

	uint256_t expo_2;
	expo_2.low.low = 2;

	uint256_t check_square_root = montgomery_exponentiation(square_root, expo_2, s256, s512);

	if (check_square_root == a) {
		return square_root;
	}

	if (square_root.isEven())
		return square_root;
	square_root = p - square_root;
	return square_root;
	
}

__global__ void sqrt_caller(uint256_t* a)
{
	*a = sqrt_permutation(*a);
}

__global__ void shared_encode(uint256_t* a, uint256_t* nonce, uint256_t farmer_id)
{
	__shared__ uint256_t s256[4];
	__shared__ uint512_t s512[6];

	unsigned global_tid = threadIdx.x + blockIdx.x * 128;  // 128 thread per block
	uint256_t feedback = nonce[global_tid] ^ farmer_id;

#pragma unroll
	for (int i = 0; i < 128; i++)
	{
		feedback = shared_sqrt_permutation(a[global_tid * 128 + i] ^ feedback, s256, s512);
		a[global_tid * 128 + i] = feedback;
	}
}

__global__ void shared_encode_coalesced(uint256_t* a, uint256_t* nonce, uint256_t farmer_id, unsigned long long piece_count)
{
	__shared__ uint256_t s256[4];
	__shared__ uint512_t s512[6];

	unsigned global_tid = threadIdx.x + blockIdx.x * 128;  // 128 thread per block
	uint256_t feedback = nonce[global_tid] ^ farmer_id;

	uint256_t one = a[global_tid];
	uint256_t two;

#pragma unroll
	for (int i = 0; i < 127; i++)
	{
		two = a[global_tid + piece_count * (i + 1)];
		feedback = shared_sqrt_permutation(one ^ feedback, s256, s512);
		one = two;
		a[global_tid + piece_count * i] = feedback;
	}

	feedback = shared_sqrt_permutation(one ^ feedback, s256, s512);
	a[global_tid + piece_count * 127] = feedback;
}
