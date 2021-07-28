// safe include
#pragma once
// CUSTOM IMPORTS
#include "uint512.h"
using namespace std;
#define PRIME uint256_t p(18446744073709551615, 18446744073709551615, 18446744073709551615, 18446744073709551427)
#define MU uint512_t mu(0, 0, 0, 1, 0, 0, 0, 189)
#define EXP uint256_t expo(4611686018427387903, 18446744073709551615, 18446744073709551615, 18446744073709551569)
#define k 256

__device__ __forceinline__ uint256_t modular_multiplication(uint256_t x, uint256_t y)
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
	uint512_t t = mul256x2(x, y);
	uint256_t th = (t >> k).low; //since k is 256 we know that high 256 bits will be zero
	uint512_t t1 = mul257_256(mu, th);
	uint256_t t1h = (t1 >> k).low;
	uint512_t t2 = mul256x2(t1h, p);
	uint512_t cbar = t - t2;
	uint512_t t3 = cbar - p;
	uint512_t t4 = cbar - p - p;

	if (t4 < p)
		return t4.low;
	else if (t3 < p)
		return t3.low;
	else
		return cbar.low;
}

__device__ __forceinline__ uint256_t montgomery_exponentiation(uint256_t a, uint256_t expo)
{

	uint256_t c0; c0.low.low = 1ull;
	uint256_t c1 = a;

	for (int i = k - 1; i >= 0; i--)
	{

		if (!((expo >> i).low.low & 1)) //returns true if number is even
		{
			c1 = modular_multiplication(c0, c1);
			c0 = modular_multiplication(c0, c0);
		}
		else
		{
			c0 = modular_multiplication(c0, c1);
			c1 = modular_multiplication(c1, c1);
		}
	}

	return c0;
}

__global__ void montgomery_caller(uint256_t *a) 
{
	EXP;

	*a = montgomery_exponentiation(*a, expo);
	
}

__device__ __forceinline__ bool legendre(uint256_t a)
{
	PRIME;

	if (montgomery_exponentiation(a, (p - 1) >> 1) == (p - 1))
		return false;
	else {
		return true;
	}
}

__global__ void legendre_caller(uint256_t *a) 
{
	bool result = legendre(*a);
    if (result) {
        printf("passed");
    }
    else {
        printf("failed");
    }
}

__device__ __forceinline__ uint256_t sqrt_permutation(uint256_t a) {

	PRIME;
	EXP;

	if (legendre(a)) {
		a = montgomery_exponentiation(a, expo);
		if (a.isOdd()) {
			a = p - a;
		}
	}
	else {
		a = p - a;
		a = montgomery_exponentiation(a, expo);
		if (a.isEven()) {
			a = p - a;
		}
	}
	return a;
}

__global__ void sqrt_caller(uint256_t* a)
{
	*a = sqrt_permutation(*a);
}

__global__ void encode(uint256_t *a, uint256_t *nonce, uint256_t farmer_id)
{
	unsigned global_tid = blockIdx.x + blockDim.x * 128;  // 128 thread per block
	uint256_t feedback = nonce[global_tid] ^ farmer_id;

#pragma unroll
	for (int i = 0; i < 128; i++)
	{
		feedback = sqrt_permutation(a[global_tid * 128 + i] ^ feedback);
		a[global_tid * 128 + i] = feedback;
	}
}


__global__ void encode_coalesced(uint256_t *a, uint256_t *nonce, uint256_t farmer_id, unsigned long long piece_count)
{
	unsigned global_tid = blockIdx.x + blockDim.x * 128;  // 128 thread per block
	uint256_t feedback = nonce[global_tid] ^ farmer_id;

#pragma unroll
	for (int i = 0; i < 128; i++)
	{
		feedback = sqrt_permutation(a[global_tid + piece_count * i] ^ feedback);
		a[global_tid + piece_count * i] = feedback;
	}
}

__global__ void empty_encode(uint256_t* a, uint256_t* nonce, uint256_t farmer_id)
{
	unsigned global_tid = threadIdx.x + blockIdx.x * 128;  // 128 thread per block
	uint256_t feedback = nonce[global_tid] ^ farmer_id;

#pragma unroll
	for (int i = 0; i < 128; i++)
	{
		feedback = a[global_tid * 128 + i] ^ feedback;
		a[global_tid * 128 + i] = feedback;
	}
}

__global__ void empty_encode_coalesced(uint256_t* a, uint256_t* nonce, uint256_t farmer_id, unsigned long long piece_count)
{
	unsigned global_tid = threadIdx.x + blockIdx.x * 128;  // 128 thread per block
	uint256_t feedback = nonce[global_tid] ^ farmer_id;

#pragma unroll
	for (int i = 0; i < 128; i++)
	{
		feedback = a[global_tid + piece_count * i] ^ feedback;
		a[global_tid + piece_count * i] = feedback;
	}
}