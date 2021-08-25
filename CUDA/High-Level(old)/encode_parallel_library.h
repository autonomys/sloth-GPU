// safe include
#pragma once

// CUSTOM IMPORTS
#include "uint512.h"

using namespace std;

#define PRIME uint256_t p(18446744073709551615, 18446744073709551615, 18446744073709551615, 18446744073709551427)
#define MU uint512_t mu(0, 0, 0, 1, 0, 0, 0, 189)
#define EXP uint256_t expo(4611686018427387903, 18446744073709551615, 18446744073709551615, 18446744073709551569)
#define k 256

#define num_threads 256

__device__ __forceinline__ uint256_t weird_reduction(uint256_t a, uint256_t b)
{

	uint512_t temp = mul256x2(a, b);
	uint256_t mull; mull.low.low = 189;

	uint512_t temp2; temp2.low = temp.low;
	temp = temp2 + mul256x2(temp.high, mull);

	uint512_t temp3; temp3.low = temp.low;

	temp2.high = temp.high.low.low & 255;

	temp = temp3 + mul256x2(temp2.high, mull);

	temp3.low = temp.low;

	temp2.high = temp.high.low.low & 1;

	temp = temp3 + mul256x2(temp2.high, mull);

	return temp.low;

}

__device__ __forceinline__ uint256_t square_mul(uint256_t a, unsigned count, uint256_t b)
{
	uint256_t t = a;

	while (count--)
	{
		t = weird_reduction(t, t);
	}
	t = weird_reduction(t, b);
	return t;
}

__device__ __forceinline__ uint256_t addition_chain_reduce(uint256_t a)
{
	PRIME;

	uint256_t x, y;

	x = square_mul(a, 1, a);  /* 0x3 */
	y = square_mul(x, 1, a);    /* 0x7 */
	x = square_mul(y, 3, y);      /* 0x3f */
	x = square_mul(x, 1, a);    /* 0x7f */
	x = square_mul(x, 7, x);      /* 0x3fff */
	x = square_mul(x, 14, x);     /* 0xfffffff */
	x = square_mul(x, 3, y);      /* 0x7fffffff */
	x = square_mul(x, 31, x);     /* 0x3fffffffffffffff */
	x = square_mul(x, 62, x);     /* 0xfffffffffffffffffffffffffffffff */
	x = square_mul(x, 124, x);    /* 0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff */
	x = square_mul(x, 2, a);    /* 0x3fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffd */
	x = square_mul(x, 4, a);    /* 0x3fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffd1 */

	if (!(x < p))
		return x - p;

	return x;
}

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

__global__ void montgomery_caller(uint256_t *a, uint256_t expo) 
{
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

__device__ __forceinline__ uint256_t sqrt_permutation_new(uint256_t a) {

	PRIME;
	EXP;

	uint256_t square_root = addition_chain_reduce(a);
	if (square_root.isOdd()) {
		square_root = p - square_root;
	}

	uint256_t check_square_root = weird_reduction(square_root, square_root);
	if (!(check_square_root < p))
		check_square_root = check_square_root - p;

	if (check_square_root == a) {
		return square_root;
	}

	if (square_root.isEven())
		square_root = p - square_root;
	return square_root;

}

__device__ __forceinline__ uint256_t sqrt_permutation_mont(uint256_t a) {

	PRIME;
	EXP;

	uint256_t square_root = montgomery_exponentiation(a, expo);
	if (square_root.isOdd()) {
		square_root = p - square_root;
	}

	uint256_t expo_2;
	expo_2.low.low = 2;

	uint256_t check_square_root = montgomery_exponentiation(square_root, expo_2);

	if (check_square_root == a) {
		return square_root;
	}

	if (square_root.isEven())
		square_root = p - square_root;
	return square_root;

}

__device__ __forceinline__ uint256_t sqrt_permutation_old(uint256_t a) {

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

/*__global__ void sqrt_caller(uint256_t* a)
{
	*a = sqrt_permutation(*a);
}*/

__global__ void encode_new(uint256_t *a, uint256_t *nonce, uint256_t farmer_id)
{
	int global_idx = threadIdx.x + blockIdx.x * num_threads;

	uint256_t feedback = nonce[global_idx] ^ farmer_id;

#pragma unroll
	for (int i = 0; i < 128; i++)
	{
		feedback = sqrt_permutation_new(a[i + global_idx * 128] ^ feedback);
		a[i + global_idx * 128] = feedback;
	}
}

__global__ void encode_mont(uint256_t* a, uint256_t* nonce, uint256_t farmer_id)
{
	int global_idx = threadIdx.x + blockIdx.x * num_threads;

	uint256_t feedback = nonce[global_idx] ^ farmer_id;

#pragma unroll
	for (int i = 0; i < 128; i++)
	{
		feedback = sqrt_permutation_mont(a[i + global_idx * 128] ^ feedback);
		a[i + global_idx * 128] = feedback;
	}
}

__global__ void encode_old(uint256_t* a, uint256_t* nonce, uint256_t farmer_id)
{
	int global_idx = threadIdx.x + blockIdx.x * num_threads;

	uint256_t feedback = nonce[global_idx] ^ farmer_id;

#pragma unroll
	for (int i = 0; i < 128; i++)
	{
		feedback = sqrt_permutation_old(a[i + global_idx * 128] ^ feedback);
		a[i + global_idx * 128] = feedback;
	}
}

/*
__global__ void encode_test_new(uint256_t* a, uint256_t expanded_iv)
{
	uint256_t feedback = expanded_iv;

#pragma unroll
	for (int i = 0; i < 128; i++)
	{
		uint256_t xor_result = a[i] ^ feedback;

		feedback = sqrt_permutation_new(a[i] ^ feedback);

		a[i] = feedback;
	}
}

__global__ void encode_test_mont(uint256_t* a, uint256_t expanded_iv)
{
	uint256_t feedback = expanded_iv;

#pragma unroll
	for (int i = 0; i < 128; i++)
	{
		uint256_t xor_result = a[i] ^ feedback;

		feedback = sqrt_permutation_mont(a[i] ^ feedback);

		a[i] = feedback;
	}
}

__global__ void encode_test_old(uint256_t* a, uint256_t expanded_iv)
{
	uint256_t feedback = expanded_iv;

#pragma unroll
	for (int i = 0; i < 128; i++)
	{
		uint256_t xor_result = a[i] ^ feedback;

		feedback = sqrt_permutation_old(a[i] ^ feedback);

		a[i] = feedback;
	}
}
*/