// safe include
#pragma once

// CUSTOM IMPORTS
#include "uint512.h"

using namespace std;

__device__ uint256_t modular_multiplication(uint256_t x, uint256_t y, uint256_t p, uint512_t mu, unsigned k)
{
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

__device__ uint256_t montgomery_exponentiation(uint256_t a, uint256_t exp, uint256_t p, uint512_t mu, unsigned k)
{
	uint256_t c0; c0.low.low = 1ull;
	uint256_t c1 = a;

	for (int i = k - 1; i >= 0; i--)
	{

		if (!((exp >> i).low.low & 1)) //returns true if number is even
		{
			c1 = modular_multiplication(c0, c1, p, mu, k);
			c0 = modular_multiplication(c0, c0, p, mu, k);
		}
		else
		{
			c0 = modular_multiplication(c0, c1, p, mu, k);
			c1 = modular_multiplication(c1, c1, p, mu, k);
		}
	}

	return c0;
}

__global__ void montgomery_caller(uint256_t *a, uint256_t exp, uint256_t p, uint512_t mu, unsigned k) 
{
	*a = montgomery_exponentiation(*a, exp, p, mu, k);
	
}

__device__ bool legendre(uint256_t a, uint256_t p, uint512_t mu, unsigned k)
{
	if (montgomery_exponentiation(a, (p - 1) >> 1, p, mu, k) == (p - 1))
		return false;
	else {
		return true;
	}
}

__global__ void legendre_caller(uint256_t *a, uint256_t p, uint512_t mu, unsigned k) 
{
	bool result = legendre(*a, p, mu, k);
    if (result) {
        printf("passed");
    }
    else {
        printf("failed");
    }
}