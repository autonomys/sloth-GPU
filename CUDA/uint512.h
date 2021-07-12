// safe include
#pragma once

// C++ imports
#include <cinttypes>

// custom imports
#include "uint256.h"

/*
This library will divide 512-bit unsigned integers into 2 (namely `low` and `high`) 256-bit unsigned integers.
All the implemented operations are done via manipulating `low` and `high` 128-bit parts.

Representation: say Q is a 256-bit unsigned integer.

Q = XXX.............................XXX
				(512-bit)


Q = XXX..............|..............XXX
	   (high 256-bit) (low 256-bit)


HOWEVER, keep in mind that, these `low` and `high` bits will be represented as decimal numbers when printed.
In other words, simply concatenating the DECIMAL representations of the `low` and `high` bits will produce wrong results.

Instead, bit representation of the `low` and `high` bits should be concatenated and printed,
then, a "binary->decimal" converter should be used

This library uses `uint128` custom library for its building blocks.
Understanding how `uint128` library works is crucial for understanding the functionalities of this library.

This library is not commented where the concept is borrowed from `uint128` library.
This library is only commented where the concept is unique to this library (not common with `uint128`)
*/

class uint512_t
{
public:

	uint256_t low;
	uint256_t high;

	__host__ __device__ __forceinline__ uint512_t()
	{
		low = 0;
		high = 0;
	}

	__host__ __device__ __forceinline__ uint512_t operator>>(const unsigned& shift)
	{
		uint512_t z;

		if (shift >= 512)
		{
			// z = 0
		}
		else if (shift > 256)
		{
			z.low = high >> (shift - 256);
		}
		else if (shift == 0)
		{
			z.high = high;
			z.low = low;
		}
		else if (shift < 256)
		{
			z.low = low >> shift;
			z.low = (high << (256 - shift)) | z.low;
			z.high = high >> shift;
		}
		else
		{
			z.low = high;
		}

		return z;
	}

	__host__ __device__ __forceinline__ uint512_t operator=(const uint512_t& l)
	{
		low = l.low;
		high = l.high;
	}
};

__host__ __device__ __forceinline__ uint512_t operator-(const uint512_t& x, const uint512_t& y)
{
	uint512_t z;

	z.low = x.low - y.low;
	z.high = x.high - y.high - (x.low < y.low);

	return z;
}

__host__ __device__ __forceinline__ uint512_t operator-(const uint512_t& x, const uint256_t& y)
{
	uint512_t z;

	z.low = x.low - y;
	z.high = x.high - (x.low < y);

	return z;
}

/* WRYYYYYYYYYYYYYYYYYYYYYYYYYYYY */
__device__ __forceinline__ uint512_t mul256x2(const uint256_t& a, const uint256_t& b)
{
	uint512_t c;
	uint256_t temp;

	c.low = mul128x2(a.low, b.low);

	temp = mul128x2(a.high, b.low);
	c.low.high = c.low.high + temp.low;
	c.high.low = temp.high + (c.low.high < temp.low);

	temp = mul128x2(a.low, b.high);
	c.low.high = c.low.high + temp.low;
	c.high.low = c.high.low + temp.high + (c.low.high < temp.low);
	c.high.high = (c.high.low < temp.high);

	temp = mul128x2(a.high, b.high);
	c.high.low = c.high.low + temp.low;
	c.high.high = c.high.high + temp.high + (c.high.low < temp.low);

	return c;
}

__device__ __forceinline__ uint512_t mul257_256(const uint512_t& a, const uint256_t& b)
{
	uint512_t c;  // inside this, we are only storing a 257 bit number (we are not utilizing all 512 bits here)

	c = mul256x2(a.low, b);  // multiply the 256 bits of a, with the whole b, and store the result in c

	if (a.high.low.low & 1)  // might be unnecessary, if a is 257 bit, a.high.low.low & 1 should always return true
		c.high = c.high + b;  // add the whole b to the high part of c
		// the reasoning behind the above line is: 
		// since the 257th bit of a is 1, we will shift b by 256 bit,
		// and add this to the c
		// which is effectively adding b to the high bits of the c

	return c;
}

__host__ __device__ __forceinline__ bool operator<(const uint512_t& x, const uint256_t& y)
{
	if ((x.high.high.high ^ 0) | (x.high.high.low ^ 0) | (x.high.low.high ^ 0) | (x.high.low.low ^ 0))
	{  // means high bits of x is not completely 0, so it's bigger than y
		return false;
	}
	else  // means high bits of x is completely 0, so we have to investigate further
	{
		if (x.low < y) 
		{
			return true;
		}
		else 
		{
			return false;
		}
	}
}
