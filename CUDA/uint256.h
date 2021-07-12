// safe include
#pragma once

// C++ imports
#include <cinttypes>
#include <math.h>

// custom imports
#include "uint128.h"

/*
This library will divide 256-bit unsigned integers into 2 (namely `low` and `high`) 128-bit unsigned integers.
All the implemented operations are done via manipulating `low` and `high` 128-bit parts.

Representation: say Q is a 256-bit unsigned integer.

Q = XXX.............................XXX
				(256-bit)


Q = XXX..............|..............XXX
	   (high 128-bit) (low 128-bit)


HOWEVER, keep in mind that, these `low` and `high` bits will be represented as decimal numbers when printed.
In other words, simply concatenating the DECIMAL representations of the `low` and `high` bits will produce wrong results.

Instead, bit representation of the `low` and `high` bits should be concatenated and printed,
then, a "binary->decimal" converter should be used

This library uses `uint128` custom library for its building blocks.
Understanding how `uint128` library works is crucial for understanding the functionalities of this library.

This library is not commented where the concept is borrowed from `uint128` library.
This library is only commented where the concept is unique to this library (not common with `uint128`)
*/

class uint256_t
{
public:

	uint128_t low;   // store the low 128-bit in here
	uint128_t high;  // store the high 128-bit in here

	__host__ __device__ __forceinline__ uint256_t()
	{
		low = 0;
		high = 0;
	}


	__host__ __device__ __forceinline__ void operator=(const uint64_t& x)
	{
		low.low = x;
		high = 0;
	}


	__host__ __device__ __forceinline__ void operator=(const uint128_t& x)
	{
		low = x;
		high = 0;
	}


	__host__ __device__ __forceinline__ void operator=(const uint256_t& x)
	{
		low = x.low;
		high = x.high;
	}


	__host__ __device__ __forceinline__ uint256_t operator<<(const unsigned& shift)
	{
		uint256_t z;

		z.high = high << shift;
		z.high = (low >> (128 - shift)) | z.high;
		z.low = low << shift;

		return z;
	}


	__host__ __device__ __forceinline__ uint256_t operator>>(const unsigned& shift)
	{
		uint256_t z;

		z.low = low >> shift;
		z.low = (high << (128 - shift)) | z.low;
		z.high = high >> shift;

		return z;
	}
};

__host__ __device__ __forceinline__ bool operator<(const uint256_t& l, const uint256_t& r)
{
	if (l.high < r.high)
		return true;
	else if (l.high > r.high)
		return false;
	else if (l.low < r.low)
		return true;
	else
		return false;
}

__host__ __device__ __forceinline__ bool operator>(const uint256_t& l, const uint256_t& r)
{
	if (l.high > r.high)
		return true;
	else if (l.high < r.high)
		return false;
	else if (l.low > r.low)
		return true;
	else
		return false;
}

__host__ __device__ __forceinline__ bool operator==(const uint256_t& l, const uint256_t& r)
{
	if ((l.low == r.low) && (l.high == r.high))  // both parts need to be equal to each other
		return true;
	else
		return false;
}

__host__ __device__ __forceinline__ uint256_t operator|(const uint256_t& x, const uint256_t& y)
{
	uint256_t z;

	z.low = x.low | y.low;
	z.high = x.high | y.high;

	return z;
}

__host__ __device__ __forceinline__ uint256_t operator-(const uint256_t& x, const uint256_t& y)
{
	uint256_t z;

	z.low = x.low - y.low;
	z.high = x.high - y.high - (x.low < y.low);

	return z;
}

__host__ __device__ __forceinline__ uint256_t operator-(const uint256_t& x, const uint64_t& y)
{
	uint256_t z;

	z.low = x.low - y;
	z.high = x.high - (x.low < y);

	return z;
}

__host__ __device__ __forceinline__ uint256_t operator+(const uint256_t& x, const uint256_t& y)
{
	uint256_t z;

	z.low = x.low + y.low;

	z.high = x.high + y.high + (z.low < x.low);

	return z;
}

__host__ __device__ __forceinline__ bool isEven(const uint256_t& x)
{
	if (x.low.low & 1) {
		return false;
	}
	return true;
}

__host__ __device__ __forceinline__ bool isOdd(const uint256_t& x)
{
	if (x.low.low & 1) {
		return true;
	}
	return false;
}


/* ZA WARUDOOOOOOOO */
__device__ __forceinline__ uint256_t mul128x2(const uint128_t& a, const uint128_t& b)
{  // uses KARATSUBA multiplication
	/*
	a = a.high|a.low
	b = b.high|b.low

	a*b can be represented as follows:

					a = a.high|a.low
					b = b.high|b.low
								   x
					----------------
						a.low*b.low

				a.high*b.low
				a.low*b.high

		a.high*b.high
								   +
		----------------------------
		   c.high    |    c.low

	*/

	uint256_t c;
	uint128_t temp;

	c.low = mul64x2(a.low, b.low); // a.low * b.low

	temp = mul64x2(a.high, b.low);
	// low part of (a.high * b.low) will be added to c.low.high
	// high part of (a.high * b.low) will be added to c.high.low
	c.low.high += temp.low;  // after this addition, there may be a carry
	// if overflow occurs, the result `c.low.high` will be smaller than both`temp.low` 
	// consider 1 digit decimal addition -> 9+3=2
	// 2 < 9  &&  2 < 3
	// so, if `c.low.high` is smaller than `temp.low`, we know that there is a carry
	c.high.low = temp.high + (c.low.high < temp.low); // add the potential carry with a boolean condition to evade branching
	// for this addition, there cannot be overflow, since it's the first time we are putting something into c.high
	// also, temp.high cannot be too large to overflow with a carry

	temp = mul64x2(a.low, b.high);
	// low part of (a.low * b.high) will be added to c.low.high
	// high part of (a.low * b.high) will be added to c.high.low
	c.low.high += temp.low;  // after this addition, there may be a carry
	c.high.low += temp.high + (c.low.high < temp.low);// add the potential carry with a boolean condition to evade branching
	// however, there can be another carry because of the last addition (c.high already had some bits in it before here) 
	c.high.high = (c.high.low < temp.high); // add the potential carry into c.high.high

	temp = mul64x2(a.high, b.high);  // a.high * b.high
	c.high.low += temp.low;  // add temp.low to high.low
	// but a carry might happen
	c.high.high += temp.high + (c.high.low < temp.low); // add the carry from previous step, along with the temp.high

	return c;
}