#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cinttypes>
#include <string>
#include <math.h>
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

/* ZA WARUDOOOOOOOOOOO */
__device__ __forceinline__ uint256_t mul64_256(const uint128_t& a, const uint128_t& b)
{  // refer to the comments of host function host128x2, logic is the same
	uint256_t c;
	uint128_t temp;

	c.low = mul64_128(a.low, b.low); //alow * blow
	
	temp = mul64_128(a.high, b.low);
	c.low.high += temp.low;
	c.high.low = temp.high + (c.low.high < temp.low); //ahigh * blow

	temp = mul64_128(a.low, b.high);
	c.low.high += temp.low;
	c.high.low += temp.high + (c.low.high < temp.low);
	c.high.high += (c.high.low < temp.high); //alow * bhigh

	temp = mul64_128(a.high, b.high);
	c.high.low += temp.low;
	c.high.high = temp.high + (c.high.low < temp.low); //ahigh * bhigh

	return c;
}