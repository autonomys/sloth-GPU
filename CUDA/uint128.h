// safe include
#pragma once

// CUDA IMPORTS
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// C++ imports
#include <cinttypes>
#include <math.h>


/*
This library will divide 128-bit unsigned integers into 2 (namely `low` and `high`) 64-bit unsigned integers.
All the implemented operations are done via manipulating `low` and `high` 64-bit parts.

Representation: say Q is a 128-bit unsigned integer.

Q = XXX...........................XXX
				(128-bit)


Q = XXX.............|.............XXX
	   (high 64-bit) (low 64-bit)


HOWEVER, keep in mind that, these `low` and `high` bits will be represented as decimal numbers when printed.
In other words, simply concatenating the DECIMAL representations of the `low` and `high` bits will produce wrong results.

Instead, bit representation of the `low` and `high` bits should be concatenated and printed,
then, a "binary->decimal" converter should be used
*/


class uint128_t
{
public:

	unsigned long long low;   // storing the low 64-bit of the 128-bit integer
	unsigned long long high;  // storing the high 64-bit of the 128-bit integer

	__host__ __device__ __forceinline__ uint128_t()  // forceinline -> makes this function inline, more optimized compilation
	{
		low = 0;
		high = 0;
	}

	__host__ __device__ __forceinline__ uint128_t(const uint64_t& x)  // when initialized from a 64-bit integer
	{
		low = x;   // we only replace the low bits with the given 64-bit number
		high = 0;  // and initialize the high bits as 0 
	}

	__host__ __device__ __forceinline__ void operator=(const uint128_t& r)  // copy 128-bit into another 128-bit
	{
		low = r.low;    // basically, copy all the parts individually
		high = r.high;  // basically, copy all the parts individually
	}

	__host__ __device__ __forceinline__ void operator=(const uint64_t& r)  // copy 64-bit into a 128-bit
	{
		low = r;   // we only replace the low bits with the given 64-bit number
		high = 0;  // and initialize the high bits as 0 
	}

	__host__ __device__ __forceinline__ uint128_t operator<<(const unsigned& shift) const  // left shift operation
	{

		uint128_t z;  // shift operator does not override the original number, instead, returns another one

		if (shift >= 128)  // if shift amount is larger or equal to 128, it means we are going out of boundary. So simply return 0
		{
			// z = 0
		}
		else if (shift > 64)  // if 128 > shift > 64
		{
			// high bits will be discarded, and some of the low bits will be transferred to high bits
			// no low bits, will stay in the low part, so low bits will be 0
			// z.low = 0  // is same as not writing this
			z.high = low >> (shift - 64);
		}
		else if (shift == 0)  // ?
		{
			z.high = high;
			z.low = low;
		}
		else if (shift < 64)  // beautiful comments present below <3
		{
			z.high = high << shift;  // left shift the `high` bits by the given amount
			z.high = (low >> (64 - shift)) | z.high;  // some of the `low` bits may become new `high` bits after the shift.
			/*
			say, shift amount is = 3

			input:
			Q = KLM..........PRS|ABC...........XYZ
				   (high 64-bit) (low 64-bit)


			intermediate results after trivial shift:
			Q = ..........PRS000|...........XYZ000
				   (high 64-bit) (low 64-bit)

			result we want:
			Q = ..........PRSABC|...........XYZ000
				   (high 64-bit) (low 64-bit)


			When we shift the `high` bits, the least significant `3` bits of the `high` bits will be 0 (000 after PRS).
			Also, the most significant `3` bits of the `low` bits will be discarded after the shift operation (ABC).
			We basically want to add these `3` discarded `low` bits to the least significant `3` bits of the `high` bits:

			000..00000ABC  ->  61*0 + ABC
			.......PRS000  ->  high bits, shifted to the right

			we can achieve [.......PRS000] by:
			high << shift;

			we can achieve [000..00000ABC] by:
			low >> (64 - shift);

			This addition between bits can be performed with | (or) operation.
			low >> (64 - shift)) | high;
			*/

			z.low = low << shift;
		}
		else  // shift == 64
		{  // this scenario is explained above
			z.high = low;
		}

		return z;
	}

	__host__ __device__ __forceinline__ uint128_t operator>>(const unsigned& shift) const  // right shift 
	{  // look at the above comments for left shift, it's the same logic
		uint128_t z;


		if (shift >= 128)
		{
			// z = 0
		}
		else if (shift > 64)
		{
			z.low = high >> (shift - 64);
		}
		else if (shift == 0)
		{
			z.high = high;
			z.low = low;
		}
		else if (shift < 64)
		{
			z.low = low >> shift;
			z.low = (high << (64 - shift)) | z.low;
			z.high = high >> shift;
		}
		else
		{
			z.low = high;
		}

		return z;
	}


	__host__ __device__ __forceinline__ void operator<<=(const unsigned& shift)
	{  // check the comments for left shift above for the logic
		low = low >> shift;
		low = (high << (64 - shift)) | low;
		high = high >> shift;

	}

	__host__ __device__ __forceinline__ uint128_t operator^(const uint128_t& r) const
	{
		uint128_t z;

		z.low = low ^ r.low;
		z.high = high ^ r.high;

		return z;
	}

	__host__ __device__ __forceinline__ uint128_t operator|(const uint128_t& r) const
	{
		uint128_t z;

		z.low = low | r.low;
		z.high = high | r.high;

		return z;
	}

	__host__ __device__ __forceinline__ uint128_t operator+(const uint128_t& r) const
	{
		uint128_t z;  // return a new value after the addition operation

		z.low = low + r.low;  // addition of low bits is trivial
		// however, an overflow might occur, and this overflow will affect high bits
		// if overflow occurs, the result `z.low` will be smaller than both`x.low` and `y.low`
		// consider 1 digit decimal addition -> 9+3=2
		// 2 < 9  &&  2 < 3
		// so, if `z.low` is smaller than `x.low`, we know that there is a carry

		z.high = high + r.high + (z.low < low);  // add the potential carry with a boolean condition to evade branching

		return z;
	}

	__host__ __device__ __forceinline__ uint128_t operator+(const uint64_t& r) const
	{ // look at the above comments, same logic applies
		uint128_t z;

		z.low = low + r;
		z.high = high + (z.low < low);

		return z;
	}

	__host__ __device__ uint128_t operator-(const uint128_t& r) const
	{ // reverse logic of addition, look at the comments of addition
		uint128_t z;

		z.low = low - r.low;
		z.high = high - r.high - (low < r.low);

		return z;
	}

	__host__ __device__ __forceinline__ uint128_t operator-(const uint64_t& r) const
	{ // reverse logic of addition, look at the comments of addition
		uint128_t z;

		z.low = low - r;
		z.high = high - (low < r);

		return z;
	}
		__host__ __device__ __forceinline__ void operator-=(const uint128_t& r)
	{ // reverse logic of addition, look at the comments of addition
		high = high - r.high - (low < r.low);
		low = low - r.low;
	}

	__host__ __device__ __forceinline__ bool operator==(const uint128_t& r) const
	{
		if ((low == r.low) && (high == r.high))  // both parts need to be equal to each other
			return true;
		else
			return false;
	}

	__host__ __device__ __forceinline__ bool operator<(const uint128_t& r) const
	{
		if (high < r.high)  // if high bits are greater, we don't need to check low bits
			return true;
		else if (high > r.high)  // if high bits are smaller, we don't need to check low bits
			return false;
		else if (low < r.low)  // only check low bits, if high bits are equal to each other
			return true;
		else
			return false;
	}

	__host__ __device__ __forceinline__ bool operator<(const uint64_t& r) const
	{
		if (high != 0)  // if high bits are not zero, 128-bit is surely greater than 64-bit num
			return false;
		else if (low >= r)
			return false;
		else
			return true;
	}

	__host__ __device__ __forceinline__ bool operator>(const uint128_t& r) const
	{
		if (high > r.high)  // if high bits are greater, we don't need to check low bits
			return true;
		else if (high < r.high)  // if high bits are smaller, we don't need to check low bits
			return false;
		else if (low > r.low)  // only check low bits, if high bits are equal to each other
			return true;
		else
			return false;
	}

	__host__ __device__ __forceinline__ bool operator<=(const uint128_t& r)
	{
		if (high < r.high)  // if high bits are greater, we don't need to check low bits
			return true;
		else if (high > r.high)  // if high bits are smaller, we don't need to check low bits
			return false;
		else if (low <= r.low)  // only check low bits, if high bits are equal to each other
			return true;
		else
			return false;
	}

	__host__ __device__ __forceinline__ bool operator>=(const uint128_t& r)
	{
		if (high > r.high)  // if high bits are greater, we don't need to check low bits
			return true;
		else if (high < r.high)  // if high bits are smaller, we don't need to check low bits
			return false;
		else if (low >= r.low)  // only check low bits, if high bits are equal to each other
			return true;
		else
			return false;
	}
};


/* ZA WARUDOOOOOOOO */
__device__ __forceinline__ uint128_t mul64x2(const unsigned long long& a, const unsigned long long& b)  // c = a*b
{  // PTX assembly for c = a*b
	uint4 res;
	uint128_t c;

	//Divide a and b into 2 32-bit unsigned integers, and perform 32-bit multiplication and addition operations.
	//Store the 128-bit result in 4 32-bit registers and later merge them.

	/*
	GLOSSARY

	mul: multiply
	mad: multiply and add. first 2 operands are multiplied, last operand is added

	.lo .hi: return lower/higher 32 bits of the resulting value
	u32: 32-bit operands

	.cc: if there is a carry, set a carry flag
	addc, madc, mulc, etc.: if there is carry flag set, consume it and act accordingly
	*/

	/*
	Divide a and b into 2 32-bit unsigned integers, and perform 32-bit multiplication and addition operations.
	Perform karatsuba multiplication in assembly.
	Store the 128-bit result in 4 32-bit registers and later merge them.
	*/

	asm("{\n\t"
		/*
				result register
						|
						|   ________ operand registers
						|   |   |
						V   V   V                                            */
		"mul.lo.u32      %3, %5, %7;    \n\t"  // %3 = low bits of (a.low * b.low) 
		"mul.hi.u32      %2, %5, %7;    \n\t"  // %2 = high bits of (a.low * b.low)
		"mad.lo.cc.u32   %2, %4, %7, %2;\n\t"  // %2 = low bits of  (a.high * b.low) + %2 -> if carry occurs, flag it
		"madc.hi.u32     %1, %4, %7,  0;\n\t"  // %1 = high bits of (a.high * b.low) + 0  -> if carry flag is set, handle it
		"mad.lo.cc.u32   %2, %5, %6, %2;\n\t"  // %2 = low bits of (a.low * b.high) + %2 -> if carry occurs, flag it
		"madc.hi.cc.u32  %1, %5, %6, %1;\n\t"  // %1 = high bits of (a.low * b.high) + %1 -> if carry flag is set, handle it; and also if carry occurs, flag it
		"madc.hi.u32     %0, %4, %6,  0;\n\t"  // %0 = high bits of (a.high * b.high) + 0 -> if carry flag is set, handle it
		"mad.lo.cc.u32   %1, %4, %6, %1;\n\t"  // %1 = low bits of (a.high * b.high) + %1 -> if carry occurs, flag it
		"addc.u32        %0, %0, 0;     \n\t"  // %0 = %0 + 0 (this is just to add the final possible carry) -> if carry flag is set, handle it
		"}"
		: "=r"(res.x), "=r"(res.y), "=r"(res.z), "=r"(res.w)  // result registers (put '=' before them)
		//		%0		     %1           %2           %3
		: "r"((unsigned)(a >> 32)), "r"((unsigned)a), "r"((unsigned)(b >> 32)), "r"((unsigned)b));  // input registers
//          		%4						  %5				%6						  %7
//         high 32-bits of a        low 32-bits of a     high 32-bits of b       low 32-bits of b

	c.high = ((unsigned long long)res.x << 32) + res.y;
	c.low = ((unsigned long long)res.z << 32) + res.w;
	//Merge the result into 128-bit unsigned integer.

	return c;
}