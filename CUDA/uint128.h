#pragma once

// TO-DO: add includes

class uint128_t
{
public:
	
	unsigned long long low;
	unsigned long long high;

	__host__ __device__ __forceinline__ uint128_t()
	{
		low = 0;
		high = 0;
	}

	__host__ __device__ __forceinline__ uint128_t(const uint64_t& x)
	{
		low = x;
		high = 0;
	}

	__host__ __device__ __forceinline__ void operator=(const uint128_t& r)
	{
		low = r.low;
		high = r.high;
	}

	__host__ __device__ __forceinline__ void operator=(const uint64_t& r)
	{
		low = r;
		high = 0;
	}

	__host__ __device__ __forceinline__ uint128_t operator<<(const unsigned& shift)
	{
		uint128_t z;

		z.high = high << shift;
		z.high = (low >> (64 - shift)) | z.high;
		z.low = low << shift;

		return z;
	}

	__host__ __device__ __forceinline__ uint128_t operator>>(const unsigned& shift)
	{
		uint128_t z;

		z.low = low >> shift;
		z.low = (high << (64 - shift)) | z.low;
		z.high = high >> shift;

		return z;
	}

};


__host__ int log_2_128(const uint128_t& x)
{
	int z = 0;

	if (x.high != 0)
		z = log2((double)x.high) + 64;
	else
		z = log2((double)x.low);

	return z;
}

__host__ __device__ __forceinline__ static void operator<<=(uint128_t& x, const unsigned& shift)
{
	x.low = x.low >> shift;
	x.low = (x.high << (64 - shift)) | x.low;
	x.high = x.high >> shift;

}

__host__ __device__ __forceinline__ bool operator==(const uint128_t& l, const uint128_t& r)
{
	if ((l.low == r.low) && (l.high == r.high))
		return true;
	else
		return false;
}

__host__ __device__ __forceinline__ bool operator<(const uint128_t& l, const uint128_t& r)
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

__host__ __device__ __forceinline__ bool operator<(const uint128_t& l, const uint64_t& r)
{
	if (l.high != 0)
		return false;
	else if (l.low > r)
		return false;
	else
		return true;
}

__host__ __device__ __forceinline__ bool operator>(const uint128_t& l, const uint128_t& r)
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

__host__ __device__ __forceinline__ uint128_t operator|(const uint128_t& x, const uint128_t& y)
{
	uint128_t z;

	z.low = x.low | y.low;
	z.high = x.high | y.high;

	return z;
}

__host__ __device__ __forceinline__ bool operator!=(const uint128_t& r, const int& l)
{
	if (r.high == 0 && r.low == l)
		return true;
	else
		return false;
}

__host__ __device__ __forceinline__ bool operator<=(const uint128_t& l, const uint128_t& r)
{
	if (l.high < r.high)
		return true;
	else if (l.high > r.high)
		return false;
	else if (l.low <= r.low)
		return true;
	else
		return false;
}

__host__ __device__ __forceinline__ bool operator>=(const uint128_t& l, const uint128_t& r)
{
	if (l.high > r.high)
		return true;
	else if (l.high < r.high)
		return false;
	else if (l.low >= r.low)
		return true;
	else
		return false;
}

__host__ __device__ __forceinline__ uint128_t operator+(const uint128_t& x, const uint128_t& y)
{
	uint128_t z;

	z.low = x.low + y.low;
	z.high = x.high + y.high + (z.low < x.low);

	return z;
}

__host__ __device__ __forceinline__ uint128_t operator+(const uint128_t& x, const uint64_t& y)
{
	uint128_t z;

	z.low = x.low + y;
	z.high = x.high + (z.low < x.low);

	return z;
}

__host__ __device__ __forceinline__ uint128_t operator-(const uint128_t& x, const uint128_t& y)
{
	uint128_t z;

	z.low = x.low - y.low;
	z.high = x.high - y.high - (x.low < y.low);

	return z;
	
}

__host__ __device__ __forceinline__ void operator-=(uint128_t& x, const uint128_t& y)
{
	x.high = x.high - y.high - (x.low < y.low);
	x.low = x.low - y.low;
}

__host__ __device__ __forceinline__ uint128_t operator-(const uint128_t& x, const uint64_t& y)
{
	uint128_t z;

	z.low = x.low - y;
	z.high = x.high - (x.low < y);

	return z;

}

__host__ inline static uint128_t host64x2(const uint64_t& x, const uint64_t& y)
{
	uint128_t z;

	uint128_t ux(x);
	uint128_t uy(y);

	int shift = 0;

	while (uy.low != 0)
	{
		if (uy.low & 1)
		{
			if (shift == 0)
				z = z + ux;
			else
				z = z + (ux << shift);
		}

		shift++;

		uy = uy >> 1;

	}

	return z;
}

__device__ __forceinline__ uint128_t mul64x2(const unsigned long long& a, const unsigned long long& b)
{
	uint4 res;
	uint128_t c;

	asm("{\n\t"
		"mul.lo.u32      %3, %5, %7;    \n\t"
		"mul.hi.u32      %2, %5, %7;    \n\t" //alow * blow
		"mad.lo.cc.u32   %2, %4, %7, %2;\n\t"
		"madc.hi.u32     %1, %4, %7,  0;\n\t" //ahigh * blow
		"mad.lo.cc.u32   %2, %5, %6, %2;\n\t"
		"madc.hi.cc.u32  %1, %5, %6, %1;\n\t" //alow * bhigh
		"madc.hi.u32     %0, %4, %6,  0;\n\t"
		"mad.lo.cc.u32   %1, %4, %6, %1;\n\t" //ahigh * bhigh
		"addc.u32        %0, %0, 0;     \n\t" //add final carry
		"}"
		: "=r"(res.x), "=r"(res.y), "=r"(res.z), "=r"(res.w)
		: "r"((unsigned)(a >> 32)), "r"((unsigned)a), "r"((unsigned)(b >> 32)), "r"((unsigned)b));

	c.high = ((unsigned long long)res.x << 32) + res.y;
	c.low = ((unsigned long long)res.z << 32) + res.w;

	return c;
}