#include "sloth256_189.c"

__kernel void throughput_kernel(__global unsigned char* data, __global unsigned char* iv)
{
	int i = get_global_id(0);
	sloth256_189_encode(data + i * 4096, 4096, iv, 1);
}