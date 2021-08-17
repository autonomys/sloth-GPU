#include "sloth256_189.c"

__kernel void hello(__global char* data, __global char* iv)
{
	/*unsigned char iv[32];
	for (int i = 0; i < 32; i++)
	{
		iv[i] = 3;
	}

	for (int i = 0; i < 4096; i++)
	{
		data[i] = 5;
	}*/

	sloth256_189_encode(data, 4096, iv, 1);
	
}