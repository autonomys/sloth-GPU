// C++ IMPORTS 
#include <iostream>
#include <bitset>

// CUSTOM IMPORTS
#include "uint128.h"

using namespace std;

__global__ void device_mul(unsigned long long* a, unsigned long long* b, uint128_t* dres)
{
	*dres = mul64x2(*a, *b);
}

int main()
{
	unsigned long long a, b;

	a = 1289312831239555555;
	b = 1290390120391092390;
	uint128_t res;

	unsigned long long * da, * db;
	uint128_t * dres;
	cudaMalloc(&da, sizeof(unsigned long long));
	cudaMalloc(&db, sizeof(unsigned long long));
	cudaMalloc(&dres, 2 * sizeof(unsigned long long));

	cudaMemcpy(da, &a, sizeof(unsigned long long), cudaMemcpyHostToDevice);
	cudaMemcpy(db, &b, sizeof(unsigned long long), cudaMemcpyHostToDevice);

	device_mul << <1, 1 >> > (da, db, dres);

	cudaMemcpy(&res, dres, 2 * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

	cout << bitset<64>(res.high) << endl << bitset<64>(res.low) << endl;

	return 0;
}