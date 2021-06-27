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

__global__ void device_equals(uint128_t* h1, uint128_t* h2, uint128_t * h3)
{   
	/* SETUP
	h1 != h2, 
    h2 = h3
	*/
	if (*h1 == *h2) {
		printf("Fail for h1==h2\n");
	}
	else {
		printf("Success for h1==h2\n");
	}
	if (*h2 == *h3) {
		printf("Success for h2==h3\n");
	}
	else {
		printf("Fail for h2==h3\n");
	}

}

__global__ void device_greater(uint128_t* h1, uint128_t* h2, uint128_t * h3)
{
	/* SETUP
	h1 < h2,
	h2 = h3
	*/
	if (*h1 > *h2) {
		printf("Fail for h1 > h2\n");
	}
	else {
		printf("Success for h1 > h2\n");
	}
	if (*h2 > *h3) {
		printf("Fail for h2 > h3\n");
	}
	else {
		printf("Success for h2 > h3\n");
	}

}

__global__ void device_lesser(uint128_t* h1, uint128_t* h2, uint128_t * h3)
{
	/* SETUP
	h1 < h2,
	h2 = h3
	*/
	if (*h1 < *h2) {
		printf("Success for h1 < h2\n");
	}
	else {
		printf("Fail for h1 < h2\n");
	}
	if (*h2 < *h3) {
		printf("Fail for h2 < h3\n");
	}
	else {
		printf("Success for h2 < h3\n");
	}

}


__global__ void device_lesser_64(uint128_t* h4, unsigned long long* a, unsigned long long* b)
{
	/* SETUP
	h4 < b
	a < h4
	*/
	if (*a < *h4) {
		printf("Success for a < h4\n");
	}
	else {
		printf("Fail for a < h4\n");
	}
	if (*b < *h4) {
		printf("Fail for b < h4\n");
	}
	else {
		printf("Success for b < h4\n");
	}

}

__global__ void device_equals_64(uint128_t* h4, unsigned long long* a, unsigned long long* b)
{
	/* SETUP
	h4 < b
	a < h4
	*/
	if (*a == *h4) {
		printf("Fail for a == h4\n");
	}
	else {
		printf("Success for a == h4\n");
	}
	if (*b == *h4) {
		printf("Success for b == h4\n");
	}
	else {
		printf("Fail for b == h4\n");
	}

}


int main()
{

	// Multiplication test start

	unsigned long long a, b;

	a = 1289312831239555555;
	b = 1290390120391092390;

	uint128_t res;

	unsigned long long *da, *db;
	uint128_t *dres;
	cudaMalloc(&da, sizeof(unsigned long long));
	cudaMalloc(&db, sizeof(unsigned long long));
	cudaMalloc(&dres, 2 * sizeof(unsigned long long));

	cudaMemcpy(da, &a, sizeof(unsigned long long), cudaMemcpyHostToDevice);
	cudaMemcpy(db, &b, sizeof(unsigned long long), cudaMemcpyHostToDevice);

	device_mul << <1, 1 >> > (da, db, dres);

	cudaMemcpy(&res, dres, 2 * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

	cout << "Multiplication test result:\n" << bitset<64>(res.high) << endl << bitset<64>(res.low) << endl;

	// Multiplication test end

	/////////////////////////

	// `==` test start

	uint128_t h1, h2, h3, h4;
	uint128_t *d1, *d2, *d3, *d4;



	h1.high = std::bitset<64>(std::string("1111111111111111111111111111111111111111111111111111111111111111")).to_ullong();
	h1.low = std::bitset<64>(std::string("1111111111111111111111110011111111111111001111100111111111111000")).to_ullong();

	h2.high = std::bitset<64>(std::string("1111111111111111111111111111111111111111111111111111111111111111")).to_ullong();
	h2.low = std::bitset<64>(std::string("1111111111111111111111110011111111111111001111100111111111111100")).to_ullong();

	h3.high = std::bitset<64>(std::string("1111111111111111111111111111111111111111111111111111111111111111")).to_ullong();
	h3.low = std::bitset<64>(std::string("1111111111111111111111110011111111111111001111100111111111111100")).to_ullong();

	h4.low = std::bitset<64>(std::string("1111111111111111111111110011111111111111001111100111111111111100")).to_ullong();  // 18446743249063149564
	
	a = 1289312831239555555;
	b = 18446743249063149564;

	cudaMalloc(&d1, 2 * sizeof(unsigned long long));
	cudaMalloc(&d2, 2 * sizeof(unsigned long long));
	cudaMalloc(&d3, 2 * sizeof(unsigned long long));
	cudaMalloc(&d4, 2 * sizeof(unsigned long long));
	cudaMalloc(&da, sizeof(unsigned long long));
	cudaMalloc(&db, sizeof(unsigned long long));


	cudaMemcpy(d1, &h1, 2 * sizeof(unsigned long long), cudaMemcpyHostToDevice);
	cudaMemcpy(d2, &h2, 2 * sizeof(unsigned long long), cudaMemcpyHostToDevice);
	cudaMemcpy(d3, &h3, 2 * sizeof(unsigned long long), cudaMemcpyHostToDevice);
	cudaMemcpy(d4, &h4, 2 * sizeof(unsigned long long), cudaMemcpyHostToDevice);
	cudaMemcpy(da, &a, sizeof(unsigned long long), cudaMemcpyHostToDevice);
	cudaMemcpy(db, &b, sizeof(unsigned long long), cudaMemcpyHostToDevice);

	device_equals << <1, 1 >> > (d1, d2, d3);

	// `== (128-128)` test end

	/////////////////////////

	// `>` test start

	device_greater << <1, 1 >> > (d1, d2, d3);

	// `>` test end

	/////////////////////////

	// `< (128-128)` test start

	device_lesser << <1, 1 >> > (d1, d2, d3);

	// `< (128-128)` test end

	/////////////////////////

	// `< (128-64)` test start

	device_lesser_64 << <1, 1 >> > (d4, da, db);

	// `< (128-64)` test end

	/////////////////////////

	// `< (128-64)` test start

	device_equals_64 << <1, 1 >> > (d4, da, db);

	// `< (128-64)` test end


	return 0;
}