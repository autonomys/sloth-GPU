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
	if (*h4 > *a) {
		printf("Success for a < h4\n");
	}
	else {
		printf("Fail for a < h4\n");
	}
	if (*h4 > *b) {
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
	if (*h4 == *a) {
		printf("Fail for a == h4\n");
	}
	else {
		printf("Success for a == h4\n");
	}
	if (*h4 == *b) {
		printf("Success for b == h4\n");
	}
	else {
		printf("Fail for b == h4\n");
	}
}

__global__ void device_rshift(uint128_t* h4, uint128_t *dres)
{
	*dres = *h4 >> 1;
}

__global__ void device_lshift(uint128_t* h4, uint128_t *dres)
{
	*dres = *h4 << 1;
}

__global__ void device_sub_128_128(uint128_t* h6, uint128_t* h7, uint128_t *dres)
{
	*dres = *h6 - *h7;
}

__global__ void device_sub_128_64(uint128_t* h6, uint128_t *dres)
{
	*dres = *h6 - 1;
}

__global__ void device_add(uint128_t* h5, uint128_t* h7, uint128_t *dres)
{
	*dres = *h5 + *h7;
}

__global__ void device_bitwise_or(uint128_t* h5, uint128_t* h7, uint128_t *dres)
{
	*dres = *h5 | *h7;
}

__global__ void device_bitwise_xor(uint128_t* h5, uint128_t* h7, uint128_t *dres)
{
	*dres = *h5 ^ *h7;
}


int main()
{
	// INITIALIZATION

	uint128_t h1, h2, h3, h4, h5, h6, h7, res;
	uint128_t *d1, *d2, *d3, *d4, *d5, *d6, *d7, *dres;

	unsigned long long a, b;
	unsigned long long *da, *db;


	cudaMalloc(&d1, 2 * sizeof(unsigned long long));
	cudaMalloc(&d2, 2 * sizeof(unsigned long long));
	cudaMalloc(&d3, 2 * sizeof(unsigned long long));
	cudaMalloc(&d4, 2 * sizeof(unsigned long long));
	cudaMalloc(&d5, 2 * sizeof(unsigned long long));
	cudaMalloc(&d6, 2 * sizeof(unsigned long long));
	cudaMalloc(&d7, 2 * sizeof(unsigned long long));
	cudaMalloc(&da, sizeof(unsigned long long));
	cudaMalloc(&db, sizeof(unsigned long long));
	cudaMalloc(&dres, 2 * sizeof(unsigned long long));


	a = std::bitset<64>(std::string("0110100000010110101100100010111110011011111101011000011100111001")).to_ullong();
	b = std::bitset<64>(std::string("0110100000010110101100100010111110011011111101011000011100111001")).to_ullong();

	cout << "Multiplication test start" << endl;

	//a = 1289312831239555555;
	//b = 1290390120391092390;

	cudaMemcpy(da, &a, sizeof(unsigned long long), cudaMemcpyHostToDevice);
	cudaMemcpy(db, &b, sizeof(unsigned long long), cudaMemcpyHostToDevice);

	device_mul << <1, 1 >> > (da, db, dres);

	cudaMemcpy(&res, dres, 2 * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

	cout << "Multiplication test result:\n" << bitset<64>(res.high) << bitset<64>(res.low) << endl;

	cout << "Multiplication test end" << endl;

	// PREPERATION FOR OTHER OPERATIONS start

	h1.high = std::bitset<64>(std::string("1111111111111111111111111111111111111111111111111111111111111111")).to_ullong();
	h1.low = std::bitset<64>(std::string("1111111111111111111111110011111111111111001111100111111111111000")).to_ullong();

	h2.high = std::bitset<64>(std::string("1111111111111111111111111111111111111111111111111111111111111111")).to_ullong();
	h2.low = std::bitset<64>(std::string("1111111111111111111111110011111111111111001111100111111111111100")).to_ullong();

	h3.high = std::bitset<64>(std::string("1111111111111111111111111111111111111111111111111111111111111111")).to_ullong();
	h3.low = std::bitset<64>(std::string("1111111111111111111111110011111111111111001111100111111111111100")).to_ullong();

	h4.low = std::bitset<64>(std::string("1111111111111111111111110011111111111111001111100111111111111100")).to_ullong();  // 18446743249063149564

	h5.high = std::bitset<64>(std::string("0111111111111111111111111111111111111111111111111111111111111111")).to_ullong();
	h5.low = std::bitset<64>(std::string("1111111111111111111111111111111111111111111111111111111111111111")).to_ullong();

	h6.high = std::bitset<64>(std::string("1000000000000000000000000000000000000000000000000000000000000000")).to_ullong();

	h7 = 1;

	a = 1289312831239555555;
	b = 18446743249063149564;

	cudaMemcpy(d1, &h1, 2 * sizeof(unsigned long long), cudaMemcpyHostToDevice);
	cudaMemcpy(d2, &h2, 2 * sizeof(unsigned long long), cudaMemcpyHostToDevice);
	cudaMemcpy(d3, &h3, 2 * sizeof(unsigned long long), cudaMemcpyHostToDevice);
	cudaMemcpy(d4, &h4, 2 * sizeof(unsigned long long), cudaMemcpyHostToDevice);
	cudaMemcpy(d5, &h5, 2 * sizeof(unsigned long long), cudaMemcpyHostToDevice);
	cudaMemcpy(d6, &h6, 2 * sizeof(unsigned long long), cudaMemcpyHostToDevice);
	cudaMemcpy(d7, &h7, 2 * sizeof(unsigned long long), cudaMemcpyHostToDevice);
	cudaMemcpy(da, &a, sizeof(unsigned long long), cudaMemcpyHostToDevice);
	cudaMemcpy(db, &b, sizeof(unsigned long long), cudaMemcpyHostToDevice);

	// PREPERATION FOR OTHER OPERATIONS end

	cout << "/////////////////////////" << endl;

	cout << "`==` test start" << endl;

	device_equals << <1, 1 >> > (d1, d2, d3);
	cudaDeviceSynchronize();

	cout << "`== (128-128)` test end" << endl;

	cout << "/////////////////////////" << endl;

	cout << "`>` test start" << endl;

	device_greater << <1, 1 >> > (d1, d2, d3);
	cudaDeviceSynchronize();

	cout << "`>` test end" << endl;

	cout << "/////////////////////////" << endl;

	cout << "`< (128-128)` test start" << endl;

	device_lesser << <1, 1 >> > (d1, d2, d3);
	cudaDeviceSynchronize();

	cout << "`< (128-128)` test end" << endl;

	cout << "/////////////////////////" << endl;

	cout << "`< (128-64)` test start" << endl;

	device_lesser_64 << <1, 1 >> > (d4, da, db);
	cudaDeviceSynchronize();

	cout << "`< (128 - 64)` test end" << endl;

	cout << "/////////////////////////" << endl;

	cout << "`== (128-64)` test start" << endl;

	device_equals_64 << <1, 1 >> > (d4, da, db);
	cudaDeviceSynchronize();

	cout << "`== (128-64)` test end" << endl;

	cout << "/////////////////////////" << endl;

	cout << "`>>` test start" << endl;

	device_rshift << <1, 1 >> > (d3, dres);
	cudaMemcpy(&res, dres, 2 * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	cout << "Right shift - input print:\n" << bitset<64>(h3.high) << bitset<64>(h3.low) << endl;
	cout << "Right shift - result print:\n" << bitset<64>(res.high) << bitset<64>(res.low) << endl;

	cout << "`>>` test end" << endl;

	cout << "/////////////////////////" << endl;

	cout << "`<<` test start" << endl;

	device_lshift << <1, 1 >> > (d4, dres);
	cudaMemcpy(&res, dres, 2 * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	cout << "Left shift - input print:\n" << bitset<64>(h4.high) << bitset<64>(h4.low) << endl;
	cout << "Left shift - result print:\n" << bitset<64>(res.high) << bitset<64>(res.low) << endl;

	cout << "`<<` test end" << endl;

	cout << "/////////////////////////" << endl;

	cout << "`- (128-128)` test start" << endl;

	device_sub_128_128 << <1, 1 >> > (d6, d7, dres);
	cudaMemcpy(&res, dres, 2 * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	cout << "Substraction - input1 print:\n" << bitset<64>(h6.high)<< bitset<64>(h6.low) << endl;
	cout << "Substraction - input2 print:\n" << bitset<64>(h7.high) << bitset<64>(h7.low) << endl;
	cout << "Substraction - result print:\n" << bitset<64>(res.high) << bitset<64>(res.low) << endl;

	cout << "`- (128-128)` test end" << endl;
	
	cout << "/////////////////////////" << endl;

	cout << "`- (128-64)` test start" << endl;

	device_sub_128_64 << <1, 1 >> > (d6, dres);
	cudaMemcpy(&res, dres, 2 * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	cout << "Substraction - input1 print:\n" << bitset<64>(h6.high) << bitset<64>(h6.low) << endl;
	cout << "Substraction - input2 print:\n" << "1" << endl;
	cout << "Substraction - result print:\n" << bitset<64>(res.high) << bitset<64>(res.low) << endl;

	cout << "`- (128-64)` test end" << endl;

	cout << "/////////////////////////" << endl;

	cout << "`+` test start" << endl;

	device_add << <1, 1 >> > (d5, d7, dres);
	cudaMemcpy(&res, dres, 2 * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	cout << "Addition - input1 print:\n" << bitset<64>(h5.high) << bitset<64>(h5.low) << endl;
	cout << "Addition - input2 print:\n" << bitset<64>(h7.high) << bitset<64>(h7.low) << endl;
	cout << "Addition - result print:\n" << bitset<64>(res.high) << bitset<64>(res.low) << endl;

	cout << "`+` test end" << endl;

	cout << "/////////////////////////" << endl;

	cout << "`|` test start" << endl;

	device_bitwise_or << <1, 1 >> > (d5, d7, dres);
	cudaMemcpy(&res, dres, 2 * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	cout << "Bitwise OR - input1 print:\n" << bitset<64>(h5.high) << bitset<64>(h5.low) << endl;
	cout << "Bitwise OR - input2 print:\n" << bitset<64>(h7.high) << bitset<64>(h7.low) << endl;
	cout << "Bitwise OR - result print:\n" << bitset<64>(res.high) << bitset<64>(res.low) << endl;

	cout << "`|` test end" << endl;


	cout << "`^` test start" << endl;

	device_bitwise_xor << <1, 1 >> > (d5, d7, dres);
	cudaMemcpy(&res, dres, 2 * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	cout << "Bitwise XOR - input1 print:\n" << bitset<64>(h5.high) << bitset<64>(h5.low) << endl;
	cout << "Bitwise XOR - input2 print:\n" << bitset<64>(h7.high) << bitset<64>(h7.low) << endl;
	cout << "Bitwise XOR - result print:\n" << bitset<64>(res.high) << bitset<64>(res.low) << endl;

	cout << "`^` test end" << endl;

	

	return 0;
}