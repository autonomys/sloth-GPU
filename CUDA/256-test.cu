// C++ IMPORTS 
#include <iostream>
#include <bitset>

// CUSTOM IMPORTS
#include "uint256.h"

using namespace std;

__global__ void device_mul(uint128_t* a, uint128_t* b, uint256_t* dres)
{
	*dres = mul128x2(*a, *b);
}
 
__global__ void device_equals(uint256_t* h1, uint256_t* h2, uint256_t * h3)
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

__global__ void device_greater(uint256_t* h1, uint256_t* h2, uint256_t * h3)
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

__global__ void device_lesser(uint256_t* h1, uint256_t* h2, uint256_t * h3)
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

__global__ void device_rshift(uint256_t* h3, uint256_t *dres)
{
	*dres = *h3 >> 1;
}

__global__ void device_lshift(uint256_t* h4, uint256_t *dres)
{
	*dres = *h4 << 1;
}

__global__ void device_sub_256_256(uint256_t* h6, uint256_t* h7, uint256_t *dres)
{
	*dres = *h6 - *h7;
}

__global__ void device_sub_256_64(uint256_t* h6, uint256_t *dres)
{
	*dres = *h6 - 1;
}

__global__ void device_add(uint256_t* h5, uint256_t* h7, uint256_t *dres)
{
	*dres = *h5 + *h7;
}

__global__ void device_bitwise_or(uint256_t* h5, uint256_t* h7, uint256_t *dres)
{
	*dres = *h5 | *h7;
}

__global__ void device_even(uint256_t* h5)
{
	if (isEven(*h5)) {
		printf("fail for isEven\n");
	}
	else {
		printf("success for isEven\n");
	}
}


__global__ void device_odd(uint256_t* h5)
{
	if (isOdd(*h5)) {
		printf("success for isOdd\n");
	}
	else {
		printf("c for isOdd\n");
	}
}



int main()
{
	// INITIALIZATION

	uint256_t h1, h2, h3, h4, h5, h6, h7, res;
	uint256_t *d1, *d2, *d3, *d4, *d5, *d6, *d7, *dres;

	uint128_t a, b;
	uint128_t *da, *db;


	cudaMalloc(&d1, 4 * sizeof(unsigned long long));
	cudaMalloc(&d2, 4 * sizeof(unsigned long long));
	cudaMalloc(&d3, 4 * sizeof(unsigned long long));
	cudaMalloc(&d4, 4 * sizeof(unsigned long long));
	cudaMalloc(&d5, 4 * sizeof(unsigned long long));
	cudaMalloc(&d6, 4 * sizeof(unsigned long long));
	cudaMalloc(&d7, 4 * sizeof(unsigned long long));
	cudaMalloc(&da, 2 * sizeof(unsigned long long));
	cudaMalloc(&db, 2 * sizeof(unsigned long long));
	cudaMalloc(&dres, 4 * sizeof(unsigned long long));

	cout << "Multiplication test start" << endl;

	a.low = 1212982123918293; a.high = 2340000000081293891;
	b.low = 99236875620028271; b.high = 3546200000189283900;

	cudaMemcpy(da, &a, 2 * sizeof(unsigned long long), cudaMemcpyHostToDevice);
	cudaMemcpy(db, &b, 2 * sizeof(unsigned long long), cudaMemcpyHostToDevice);

	device_mul << <1, 1 >> > (da, db, dres);

	cudaMemcpy(&res, dres, 4 * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

	cout << "Multiplication operand1:\n" << bitset<64>(a.high) << bitset<64>(a.low) << endl;
	cout << "Multiplication operand2:\n" << bitset<64>(b.high) << bitset<64>(b.low) << endl;

	cout << "Multiplication test result:\n" << bitset<64>(res.high.high) << bitset<64>(res.high.low) << bitset<64>(res.low.high) << bitset<64>(res.low.low) << endl;

	cout << "Multiplication test end" << endl;

	// PREPERATION FOR OTHER OPERATIONS start

	h1.high.high = std::bitset<64>(std::string("1111111111111111111111111111111111111111111111111111111111111111")).to_ullong();
	h1.high.low = std::bitset<64>(std::string("1111111111111111111111110011111111111111001111100111111111111000")).to_ullong();
	h1.low.high = std::bitset<64>(std::string("1111111111111111111111111111111111111111111111111111111111111111")).to_ullong();
	h1.low.low = std::bitset<64>(std::string("1111111111111111111111110011111111111111001111100111111111111000")).to_ullong();

	h2.high.high = std::bitset<64>(std::string("1111111111111111111111111111111111111111111111111111111111111111")).to_ullong();
	h2.high.low = std::bitset<64>(std::string("1111111111111111111111110011111111111111001111100111111111111100")).to_ullong();
	h2.low.high = std::bitset<64>(std::string("1111111111111111111111111111111111111111111111111111111111111111")).to_ullong();
	h2.low.low = std::bitset<64>(std::string("1111111111111111111111110011111111111111001111100111111111111100")).to_ullong();

	h3.high.high = std::bitset<64>(std::string("1111111111111111111111111111111111111111111111111111111111111111")).to_ullong();
	h3.high.low = std::bitset<64>(std::string("1111111111111111111111110011111111111111001111100111111111111100")).to_ullong();
	h3.low.high = std::bitset<64>(std::string("1111111111111111111111111111111111111111111111111111111111111111")).to_ullong();
	h3.low.low = std::bitset<64>(std::string("1111111111111111111111110011111111111111001111100111111111111100")).to_ullong();

	h4.high.high = std::bitset<64>(std::string("1111111111111111111111110011111111111111001111100111111111111100")).to_ullong();
	h4.high.low = std::bitset<64>(std::string("1111111111111111111111110011111111111111001111100111111111111100")).to_ullong();
	h4.low.high = std::bitset<64>(std::string("1111111111111111111111110011111111111111001111100111111111111100")).to_ullong();
	h4.low.low = std::bitset<64>(std::string("1111111111111111111111110011111111111111001111100111111111111100")).to_ullong();

	h5.high.high = std::bitset<64>(std::string("0111111111111111111111111111111111111111111111111111111111111111")).to_ullong();
	h5.high.low = std::bitset<64>(std::string("1111111111111111111111111111111111111111111111111111111111111111")).to_ullong();
	h5.low.high = std::bitset<64>(std::string("1111111111111111111111111111111111111111111111111111111111111111")).to_ullong();
	h5.low.low = std::bitset<64>(std::string("1111111111111111111111111111111111111111111111111111111111111111")).to_ullong();

	h6.high.high = std::bitset<64>(std::string("1000000000000000000000000000000000000000000000000000000000000000")).to_ullong();

	h7 = 1;

	a = 1289312831239555555;
	b = 18446743249063149564;

	cudaMemcpy(d1, &h1, 4 * sizeof(unsigned long long), cudaMemcpyHostToDevice);
	cudaMemcpy(d2, &h2, 4 * sizeof(unsigned long long), cudaMemcpyHostToDevice);
	cudaMemcpy(d3, &h3, 4 * sizeof(unsigned long long), cudaMemcpyHostToDevice);
	cudaMemcpy(d4, &h4, 4 * sizeof(unsigned long long), cudaMemcpyHostToDevice);
	cudaMemcpy(d5, &h5, 4 * sizeof(unsigned long long), cudaMemcpyHostToDevice);
	cudaMemcpy(d6, &h6, 4 * sizeof(unsigned long long), cudaMemcpyHostToDevice);
	cudaMemcpy(d7, &h7, 4 * sizeof(unsigned long long), cudaMemcpyHostToDevice);
	cudaMemcpy(da, &a, 2 * sizeof(unsigned long long), cudaMemcpyHostToDevice);
	cudaMemcpy(db, &b, 2 * sizeof(unsigned long long), cudaMemcpyHostToDevice);

	// PREPERATION FOR OTHER OPERATIONS end

	cout << "/////////////////////////" << endl;

	cout << "`==` test start" << endl;

	device_equals << <1, 1 >> > (d1, d2, d3);
	cudaDeviceSynchronize();

	cout << "`== (256-256)` test end" << endl;

	cout << "/////////////////////////" << endl;

	cout << "`>` test start" << endl;

	device_greater << <1, 1 >> > (d1, d2, d3);
	cudaDeviceSynchronize();

	cout << "`>` test end" << endl;

	cout << "/////////////////////////" << endl;

	cout << "`< (256-256)` test start" << endl;

	device_lesser << <1, 1 >> > (d1, d2, d3);
	cudaDeviceSynchronize();

	cout << "`< (256-256)` test end" << endl;

	cout << "/////////////////////////" << endl;

	cout << "`>>` test start" << endl;

	device_rshift << <1, 1 >> > (d3, dres);
	cudaMemcpy(&res, dres, 4 * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	cout << "Right shift - input print:\n" << bitset<64>(h3.high.high) << bitset<64>(h3.high.low) << bitset<64>(h3.low.high) << bitset<64>(h3.low.low) << endl;
	cout << "Right shift - result print:\n" << bitset<64>(res.high.high) << bitset<64>(res.high.low) << bitset<64>(res.low.high) << bitset<64>(res.low.low) << endl;

	cout << "`>>` test end" << endl;

	cout << "/////////////////////////" << endl;

	cout << "`<<` test start" << endl;

	device_lshift << <1, 1 >> > (d4, dres);
	cudaMemcpy(&res, dres, 4 * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	cout << "Left shift - input print:\n" << bitset<64>(h4.high.high) << bitset<64>(h4.high.low) << bitset<64>(h4.low.high) << bitset<64>(h4.low.low) << endl;
	cout << "Left shift - result print:\n" << bitset<64>(res.high.high) << bitset<64>(res.high.low) << bitset<64>(res.low.high) << bitset<64>(res.low.low) << endl;

	cout << "`<<` test end" << endl;

	cout << "/////////////////////////" << endl;

	cout << "`- (256-256)` test start" << endl;

	device_sub_256_256 << <1, 1 >> > (d6, d7, dres);
	cudaMemcpy(&res, dres, 4 * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	cout << "Substraction - input1 print:\n" << bitset<64>(h6.high.high) << bitset<64>(h6.high.low) << bitset<64>(h6.low.high) << bitset<64>(h6.low.low) << endl;
	cout << "Substraction - input2 print:\n" << bitset<64>(h7.high.high) << bitset<64>(h7.high.low) << bitset<64>(h7.low.high) << bitset<64>(h7.low.low) << endl;
	cout << "Substraction - result print:\n" << bitset<64>(res.high.high) << bitset<64>(res.high.low) << bitset<64>(res.low.high) << bitset<64>(res.low.low) << endl;

	cout << "`- (256-256)` test end" << endl;

	cout << "/////////////////////////" << endl;

	cout << "`- (256-64)` test start" << endl;

	device_sub_256_64 << <1, 1 >> > (d6, dres);
	cudaMemcpy(&res, dres, 4 * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	cout << "Substraction - input1 print:\n" << bitset<64>(h6.high.high) << bitset<64>(h6.high.low) << bitset<64>(h6.low.high) << bitset<64>(h6.low.low) << endl;
	cout << "Substraction - input2 print:\n" << "1" << endl;
	cout << "Substraction - result print:\n" << bitset<64>(res.high.high) << bitset<64>(res.high.low) << bitset<64>(res.low.high) << bitset<64>(res.low.low) << endl;

	cout << "`- (256-64)` test end" << endl;

	cout << "/////////////////////////" << endl;

	cout << "`+` test start" << endl;

	device_add << <1, 1 >> > (d5, d7, dres);
	cudaMemcpy(&res, dres, 4 * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	cout << "Addition - input1 print:\n" << bitset<64>(h5.high.high) << bitset<64>(h5.high.low) << bitset<64>(h5.low.high) << bitset<64>(h5.low.low) << endl;
	cout << "Addition - input2 print:\n" << bitset<64>(h7.high.high) << bitset<64>(h7.high.low) << bitset<64>(h7.low.high) << bitset<64>(h7.low.low) << endl;
	cout << "Addition - result print:\n" << bitset<64>(res.high.high) << bitset<64>(res.high.low) << bitset<64>(res.low.high) << bitset<64>(res.low.low) << endl;

	cout << "`+` test end" << endl;

	cout << "/////////////////////////" << endl;

	cout << "`|` test start" << endl;

	device_bitwise_or << <1, 1 >> > (d5, d7, dres);
	cudaMemcpy(&res, dres, 4 * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	cout << "Bitwise OR - input1 print:\n" << bitset<64>(h5.high.high) << bitset<64>(h5.high.low) << bitset<64>(h5.low.high) << bitset<64>(h5.low.low) << endl;
	cout << "Bitwise OR - input2 print:\n" << bitset<64>(h7.high.high) << bitset<64>(h7.high.low) << bitset<64>(h7.low.high) << bitset<64>(h7.low.low) << endl;
	cout << "Bitwise OR - result print:\n" << bitset<64>(res.high.high) << bitset<64>(res.high.low) << bitset<64>(res.low.high) << bitset<64>(res.low.low) << endl;

	cout << "`|` test end" << endl;

	cout << "/////////////////////////" << endl;

	cout << "`isEven` test start" << endl;

	device_even << <1, 1 >> > (d5);
	cudaDeviceSynchronize();

	cout << "`isEven` test end" << endl;

	cout << "/////////////////////////" << endl;

	cout << "`isOdd` test start" << endl;

	device_odd << <1, 1 >> > (d5);
	cudaDeviceSynchronize();

	cout << "`isOdd` test end" << endl;



	return 0;
}