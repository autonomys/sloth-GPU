// C++ IMPORTS 
#include <iostream>
#include <bitset>

// CUSTOM IMPORTS
#include "uint512.h"

using namespace std;

__global__ void device_rshift(uint512_t *d_e, uint512_t *d_res) {
	*d_res = *d_e >> 1;
}

__global__ void device_sub_512_512(uint512_t *d_a, uint512_t *d_c, uint512_t *d_res) {
	*d_res = *d_a - *d_c;
}

__global__ void device_sub_512_256(uint512_t *d_a, uint256_t *d_d, uint512_t *d_res) {
	*d_res = *d_a - *d_d;
}

__global__ void device_mul_257_256(uint512_t *d_mu, uint256_t *d_b, uint512_t *d_res) {
	*d_res = mul257_256(*d_mu, *d_b);
}

__global__ void device_mul_256x2(uint256_t *d_b, uint256_t *d_x, uint512_t *d_res) {
	*d_res = mul256x2(*d_b, *d_x);
}

int main()
{
	uint512_t a, *d_a, c, *d_c, e, *d_e, mu, *d_mu, res, *d_res, test, *d_test;
	uint256_t b, *d_b, d, *d_d;


	//cout << std::bitset<64>(std::string("100")).to_ullong() << endl;

	mu.low.low.low = std::bitset<64>(std::string("0000000000000000000000000000000000000000000000000000000010111101")).to_ullong();
	mu.high.low.low = 1;

	test.low.low.low = std::bitset<64>(std::string("0110100000010110101100100010111110011011111101011000011100111001")).to_ullong();
	test.low.low.high = std::bitset<64>(std::string("1101010001110100000010011110100001101111111111000011001001011010")).to_ullong();
	test.low.high.low = std::bitset<64>(std::string("1010011111000100000010110101110001110111001110111000111110000001")).to_ullong();
	test.low.high.high = std::bitset<64>(std::string("0000000011110000010101100101001101101011010001011010001010110000")).to_ullong();
	test.high.low.low = 1;

	b.low.low = std::bitset<64>(std::string("0110100000010110101100100010111110011011111101011000011100111001")).to_ullong();
	b.low.high = std::bitset<64>(std::string("1101010001110100000010011110100001101111111111000011001001011010")).to_ullong();
	b.high.low = std::bitset<64>(std::string("1010011111000100000010110101110001110111001110111000111110000001")).to_ullong();
	b.high.high = std::bitset<64>(std::string("1111111011110000010101100101001101101011010001011010001010110000")).to_ullong();

	a.high.low.low = 1;  // only 257th bit is 1
	c.low.low.low = 1;  // 1
	d.low.low = 1;  // 1

	e.low.low.low = std::bitset<64>(std::string("0110100000010110101100100010111110011011111101011000011100111001")).to_ullong();
	e.low.low.high = std::bitset<64>(std::string("1101010001110100000010011110100001101111111111000011001001011010")).to_ullong();
	e.low.high.low = std::bitset<64>(std::string("1010011111000100000010110101110001110111001110111000111110000001")).to_ullong();
	e.low.high.high = std::bitset<64>(std::string("1111111011110000010101100101001101101011010001011010001010110000")).to_ullong();
	e.high.low.low = std::bitset<64>(std::string("0110100000010110101100100010111110011011111101011000011100111001")).to_ullong();
	e.high.low.high = std::bitset<64>(std::string("1101010001110100000010011110100001101111111111000011001001011010")).to_ullong();
	e.high.high.low = std::bitset<64>(std::string("1010011111000100000010110101110001110111001110111000111110000001")).to_ullong();
	e.high.high.high = std::bitset<64>(std::string("1111111011110000010101100101001101101011010001011010001010110000")).to_ullong();

	cudaMalloc(&d_a, 8 * sizeof(unsigned long long));
	cudaMalloc(&d_c, 8 * sizeof(unsigned long long));
	cudaMalloc(&d_e, 8 * sizeof(unsigned long long));
	cudaMalloc(&d_mu, 8 * sizeof(unsigned long long));
	cudaMalloc(&d_test, 8 * sizeof(unsigned long long));
	cudaMalloc(&d_res, 8 * sizeof(unsigned long long));
	cudaMalloc(&d_b, 4 * sizeof(unsigned long long));
	cudaMalloc(&d_d, 4 * sizeof(unsigned long long));

	cudaMemcpy(d_a, &a, 8 * sizeof(unsigned long long), cudaMemcpyHostToDevice);
	cudaMemcpy(d_c, &c, 8 * sizeof(unsigned long long), cudaMemcpyHostToDevice);
	cudaMemcpy(d_e, &e, 8 * sizeof(unsigned long long), cudaMemcpyHostToDevice);
	cudaMemcpy(d_mu, &mu, 8 * sizeof(unsigned long long), cudaMemcpyHostToDevice);
	cudaMemcpy(d_test, &test, 8 * sizeof(unsigned long long), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, &b, 4 * sizeof(unsigned long long), cudaMemcpyHostToDevice);
	cudaMemcpy(d_d, &d, 4 * sizeof(unsigned long long), cudaMemcpyHostToDevice);

	cout << "`>>` test start" << endl;

	device_rshift << <1, 1 >> > (d_e, d_res);
	cudaMemcpy(&res, d_res, 8 * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	cout << "Right shift - input print:\n" << bitset<64>(e.high.high.high) << bitset<64>(e.high.high.low) << bitset<64>(e.high.low.high) << bitset<64>(e.high.low.low)
		<< bitset<64>(e.low.high.high) << bitset<64>(e.low.high.low) << bitset<64>(e.low.low.high) << bitset<64>(e.low.low.low) << endl;
	cout << "Right shift - result print:\n" << bitset<64>(res.high.high.high) << bitset<64>(res.high.high.low) << bitset<64>(res.high.low.high) << bitset<64>(res.high.low.low)
		<< bitset<64>(res.low.high.high) << bitset<64>(res.low.high.low) << bitset<64>(res.low.low.high) << bitset<64>(res.low.low.low) << endl;

	cout << "`>>` test end" << endl;

	cout << "/////////////////////////" << endl;

	cout << "`- (512-512)` test start" << endl;

	device_sub_512_512 << <1, 1 >> > (d_a, d_c, d_res);
	cudaMemcpy(&res, d_res, 8 * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	cout << "First operand print:\n" << bitset<64>(a.high.high.high) << bitset<64>(a.high.high.low) << bitset<64>(a.high.low.high) << bitset<64>(a.high.low.low)
		<< bitset<64>(a.low.high.high) << bitset<64>(a.low.high.low) << bitset<64>(a.low.low.high) << bitset<64>(a.low.low.low) << endl << endl;
	cout << "Second operand print:\n" << bitset<64>(c.high.high.high) << bitset<64>(c.high.high.low) << bitset<64>(c.high.low.high) << bitset<64>(c.high.low.low)
		<< bitset<64>(c.low.high.high) << bitset<64>(c.low.high.low) << bitset<64>(c.low.low.high) << bitset<64>(c.low.low.low) << endl;
	cout << "Result print:\n" << bitset<64>(res.high.high.high) << bitset<64>(res.high.high.low) << bitset<64>(res.high.low.high) << bitset<64>(res.high.low.low)
		<< bitset<64>(res.low.high.high) << bitset<64>(res.low.high.low) << bitset<64>(res.low.low.high) << bitset<64>(res.low.low.low) << endl;

	cout << "`- (512-512)` test end" << endl;

	cout << "/////////////////////////" << endl;

	cout << "`- (512-256)` test start" << endl;

	device_sub_512_256 << <1, 1 >> > (d_a, d_d, d_res);
	cudaMemcpy(&res, d_res, 8 * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	cout << "First operand print:\n" << bitset<64>(a.high.high.high) << bitset<64>(a.high.high.low) << bitset<64>(a.high.low.high) << bitset<64>(a.high.low.low)
		<< bitset<64>(a.low.high.high) << bitset<64>(a.low.high.low) << bitset<64>(a.low.low.high) << bitset<64>(a.low.low.low) << endl << endl;
	cout << "Second operand print:\n" << bitset<64>(d.high.high) << bitset<64>(d.high.low) << bitset<64>(d.low.high) << bitset<64>(d.low.low) << endl;
	cout << "Result print:\n" << bitset<64>(res.high.high.high) << bitset<64>(res.high.high.low) << bitset<64>(res.high.low.high) << bitset<64>(res.high.low.low)
		<< bitset<64>(res.low.high.high) << bitset<64>(res.low.high.low) << bitset<64>(res.low.low.high) << bitset<64>(res.low.low.low) << endl;

	cout << "`- (512-256)` test end" << endl;

	cout << "/////////////////////////" << endl;

	cout << "`mul 256-257` test start" << endl;

	device_mul_257_256 << < 1, 1 >> > (d_mu, d_b, d_res);
	cudaMemcpy(&res, d_res, 8 * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	cout << "First operand print:\n" << bitset<64>(mu.high.high.high) << bitset<64>(mu.high.high.low) << bitset<64>(mu.high.low.high) << bitset<64>(mu.high.low.low)
		<< bitset<64>(mu.low.high.high) << bitset<64>(mu.low.high.low) << bitset<64>(mu.low.low.high) << bitset<64>(mu.low.low.low) << endl;

	cout << "Second operand print:\n" << bitset<64>(b.high.high) << bitset<64>(b.high.low) << bitset<64>(b.low.high) << bitset<64>(b.low.low) << endl;

	cout << "Result print:\n" << bitset<64>(res.high.high.high) << bitset<64>(res.high.high.low) << bitset<64>(res.high.low.high) << bitset<64>(res.high.low.low)
		<< bitset<64>(res.low.high.high) << bitset<64>(res.low.high.low) << bitset<64>(res.low.low.high) << bitset<64>(res.low.low.low) << endl;

	cout << "`mul 256-257` 1st test end" << endl;

	cout << "/////////////////////////" << endl;

	cout << "`mul 256-257` 2nd test start" << endl;

	device_mul_257_256 << < 1, 1 >> > (d_test, d_b, d_res);
	cudaMemcpy(&res, d_res, 8 * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	cout << "First operand print:\n" << bitset<64>(test.high.high.high) << bitset<64>(test.high.high.low) << bitset<64>(test.high.low.high) << bitset<64>(test.high.low.low)
		<< bitset<64>(test.low.high.high) << bitset<64>(test.low.high.low) << bitset<64>(test.low.low.high) << bitset<64>(test.low.low.low) << endl;

	cout << "Second operand print:\n" << bitset<64>(b.high.high) << bitset<64>(b.high.low) << bitset<64>(b.low.high) << bitset<64>(b.low.low) << endl;

	cout << "Result print:\n" << bitset<64>(res.high.high.high) << bitset<64>(res.high.high.low) << bitset<64>(res.high.low.high) << bitset<64>(res.high.low.low)
		<< bitset<64>(res.low.high.high) << bitset<64>(res.low.high.low) << bitset<64>(res.low.low.high) << bitset<64>(res.low.low.low) << endl;

	cout << "`mul 256-257` 2nd test end" << endl;

	cout << "/////////////////////////" << endl;


	cout << "`mul 256x2` test start" << endl;

	device_mul_256x2 << < 1, 1 >> > (d_b, d_b, d_res);
	cudaMemcpy(&res, d_res, 8 * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	cout << "First operand print:\n" << bitset<64>(b.high.high) << bitset<64>(b.high.low) << bitset<64>(b.low.high) << bitset<64>(b.low.low) << endl;

	cout << "Second operand print:\n" << bitset<64>(b.high.high) << bitset<64>(b.high.low) << bitset<64>(b.low.high) << bitset<64>(b.low.low) << endl;

	cout << "Result print:\n" << bitset<64>(res.high.high.high) << bitset<64>(res.high.high.low) << bitset<64>(res.high.low.high) << bitset<64>(res.high.low.low)
		<< bitset<64>(res.low.high.high) << bitset<64>(res.low.high.low) << bitset<64>(res.low.low.high) << bitset<64>(res.low.low.low) << endl;

	cout << "`mul 256x2` test end" << endl;

	cout << "/////////////////////////" << endl;

	return 0;
}