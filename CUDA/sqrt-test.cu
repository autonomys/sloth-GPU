// C++ IMPORTS 
#include <iostream>
#include <bitset>

// CUSTOM IMPORTS
#include "sqrt_library.h"

using namespace std;

int main()
{
	uint256_t p, *d_p, a, *d_a, exp, *d_exp;
	uint512_t mu, *d_mu;
	unsigned k = 256;

	p.high.high = std::bitset<64>(std::string("1111111111111111111111111111111111111111111111111111111111111111")).to_ullong();
	p.high.low = std::bitset<64>(std::string("1111111111111111111111111111111111111111111111111111111111111111")).to_ullong();
	p.low.high = std::bitset<64>(std::string("1111111111111111111111111111111111111111111111111111111111111111")).to_ullong();
	p.low.low = std::bitset<64>(std::string("1111111111111111111111111111111111111111111111111111111101000011")).to_ullong();

	mu.low.low.low = std::bitset<64>(std::string("0000000000000000000000000000000000000000000000000000000010111101")).to_ullong();
	mu.high.low.low = 1ull;

	a.high.high = std::bitset<64>(std::string("1111111111111111111111111111111111111111111111111111111111111111")).to_ullong();
	a.high.low = std::bitset<64>(std::string("1111111111111111111111111111111111111111111111111111111111111111")).to_ullong();
	a.low.high = std::bitset<64>(std::string("1111111111111111111111111111111111111111111111111111111111111111")).to_ullong();
	a.low.low = std::bitset<64>(std::string("1111111111111111111111111111111111111111111111111111111100111100")).to_ullong();

	exp.high.high = std::bitset<64>(std::string("0011111111111111111111111111111111111111111111111111111111111111")).to_ullong();
	exp.high.low = std::bitset<64>(std::string("1111111111111111111111111111111111111111111111111111111111111111")).to_ullong();
	exp.low.high = std::bitset<64>(std::string("1111111111111111111111111111111111111111111111111111111111111111")).to_ullong();
	exp.low.low = std::bitset<64>(std::string("1111111111111111111111111111111111111111111111111111111111010001")).to_ullong();


	cudaMalloc(&d_p, 4 * (sizeof(unsigned long long)));
	cudaMalloc(&d_mu, 8 * (sizeof(unsigned long long)));
	cudaMalloc(&d_a, 4 * (sizeof(unsigned long long)));
	cudaMalloc(&d_exp, 4 * (sizeof(unsigned long long)));

	cudaMemcpy(d_p, &p, 4 * (sizeof(unsigned long long)), cudaMemcpyHostToDevice);
	cudaMemcpy(d_mu, &mu, 8 * (sizeof(unsigned long long)), cudaMemcpyHostToDevice);
	cudaMemcpy(d_a, &a, 4 * (sizeof(unsigned long long)), cudaMemcpyHostToDevice);
	cudaMemcpy(d_exp, &exp, 4 * (sizeof(unsigned long long)), cudaMemcpyHostToDevice);

	//cout << "a: " << bitset<64>(a.high.high) << bitset<64>(a.high.low) << bitset<64>(a.low.high) << bitset<64>(a.low.low) << endl;
	//cout << "p: " << bitset<64>(p.high.high) << bitset<64>(p.high.low) << bitset<64>(p.low.high) << bitset<64>(p.low.low) << endl;

	cout << "Square-root permutation test ...";
	sqrt_permutation << <1, 1 >> > (d_a, exp, p, mu, k);
	cudaMemcpy(&a, d_a, 4 * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	if (bitset<64>(a.high.high) == std::bitset<64>(std::string("0101001110000101000111011100000110110011100000101100011000001000")) &&
		bitset<64>(a.high.low) == std::bitset<64>(std::string("0010000010111000010011001110101001110011111011111001000100111100")) &&
		bitset<64>(a.low.high) == std::bitset<64>(std::string("0001010000111011111111001001010100011110001110011010001110101101")) &&
		bitset<64>(a.low.low) == std::bitset<64>(std::string("1101011000011000111100111100100011101011110011110101100110111000"))
		) {

		printf("passed!\n");
	}
	else
	{
		printf("failed!\n");
	}

	//cout << bitset<64>(a.high.high) << endl << bitset<64>(a.high.low) << endl << bitset<64>(a.low.high) << endl << bitset<64>(a.low.low) << endl;

	return 0;
}