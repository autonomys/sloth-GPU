// C++ IMPORTS 
#include <iostream>
#include <bitset>
#include <random>

// CUSTOM IMPORTS
#include "encode_library.h"

using namespace std;

std::random_device dev;
std::mt19937_64 rng(dev());

uint256_t random_256()
{
	uint256_t res;

	std::uniform_int_distribution<unsigned long long> randnum(0, UINT64_MAX);

	res.high.high = randnum(rng);
	res.high.low = randnum(rng);
	res.low.high = randnum(rng);
	res.low.low = randnum(rng);

	return res;
}

void random_array_256(uint256_t a[])
{
	std::uniform_int_distribution<unsigned long long> randnum(0, UINT64_MAX);

	for (int i = 0; i < 128; i++)
	{
		a[i] = random_256();
	}
}

int main()
{
	uint256_t a, *d_a;

	a.high.high = std::bitset<64>(std::string("1111111111111111111111111111111111111111111111111111111111111111")).to_ullong();
	a.high.low = std::bitset<64>(std::string("1111111111111111111111111111111111111111111111111111111111111111")).to_ullong();
	a.low.high = std::bitset<64>(std::string("1111111111111111111111111111111111111111111111111111111111111111")).to_ullong();
	a.low.low = std::bitset<64>(std::string("1111111111111111111111111111111111111111111111111111111100111100")).to_ullong();

	cudaMalloc(&d_a, 4 * (sizeof(unsigned long long)));
	cudaMemcpy(d_a, &a, 4 * (sizeof(unsigned long long)), cudaMemcpyHostToDevice);

	//cout << "a: " << bitset<64>(a.high.high) << bitset<64>(a.high.low) << bitset<64>(a.low.high) << bitset<64>(a.low.low) << endl;

	cout << "Square-root permutation test ...";
	sqrt_caller << <1, 1 >> > (d_a);
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

	uint256_t piece[128], * d_piece, * d_nonce;
	random_array_256(piece);
	uint256_t nonce = random_256();
	uint256_t farmer_id = random_256();

	cudaMalloc(&d_piece, 4 * sizeof(unsigned long long) * 128); // TODO: pinned memory for parallelized version
	cudaMalloc(&d_nonce, 4 * sizeof(unsigned long long));
	cudaMemcpy(d_piece, piece, 4 * sizeof(unsigned long long) * 128, cudaMemcpyHostToDevice);
	cudaMemcpy(d_nonce, &nonce, 4 * sizeof(unsigned long long), cudaMemcpyHostToDevice);
	
	encode << <1, 1 >> > (d_piece, d_nonce, farmer_id);

	cudaMemcpy(piece, d_piece, 4 * sizeof(unsigned long long) * 128, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	/*for (int i = 0; i < 128; i++)
	{
		cout << bitset<64>(piece[i].high.high) << bitset<64>(piece[i].high.low) << bitset<64>(piece[i].low.high) << bitset<64>(piece[i].low.low) << endl;
	}*/

	return 0;
}