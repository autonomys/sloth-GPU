// C++ IMPORTS 
#include <iostream>
#include <bitset>

// CUSTOM IMPORTS
#include "encode_library.h"

using namespace std;

__global__ void montgomery_test(uint256_t* a, uint256_t* m)
{
	uint256_t aa = *a, expoo;
	expoo.low.low = 2;

	uint256_t montgo_res = montgomery_exponentiation(aa, expoo);

	*m = montgo_res;
}

__global__ void weird_test(uint256_t* a, uint256_t* w)
{
	uint256_t aa = *a;

	uint256_t weird_res = weird_reduction(aa, aa);

	*w = weird_res;
}

int main()
{
	uint256_t a, m, w, * d_a, * d_m, * d_w;

	a.high.high = std::bitset<64>(std::string("1111111111111111111111111111111111111111111111111111111111111111")).to_ullong();
	a.high.low = std::bitset<64>(std::string("1111111111111111111111111111111111111111111111111111111111111111")).to_ullong();
	a.low.high = std::bitset<64>(std::string("1111111111111111111111111111111111111111111111111111111111111111")).to_ullong();
	a.low.low = std::bitset<64>(std::string("1111111111111111111111111111111111111111111111111111111100111100")).to_ullong();

	cudaMalloc(&d_a, 4 * (sizeof(unsigned long long)));
	cudaMalloc(&d_m, 4 * (sizeof(unsigned long long)));
	cudaMalloc(&d_w, 4 * (sizeof(unsigned long long)));

	cudaMemcpy(d_a, &a, 4 * (sizeof(unsigned long long)), cudaMemcpyHostToDevice);

	//cout << "a: " << bitset<64>(a.high.high) << bitset<64>(a.high.low) << bitset<64>(a.low.high) << bitset<64>(a.low.low) << endl;
	//cout << "p: " << bitset<64>(p.high.high) << bitset<64>(p.high.low) << bitset<64>(p.low.high) << bitset<64>(p.low.low) << endl;

	montgomery_test << <1, 1 >> > (d_a, d_m);
	weird_test << <1, 1 >> > (d_a, d_w);


	cudaMemcpy(&m, d_m, 4 * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
	cudaMemcpy(&w, d_w, 4 * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	//cout << bitset<64>(m.high.high) << bitset<64>(m.high.low) << bitset<64>(m.low.high) << bitset<64>(m.low.low) << endl;
	//cout << bitset<64>(w.high.high) << bitset<64>(w.high.low) << bitset<64>(w.low.high) << bitset<64>(w.low.low) << endl;

	return 0;
}