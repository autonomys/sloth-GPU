// C++ IMPORTS 
#include <iostream>
#include <bitset>

// CUSTOM IMPORTS
#include "uint512.h"

using namespace std;

__global__ void device_mul_257_256(uint512_t *d_mu, uint256_t *d_b, uint512_t *d_res) {
	*d_res = mul257_256(*d_mu, *d_b);
}

int main()
{
    uint512_t mu, *d_mu, res512, *d_res;
    uint256_t b, *d_b;


    //cout << std::bitset<64>(std::string("100")).to_ullong() << endl;

    mu.low.low.low = std::bitset<64>(std::string("0000000000000000000000000000000000000000000000000000000010111101")).to_ullong();
    mu.high.low.low = 1ull;

    b.low.low = std::bitset<64>(std::string("0110100000010110101100100010111110011011111101011000011100111001")).to_ullong(); 
    b.low.high = std::bitset<64>(std::string("1101010001110100000010011110100001101111111111000011001001011010")).to_ullong(); 
    b.high.low = std::bitset<64>(std::string("1010011111000100000010110101110001110111001110111000111110000001")).to_ullong(); 
    b.high.high = std::bitset<64>(std::string("1111111011110000010101100101001101101011010001011010001010110000")).to_ullong();

	cudaMalloc(&d_mu, 8 * sizeof(unsigned long long));
	cudaMalloc(&d_res, 8 * sizeof(unsigned long long));
	cudaMalloc(&d_b, 4 * sizeof(unsigned long long));

	cudaMemcpy(d_mu, &mu, 8 * sizeof(unsigned long long), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, &b, 4 * sizeof(unsigned long long), cudaMemcpyHostToDevice);

	device_mul_257_256 << < 1, 1 >> > (d_mu, d_b, d_res);

	cudaMemcpy(&res512, d_res, 8 * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

	cout << bitset<64>(mu.high.high.high) << bitset<64>(mu.high.high.low) << bitset<64>(mu.high.low.high) << bitset<64>(mu.high.low.low)
		<< bitset<64>(mu.low.high.high) << bitset<64>(mu.low.high.low) << bitset<64>(mu.low.low.high) << bitset<64>(mu.low.low.low) << endl << endl;

	cout << bitset<64>(b.high.high) << bitset<64>(b.high.low) << bitset<64>(b.low.high) << bitset<64>(b.low.low) << endl << endl;

    cout << bitset<64>(res512.high.high.high) << bitset<64>(res512.high.high.low) << bitset<64>(res512.high.low.high) << bitset<64>(res512.high.low.low)
        << bitset<64>(res512.low.high.high) << bitset<64>(res512.low.high.low) << bitset<64>(res512.low.low.high) << bitset<64>(res512.low.low.low) << endl;

    return 0;
}