import random

#######################
## FUNCTION DEFINITIONS START
def jacobi(a, n):
    if n <= 0:
        raise ValueError("'n' must be a positive integer.")
    if n % 2 == 0:
        raise ValueError("'n' must be odd.")
    a %= n
    result = 1
    while a != 0:
        while a % 2 == 0:
            a /= 2
            n_mod_8 = n % 8
            if n_mod_8 in (3, 5):
                result = -result
        a, n = n, a
        if a % 4 == 3 and n % 4 == 3:
            result = -result
        a %= n
    if n == 1:
        return result
    else:
        return 0


def legendre(a, p):
    ls = pow(a, (p-1)//2, p)

    if ls == (p - 1):
        return 0
    else:
        return ls


def original_sqrt(data, prime, exponent):
    if data > prime:
        raise ValueError("data must be smaller than prime")

    if jacobi(data, prime) == 1:
        data = pow(data, exponent, prime)
        if data % 2 == 1:
            data = -data
            data += prime
    
    else:
        data = -data
        data += prime
        data = pow(data, exponent, prime)
        if data % 2 == 0:
            data = -data
            data += prime


def optimized_sqrt(data, prime, exponent):
    if data > prime:
        raise ValueError("data must be smaller than prime")

    if legendre(data, prime) == 1:
        data = pow(data, exponent, prime)
        if data % 2 == 1:
            data = prime - data
    
    else:
        data = prime - data
        data = pow(data, exponent, prime)
        if data % 2 == 0:
            data = prime - data


## FUNCTION DEFINITIONS END
#######################



## test
test_amount = 10  # set this as how many random trials wanted to check validity
check = True  # assume tests will pass
for x in range(test_amount): 
    prime = 115792089237316195423570985008687907853269984665640564039457584007913129639747
    data = random.randint(0, prime)
    exponent = (prime+1) // 4

    if optimized_sqrt(data, prime, exponent) != original_sqrt(data, prime, exponent):
        check = False
        break


if check:
    print("Tests PASSED")
else:
    print("Tests FAILED!")


