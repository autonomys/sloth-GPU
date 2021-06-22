// template for legendre symbol
// treat this as a pseudo-code

fn legendre_symbol(a: u128, p: u128) -> u128 {
    let ls = mod_exp(a, (p - 1) / 2, p);
    if ls == (p - 1) {
        return 0;
    } else {
        return ls;
    }
}

/* 
Compute the Legendre symbol a|p using Euler's criterion. 
p is a prime, a is relatively prime to p 
(if p divides a, then a|p = 0)

Returns 1 if a has a square root modulo p, 
0 otherwise.
*/
