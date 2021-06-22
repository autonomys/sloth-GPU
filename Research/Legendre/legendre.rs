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
