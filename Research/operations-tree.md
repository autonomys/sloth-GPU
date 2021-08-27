# Required Operations (with addition chain and fast reduction)

## Required Operations Tree (common leafs may be present)
- square-root permutation
    - operator `>` (lhs: 256-bit, rhs: 256-bit)
    - operator `==` (lhs: 256-bit, rhs: 256-bit)
    - operator `-` (lhs: 256-bit, rhs: 256-bit)
    - isEven(256-bit)
    - isOdd(256-bit)
    - multiplication with reduction
        - operator `x` (lhs: 256-bit, rhs: 256-bit)
        - operator `+` (lhs: 256-bit, rhs: 256-bit)
        - operator `+` (lhs: 256-bit, rhs: 8-bit)
        - operator `+` (lhs: 256-bit, rhs: 1-bit)
    - operator `|` (lhs: 256-bit, rhs: 256-bit)
