# Required Operations

## Required Operations Tree (common leafs may be present)
- square-root permutation
    - operator `>` (lhs: 256-bit, rhs: 256-bit)
        - operator `>` (lhs: 128-bit, rhs: 128-bit)
        - operator `<` (lhs: 128-bit, rhs: 128-bit)
    - legendre(256-bit, 256-bit)
        - Montgomery(256-bit, 255-bit, 256-bit)
            - to be filled
        - operator `-` (lhs: 256-bit, rhs: 64-bit)
            - operator `-` (lhs: 128-bit, rhs: 64-bit)
            - operator `<` (lhs: 128-bit, rhs: 64-bit)
        - operator `>>` (lhs: 256-bit, rhs: 32-bit)
            - operator `>>` (lhs: 128-bit, rhs: 32-bit)
            - operator `<<` (lhs: 128-bit, rhs: 32-bit)
            - operator `|` (lhs: 128-bit, rhs: 128-bit)
        - operator `==` (lhs: 256-bit, rhs: 256-bit)
            - operator `==` (lhs: 128-bit, rhs: 128-bit)
    - operator `==` (lhs: 256-bit, rhs: 64-bit)
        - operator `==` (lhs: 128-bit, rhs: 64-bit)
    - Montomgery(256-bit, 254-bit, 256-bit)
        - to be filled
    - operator `=` (lhs: 256-bit, rhs: 256-bit)
        - operator `=` (lhs: 128-bit, rhs: 128-bit)
    - operator `-` (lhs: 256-bit, rhs: 256-bit)
        - operator `-` (lhs: 128-bit, rhs: 128-bit)
        - operator `<` (lhs: 128-bit, rhs: 128-bit)
        - operator `-` (lhs: 128-bit, rhs: 64-bit)
    - isEven(256-bit)
    - isOdd(256-bit)


## Required Arithmetic Operations Set
*256-bit operations:*
- operator `=` (lhs: 256-bit, rhs: 256-bit)
- operator `>` (lhs: 256-bit, rhs: 256-bit)
- operator `>>` (lhs: 256-bit, rhs: 32-bit)
- operator `==` (lhs: 256-bit, rhs: 256-bit)
- operator `==` (lhs: 256-bit, rhs: 64-bit)
- operator `-` (lhs: 256-bit, rhs: 256-bit)
- operator `-` (lhs: 256-bit, rhs: 64-bit)
- isEven(256-bit)
- isOdd(256-bit)

*128-bit operations:*
- operator `=` (lhs: 128-bit, rhs: 128-bit)
- operator `>` (lhs: 128-bit, rhs: 128-bit)
- operator `<` (lhs: 128-bit, rhs: 128-bit)
- operator `<` (lhs: 128-bit, rhs: 64-bit)
- operator `>>` (lhs: 128-bit, rhs: 32-bit)
- operator `<<` (lhs: 128-bit, rhs: 32-bit)
- operator `==` (lhs: 128-bit, rhs: 128-bit)
- operator `==` (lhs: 128-bit, rhs: 64-bit)
- operator `-` (lhs: 128-bit, rhs: 128-bit)
- operator `-` (lhs: 128-bit, rhs: 64-bit)
- operator `|` (lhs: 128-bit, rhs: 128-bit)




