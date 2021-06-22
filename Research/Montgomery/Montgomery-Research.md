# Research on Montgomery

## Versions of Montgomery
There are multiple versions of Montgomery applications, namely:
1. Montgomery for modulo: A mod N
2. Montgomery for modular multiplication: AxB mod N
3. Montgomery for modular exponentiation: A^B mod N

In sloth, we are aiming to replace modular exponentiation with its faster alternative for processing units, Montgomery(3rd version).


## Sub-versions of Modular Exponentiationt Montgomery
There exist, again, multiple versions of this algorithm. 

*Our criteria:*
1. Evade divisions and modulo operations as much as possible
2. Evade loops as much as possible

The version of Montgomery present in this folder is the version satisfies our criteria the most.
