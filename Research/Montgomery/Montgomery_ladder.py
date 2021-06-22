import random
import math


k=64

def Modular_Multiplication(X,Y,M,mu):
    T = X * Y
    TH = T >> k

    T1 = TH * mu
    T1H = T1 >> k

    T2 = T1H * M
    Cbar = T - T2

    T3 = Cbar - M
    T4 = Cbar - 2*M

    if(T4 >= 0):
        results = T4
    elif(T3 >= 0):
        results = T3
    else:
        results = Cbar

    return results



p= random.randint(2**(k-1), 2**k-1)
mu =((2**(2*k)//p))
A = random.randint(0,p-1)
B = random.randint(0,p-1)

Expected_Result = pow(A,B,p)
"""
Normal MONTGOMERY_LADDER

C0=1
C1=A
for i in range(k-1,-1,-1):
    if(((B>>i)%2)==0):
        C1 = (C0 * C1) % p
        C0 = (C0 * C0) % p
    else:
        C0 = (C0 * C1) % p
        C1 = (C1 * C1) % p

if (Expected_Result == C0):
    print("works")
else:
    print("dayum")
"""


C0=1
C1=A
for i in range(k-1,-1,-1):
    if(((B>>i)%2)==0):
        C1 = Modular_Multiplication(C0,C1,p,mu)
        C0 = Modular_Multiplication(C0,C0,p,mu)

    else:
        C0 = Modular_Multiplication(C0,C1,p,mu)
        C1 = Modular_Multiplication(C1,C1,p,mu)

if (Expected_Result == C0):
    print("works")
else:
    print("dayum")


