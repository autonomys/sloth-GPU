import random
from math import log2
global_list = []

def Modular_Multiplication(X,Y,M,mu):
    global global_list

    #if X > 2**256:
    #    print("damn X")
    #if Y > 2**256:
    #    print("damn Y")
    T = X * Y  # 512-bit (256 * 256)
    #if (T > 2**512):
    #    print("damn T")
    TH = T >> k  # 256-bit (512 >> 256)
    global_list.append(TH)
    #if TH > 2**256:
    #    print("damn TH")
    T1 = TH * mu  # 512-bit (256 * 257)
    #if (T1 > 2**512):  
    #    print("damn T1")
    T1H = T1 >> k  # 256-bit (512 >> 256)
    #if T1H > 2**256:
    #    print("damn T1H")
    T2 = T1H * M  # 512-bit (256 * 256)
    #if (T2 > 2**512):
    #    print("damn T2")
    Cbar = T - T2  # 257-bit (512 - 512)
    #Cbar = T - (((T >> k) * mu) >> k) * M
    #if (Cbar > 2 ** 257):
    #    print("damn Cbar")
    T3 = Cbar - M  # 256-bit (257 - 256) 
    #if T3 > (2**257):
    #    print("damn T3")
    T4 = Cbar - 2*M  # 257-bit (257 - 257)
    #if (2*M - Cbar) > 2**257:
    #    print("damn T4")
    if(T4 >= 0):  # (257 comparison)
        results = T4  # (257 =)
    elif(T3 >= 0):  # (256 comparison)
        results = T3  # (256 =)
    else:
        results = Cbar  # (257 =)

    return results


p = 115792089237316195423570985008687907853269984665640564039457584007913129639747
k= int(log2(p))
mu =(2**(2*(k)) // p)
#mu = 2 ** 256
#print(mu)
A = random.randint(0,p-1)
B = random.randint(0,p-1)

Expected_Result = pow(A,B,p)

C0=1
C1=A
for i in range(k-1,-1,-1):
    if(((B>>i)%2)==0):
        res1 = log2(C0 * C1 * mu)
        res2 = log2(C0 * C0 * mu)
        if(512+256 < res1):
            print("res1 overflowing")
        if(512+256 < res2):
            print("res2 overflowing")


        C1 = Modular_Multiplication(C0,C1,p,mu)
        C0 = Modular_Multiplication(C0,C0,p,mu)

    else:
        res1 = log2(C0 * C1 * mu)
        res2 = log2(C1 * C1 * mu)
        if(512+256 < res1):
            print("res1 overflowing")
        if(512+256 < res2):
            print("res2 overflowing")
        C0 = Modular_Multiplication(C0,C1,p,mu)
        C1 = Modular_Multiplication(C1,C1,p,mu)

if (Expected_Result == C0):
    print("SUCCESS")
else:
    print("FAIL")

temp = max(global_list)
#print(temp)
#print(mu)
#print(temp*mu)


