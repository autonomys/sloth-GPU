from math import log2


def DecimalToBinary(n):  # returns a string of bit representation of the num
    return "{0:b}".format(n)  

def BinaryToDecimal(n):
    return int(n,2)

def chunks_64(op):
    op_array = []
    for iter in range(len(op)//64):
        op_array.append(op[iter*64: (iter+1)*64])
    return op_array

# put operand 1 as bit
#operand1_decimal = 1289312831239555555
#op1 = DecimalToBinary(operand1_decimal)
op1 = "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111100111100"
operand1_decimal = BinaryToDecimal(op1)

# put operand2 as bit
#operand2_decimal = 1290390120391092390
#op2 = DecimalToBinary(operand2_decimal)
op2 = "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111101000011"
operand2_decimal = BinaryToDecimal(op2)

# put the result to be tested as bits
to_be_tested = "0101001110000101000111011100000110110011100000101100011000001000001000001011100001001100111010100111001111101111100100010011110000010100001110111111110010010101000111100011100110100011101011011101011000011000111100111100100011101011110011110101100110111000"

# put operand 1's boundary bit length in here
op1_len = 256

# put operand 2's boundary bit length in here
op2_len = 256

# required for multiplication: determine the bit-lenght of the output
result_len = 256

# DEFINE THE OEPRATION HERE
p = 115792089237316195423570985008687907853269984665640564039457584007913129639747
if operand2_decimal != p:
    assert("prime is different")
#excepted_result_decimal = pow(operand1_decimal, operand2_decimal, p)
temp = BinaryToDecimal(to_be_tested)
expected_result_decimal = pow(temp, 2, p)






# DO NOT TOUCH THE REST FROM HERE
temp = op1_len - len(op1)
op1 = "0" * temp + op1

temp = op2_len - len(op2)
op2 = "0" * temp + op2

op1_array = chunks_64(op1)
op2_array = chunks_64(op2)
result_array = chunks_64(to_be_tested)

print("Operand 1:")
print(op1_array)
print("-------")

print("Operand 2:")
print(op2_array)
print("-------\n")

print("Result:")
print(result_array)
print("-------\n")


if (BinaryToDecimal(op1) == expected_result_decimal):
    print("SUCCESS!")
else:
    print("FAIL!")
    print(log2(expected_result_decimal))
    temp = DecimalToBinary(expected_result_decimal)
    count = 0
    '''
    for x in range(len(to_be_tested)):
        if to_be_tested[x] != temp[x]:
            print(str(count) + "th bit is different")
        count += 1
    '''








