def DecimalToBinary(n):  # returns a string of bit representation of the num
    return "{0:b}".format(n)  

def BinaryToDecimal(n):
    return int(n,2)

def chunks_64(op):
    op_array = []
    for iter in range(len(op)//64):
        op_array.append(op[iter*64: (iter+1)*64])
    return op_array

# put operand 1 as decimal
operand1_decimal = 1289312831239555555
op1 = DecimalToBinary(operand1_decimal)

# put operand2 as decimal
operand2_decimal = 1290390120391092390
op2 = DecimalToBinary(operand2_decimal)

# excepted result as decimal
excepted_result_decimal = operand1_decimal * operand2_decimal

# put the result to be tested as bits
to_be_tested = "00000001010000000110101110001110111101110110011011100010100010110011101000110011000110001000011001110100110101000110000100110010"

# put operand 1's boundary bit length in here
op1_len = 64

# put operand 2's boundary bit length in here
op2_len = 64

result_len = op1_len + op2_len


temp = op1_len - len(op1)
op1 = "0" * temp + op1


temp = op2_len - len(op2)
op2 = "0" * temp + op2


op1_array = chunks_64(op1)
op2_array = chunks_64(op2)


print("Operand 1:")
print(op1_array)
print("-------")

print("Operand 2:")
print(op2_array)
print("-------\n")

if (BinaryToDecimal(to_be_tested) == excepted_result_decimal):
    print("SUCCESS!")
else:
    print("FAIL!")









