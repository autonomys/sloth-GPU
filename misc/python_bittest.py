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
op1 = "00100000011110010101100110011010001010110100001001110010010000110000000000000100010011110011001101101000100111010111001111010101"
operand1_decimal = BinaryToDecimal(op1)

# put operand2 as bit
#operand2_decimal = 1290390120391092390
#op2 = DecimalToBinary(operand2_decimal)
op2 = "00110001001101101010001000011100100100011100010110111110001111000000000101100000100011110110100110011111111100111000111101101111"
operand2_decimal = BinaryToDecimal(op2)

# DEFINE THE OEPRATION HERE: excepted result as decimal
excepted_result_decimal = operand1_decimal * operand2_decimal

# put the result to be tested as bits
to_be_tested = "0000011000111110001010000100111111010011010001000000010101100011101001111111011111001011100010111111111001111111111110011110010100110001001000110111101000111010011010111011100011110001010010001010011100001010101001000111110110001110001010000011010001011011"

# put operand 1's boundary bit length in here
op1_len = 128

# put operand 2's boundary bit length in here
op2_len = 128

# for multiplication: determine the bit-lenght of the output
result_len = op1_len + op2_len


# DO NOT TOUCH THE REST FROM HERE
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