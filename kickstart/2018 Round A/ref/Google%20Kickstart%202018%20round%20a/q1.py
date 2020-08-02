import sys
test_case = int(input())

for x in range(test_case):
    ops = 0
    change_ops = False

    num = input().rstrip()
    for index in range(len(num)):
        if int(num[index]) % 2 != 0:
            change_ops = True
            break
    if change_ops:
        ans = int (num[index:])
        multiplier = 1
        additional = 0
        if len(num) - 1 > index:
            multiplier = 10 ** (len(num) - index - 1)
            additional = int('8' * (len(num) - index - 1))
        lower_bound = (int(num[index]) - 1) * multiplier + additional
        #Calculate upper bound based on whether or not the first odd number is a 9
        if (int(num[index]) == 9 ):
            if (index > 0 and int(num[index - 1]) % 2 != 0):
                upper_bound = int(num[index - 1]) * (multiplier + 1)
            else: 
                upper_bound = lower_bound
        else:
            upper_bound = (int(num[index]) + 1) * multiplier
        ops = min(abs(lower_bound - ans), abs(upper_bound - ans))
    print ("Case #" + str(x + 1) + ": " + str(ops))