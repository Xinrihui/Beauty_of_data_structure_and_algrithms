numtest = int(input())

def binary_search_custom(arr, avg):
    left = 0
    right = len(arr) - 1
    mid = 0
    while left < right:
        mid = int((left + right)/2)
        if arr[mid] == avg:
            return mid
        elif avg < arr[mid]:
            if (mid - 1 > 0 and  avg > arr[mid-1] ):
                return mid - 1
            else:
                right = mid - 1
        else:
            if (mid + 1 < len(arr) - 1 and avg < arr[mid + 1]):
                return mid 
            else:
                left = mid + 1

    return mid
for test in range (numtest):
    num_element, k_times = [int(x) for x in input().rstrip().split(" ")]
    values = list(map(lambda x: int(x), input().rstrip().split(" ")))
    values.sort()
    mean = sum(values)/float(len(values))
    expected_value = []
    expected_value.append(mean)
    for _ in range(k_times):
        below_index = binary_search_custom(values, expected_value[-1])
        result = (below_index + 1)  * expected_value[-1]
        for a in range(below_index + 1, len(values)):
            result += values[a]
        result /= float(len(values))
        expected_value.append(result)
    print (f'Case #{test + 1}: {expected_value[-1]:.6f}')