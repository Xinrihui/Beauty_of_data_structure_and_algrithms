test_case = int(input())

for i in range(test_case):
    num_words = int(input())
    words = input().rstrip().split(" ") 
    unique_lens = {}
    result = 0
    for word in words:
        if str(len(word)) not in unique_lens:
            unique_lens[str(len(word))] = [word]
        else:
            unique_lens[str(len(word))].append(word)
    start_char, stop_char, word_length, acoeff, bcoeff, ccoeff, modulo = input().rstrip().split(" ")
    analyze_word = [start_char, stop_char]
    ascii_values = [ord(start_char), ord(stop_char)]
    for j in range(2, int(word_length)):
        ascii_char = (int(acoeff) * ascii_values[-1] + int(bcoeff) * ascii_values[-2] + int(ccoeff)) % int(modulo)
        analyze_word.append(chr(97 + (ascii_char % 26)))
        ascii_values.append(ascii_char)
    for length, word in unique_lens.items():
        length = int(length)
        indices = []
        if (length < len(analyze_word)):
            #Each word that has a length of length
            substring_freq = [0] * 26
            freq_char = []
            indices = []
            #Calculate character frequency for substring
            for index in range (length):
                substring_freq[ord(analyze_word[index]) - 97] += 1
            #Calculate character frequency for words with length, length
            for index in range (len(word)):
                freq_char.append([0]*26)
                for word_char in word[index]:
                    freq_char[index][ord(word_char) - 97] += 1
            #Check the starting substring if they match with the available words
            for index in range(len(word)):
                if word[index][0] == analyze_word[0] and word[index][-1] == analyze_word[length - 1] and freq_char[index] == substring_freq:
                    result += 1
                    indices.append(index)
            while len(indices) > 0:
                remove_index = indices.pop()
                del freq_char[remove_index]
                del word[remove_index]
            #Loop through all substrings to see if there's a match
            for index in range(length, len(analyze_word)):
                substring_freq[ord(analyze_word[index]) - 97] += 1
                substring_freq[ord(analyze_word[index - length]) - 97] -= 1
                for found_index in range(len(word)):
                    if analyze_word[index - length + 1] == word[found_index][0] and analyze_word[index] == word[found_index][-1] and freq_char[found_index] == substring_freq:
                        result += 1
                        indices.append(found_index)

                while len(indices) > 0:
                    remove_index = indices.pop()
                    del freq_char[remove_index]
                    del word[remove_index]
                #Break if found all words
                if (result == num_words):
                    break
    print (f'Case #{i + 1}: {result}')
