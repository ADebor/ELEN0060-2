from utils import binarize
from math import log, ceil

def online_LZ(seq):
    """
    Implements the on-line Lempel-Ziv compression method
    ---
    Input :
    - seq (string) : sequence of symbols to encode
    ---
    Output :
    - dict (dictionary) : dictionary with {key : value} := {source word : (address, bit)}
    - stream (string) : encoded sequence
    """

    # Initialization of the dictionary
    dict = {}
    dict[''] = ('', '')

    # String sequence casted as a list of binary elements
    seq = list(seq)

    prefix = ''

    # Encoding
    for symbol in seq:
        if prefix + symbol in dict:
            prefix += symbol
        else:
            size = ceil(log(len(dict), 2))
            address = binarize(list(dict).index(prefix), size)
            if size == 0:
                address = ''
            dict[prefix + symbol] = (address, symbol)
            prefix = ''

    # Generating the binary stream
    val = list(dict.values())
    val = [i+j for i,j in val]
    stream = ''.join(val)

    return dict, stream

def LZ77(seq, l):
    """
    Implements the Lempel-Ziv 77 compression method
    ---
    Input :
    - seq (string) : sequence of symbols to encode
    - l (int) : size of the sliding window
    ---
    Output :
    - encoded_input (list) : encoded sequence
    """

    buffer = list(seq)
    window = l*['']
    encoded_input = []
    while buffer :
        distance, length, char = get_prefix(buffer[:l+1], window)
        encoded_input.append((distance, length, char))

        del window[0:length+1]
        window.extend(buffer[0:length+1])
        del buffer[0:length+1]

    return encoded_input

def get_prefix(buffer, window):
    """
    Finds the longest prefix of input that begins in window, in the frame of the Lempel-Ziv compression algorithm
    ---
    Input :
    - buffer (list) : search buffer
    - window (list) : sliding window
    ---
    Output :
    - distance (int) : distance of the longest prefix in the search buffer
    - length (int) : length of the longest prefix found
    - next_char (string) : symbol following the prefix, '' if no symbol after the prefix.
    """

    i = 0
    j = 0
    match = []
    best_match = []

    buff_flag = True
    win_flag = True
    match_flag = False

    distance = 0
    length = 0
    next_char = buffer[0]

    while buff_flag:

        symbol1 = buffer[i]
        win_flag = True

        while win_flag:

            symbol2 = window[j]

            if symbol1 == symbol2: # match found
                match_flag = True
                match.append(symbol2)
                win_flag = not win_flag

                if i == len(buffer) - 1:
                    # match until the end of the buffer
                    match_flag = False
                    if len(match) >= len(best_match):
                        best_match = match.copy()
                        match = []
                        distance = (len(best_match) - 1 ) + (len(window) - j)
                        length = len(best_match)
                        next_char = ''
                    j -= 1


                if j == len(window) - 1:
                    # match until the end of the window
                    match_flag = False
                    if len(match) >= len(best_match):
                        best_match = match.copy()
                        match = []
                        distance = len(best_match) + (len(window) - j - 1)
                        length = len(best_match)
                        next_char = '' if i == len(buffer) - 1 else buffer[i+1]

            elif match_flag == True:
                # match ended
                match_flag = False
                if len(match) >= len(best_match):
                    best_match = match.copy()
                    match = []
                    distance = len(best_match) + (len(window) - j)
                    length = len(best_match)
                    next_char = buffer[i]

                win_flag = False
                i = -1
                j -= 1

            j += 1
            if j == len(window):
                win_flag = False
                buff_flag = False

        i = (i + 1) % len(buffer)

    return distance, length, next_char
