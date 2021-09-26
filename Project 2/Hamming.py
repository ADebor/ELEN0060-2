import numpy as np

def Hamming_7_4_cod(stream):
    """
    Encodes a given sequence of binary symbols using the Hamming (7,4) code
    ---
    Input:
    - stream (list) : sequence of binary symbols to be encoded
    ---
    Output:
    - stream (list) : encoded sequence
    """

    if (len(stream) % 4) != 0:
        exit("Aborted coding: non valid number of bits")

    G = np.array([[1, 0, 0, 0, 1, 0, 1], [0, 1, 0, 0, 1, 1, 0], [0, 0, 1, 0, 1, 1, 1], [0, 0, 0, 1, 0, 1, 1]])
    stream = [stream[k:k+4] for k in range(0, len(stream), 4)]
    for i, word in enumerate(stream):
        word = [int(i) for i in word]
        stream[i] = ''.join(map(str, np.asarray(word).dot(G) % 2))

    return stream

def Hamming_7_4_dec(stream):
    """
    Decodes a given sequence encoded using the Hamming (7,4) code
    ---
    Input:
    - stream (list) : encoded stream
    ---
    Output:
    - stream (list) : decoded sequence
    """

    if (len(stream) % 7) != 0:
        exit("Aborted decoding: non valid number of bits")

    synd_tab = {"000" : np.array([0, 0, 0, 0]),
        "001" : np.array([0, 0, 0, 0]),
        "010" : np.array([0, 0, 0, 0]),
        "100" : np.array([0, 0, 0, 0]),
        "101" : np.array([1, 0, 0, 0]),
        "110" : np.array([0, 1, 0, 0]),
        "111" : np.array([0, 0, 1, 0]),
        "011" : np.array([0, 0, 0, 1])
       }

    G = np.array([[1, 0, 0, 0, 1, 0, 1], [0, 1, 0, 0, 1, 1, 0], [0, 0, 1, 0, 1, 1, 1], [0, 0, 0, 1, 0, 1, 1]])
    stream = [stream[k:k+7] for k in range(0, len(stream), 7)]
    for i, word in enumerate(stream):
        sig = word[0:4]
        par = word[4:]

        sig = np.asarray([int(i) for i in sig])
        par = np.asarray([int(i) for i in par])

        code_word = sig.dot(G) % 2
        par_word = code_word[4:]

        syndrom =  (par + par_word) % 2
        synd = ''
        for bit in syndrom:
            synd += str(bit)

        mpep = synd_tab[synd]
        stream[i] = (sig + mpep) % 2

    return stream
