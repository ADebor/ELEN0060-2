import utils
import Huffman
import Lempel_Ziv
import numpy as np
import math
import matplotlib.pyplot as plt

def LZ77Huffman(seq,l):
    """
    Algorithm combining LZ77 and Huffman.
    ---
    Input:
    - seq : sequence to encode
    - l : length of the window size for LZ77
    ---
    Output:
    - Huffgen : encoded sequence
    """

    print("First encoding: LZ77...")
    LZ77gen = Lempel_Ziv.LZ77(seq, l)

    print("Second encoding: Huffman...")
    val, counts = np.unique(LZ77gen, return_counts=True, axis=0)
    tuples = []
    for i in val:
        tuples.append(tuple(i))

    prob = counts / sum(counts)
    code = Huffman.huffman_procedure(prob)

    print("Compressing...")
    zip_iterator = zip(tuples, code)
    dictionary = dict(zip_iterator)

    Huffgen =''
    for element in LZ77gen:
        element = tuple(map(str, element))
        Huffgen += dictionary[element]

    """
    alphab = tuples
    pos = np.arange(len(alphab))
    width = 0.5     # gives histogram aspect to the bar diagram

    ax = plt.axes()
    plt.xlabel("Tuples")
    plt.ylabel("Proportion")

    plt.bar(pos, prob, width, color='blue')
    plt.savefig("LZ77Huffman.pdf")
    plt.show()
    """
    
    return Huffgen
