from Huffman import huffman_procedure
from LZ77Huffman import LZ77Huffman
from Lempel_Ziv import online_LZ, LZ77
import math
import utils
import csv
from collections import defaultdict
from utils import load_text_sample
import matplotlib.pyplot as plt
import numpy as np

def Question7(dictionnary,genome_length):

    nb_of_codons = genome_length/3
    codon_marginal_pb = []
    codon_list = []
    for key, value in dictionnary.items():  
        codon_marginal_pb.append(value/nb_of_codons)
        codon_list.append(key)
    genome_code_out = huffman_procedure(codon_marginal_pb)
    encoded_genome_length = 0
    for i, pi in enumerate(genome_code_out):
        li = len(genome_code_out[i])
        encoded_genome_length += li*codon_dic[codon_list[i]]

    return encoded_genome_length/nb_of_codons

def entropy(marginal_p_distrib):
    
    """
    Computes H(X) from P_X
    """
    H = 0
    for p in marginal_p_distrib:
        if p != 0:
            H -= p*math.log(p, 2)
    return H

# Convert the dictionnary from the codon alphabet to the amino acid alphabet in order to compute the codon marginal probabilities
def convert(bases):
    codons = defaultdict(int)
    codons['Ala'] = bases['GCT'] + bases['GCC'] + bases['GCA'] + bases['GCG']
    codons['Arg'] = bases['CGT'] + bases['CGC'] + bases['CGA'] + bases['CGG'] + bases['AGA'] + bases['AGG']
    codons['Asn'] = bases['AAT'] + bases['AAC']
    codons['Asp'] = bases['GAT'] + bases['GAC']
    codons['Cys'] = bases['TGT'] + bases['TGC']
    codons['Gln'] = bases['CAA'] + bases['CAG']
    codons['Glu'] = bases['GAA'] + bases['GAG']
    codons['Gly'] = bases['GGT'] + bases['GGC'] + bases['GGA'] + bases['GGG']
    codons['His'] = bases['CAT'] + bases['CAC']
    codons['Ile'] = bases['ATT'] + bases['ATC'] + bases['ATA']
    codons['Leu'] = bases['CTT'] + bases['CTC'] + bases['CTA'] + bases['CTG'] + bases['TTA'] + bases['TTG']
    codons['Lys'] = bases['AAA'] + bases['AAG']
    codons['Met'] = bases['ATG']
    codons['Phe'] = bases['TTT'] + bases['TTC']
    codons['Pro'] = bases['CCT'] + bases['CCC'] + bases['CCA'] + bases['CCG']
    codons['Ser'] = bases['TCT'] + bases['TCC'] + bases['TCA'] + bases['TCG'] + bases['AGT'] + bases['AGC']
    codons['Thr'] = bases['ACT'] + bases['ACC'] + bases['ACA'] + bases['ACG']
    codons['Trp'] = bases['TGG']
    codons['Tyr'] = bases['TAT'] + bases['TAC']
    codons['Val'] = bases['GTT'] + bases['GTC'] + bases['GTA'] + bases['GTG']
    codons['Stop'] = bases['TAA'] + bases['TGA'] + bases['TAG']
    return codons

if __name__ == "__main__":
    # print facilities
    star = 14 * '*'

    # -- QUESTION 1 --
    print("\n" + star)
    print("* QUESTION 1 *")
    print(star)
    print("\nVerifying on ex 7 of the 2nd list of exercises...")
    p_in = [0.05, 0.10, 0.15, 0.15, 0.2, 0.35]
    code_out = huffman_procedure(p_in)
    print("Results:\n")
    print("{:<10}{:^7}{:>12}".format("probability", "|", "code word"))
    print("-"*30)
    for i, p in enumerate(p_in):
        print("{:<10}{:^10}{:>10}".format(p, '|' ,code_out[i]))

    # -- QUESTION 2 --
    print("\n" + star)
    print("* QUESTION 2 *")
    print(star)
    print("\nVerifying on example of slide 50/53...")

    T = '1011010100010'

    #T = utils.load_text_sample() #uncomment to deal with the genome sequence

    dict, stream = online_LZ(T)

    cnter = 0
    for elem in stream:
        if elem != '1' and elem != '0':
            cnter += 1

    CR = len(T)/(len(stream)+cnter) * math.log2(4)/math.log2(2)

    print("Results:\n")
    print("Dictionary: \n", dict)
    print("\nBit stream: \n", stream)

    print("\nCompression rate : ", CR)

    # -- QUESTION 4 --
    print("\n" + star)
    print("* QUESTION 4 *")
    print(star)
    print("\nReproducing the example of Figure 2 with l=7...")
    seq = 'abracadabrad'
    #seq = 'cocorico' #additional test

    l = 7
    output = LZ77(seq, l)
    print("Results:\n")
    print("output")
    print("------")
    for element in output:
        print(element)

    # --- Question 5 ---

    # Read genome data from the file given by TA
    genome_text = load_text_sample()
    genome_length = len(genome_text)
    nb_of_codons = genome_length/3
    codon_dic = defaultdict(int)
    i = 0
    codon = ""

    # Split the line 3 characters at once to highlight the base triplet (codons)
    for base in genome_text:
        i += 1
        codon += base
        if(i%3==0):
            codon_dic[codon] +=1
            codon = ""

    codon_marginal_pb = []
    codon_list = []
    for key, value in codon_dic.items():    
        codon_marginal_pb.append(value/nb_of_codons)
        codon_list.append(key)

    # Call the function to create a binary huffman code
    genome_code_out = huffman_procedure(codon_marginal_pb)

    #Compute the values asked for the project
    encoded_genome_length = 0
    expected_average_length = 0
    empirical_average_length = 0
    codeword_len = []

    print("\n" + star)
    print("* QUESTION 5 *")
    print(star)

    for i, pi in enumerate(codon_marginal_pb):
        li = len(genome_code_out[i])
        codeword_len.append(li)
        encoded_genome_length += li*codon_dic[codon_list[i]]
        expected_average_length += li*pi

    EncodedCondonData = open("EDC.txt", "w")
    for index in range(len(codon_list)):
        EncodedCondonData.write(str(codon_list[index]) + " " + str(codon_marginal_pb[index]) + " " + str(genome_code_out[index]) + " " + str(codeword_len[index]) + "\n")
    EncodedCondonData.close()

    # Display the values asked for this project
    print("Genome length = ",genome_length,"\n")
    print("Number of codons = ",nb_of_codons,"\n")
    print("Encoded genome length = ",encoded_genome_length,"\n")
    compression_rate = genome_length*math.log(4,2)/encoded_genome_length
    print("Compression rate = ",compression_rate,"\n")

    # --- QUESTION 6 ---
    print("\n" + star)
    print("* QUESTION 6 *")
    print(star)
    empirical_average_length = encoded_genome_length/nb_of_codons
    print("Expected average length = ",expected_average_length,"\n")
    print("Empirical average length = ",empirical_average_length,"\n")
    source_entropy = entropy(codon_marginal_pb)
    print("Theoritical bounds : [ ",source_entropy," ; ",source_entropy+1," ]\n")

    # --- QUESTION 7 ---
    List = list(range(1, 3196))
    # We build a list of increasing input genome length used to build our huffman code
    index_list = [(element * 300) + 57 for element in List] # Choose a number splittable by 3 in order to get as close as possible to the genome length
    i = 0
    codon_dic = defaultdict(int)
    empirical_average_lens = []
    codon = ''
    for base in genome_text:
        i += 1
        codon += base
        if(i%3==0):
            codon_dic[codon] += 1
            codon = ''
        if i in index_list: 
            empirical_average_lens.append(Question7(codon_dic,i))
    
    empirical_average_len_file = open("EAL.txt","w")
    for index in range(len(empirical_average_lens)):
        empirical_average_len_file.write(str(index_list[index]) + " " + str(empirical_average_lens[index]) + "\n")
    empirical_average_len_file.close()
    M = np.loadtxt('EAL.txt')
    plt.plot(M[:,0], M[:,1])
    plt.xlabel('Genome length in number of bases')
    plt.ylabel('Empirical average length')
    plt.savefig("EAL.png")

    # --- QUESTION 8 ---
    aa_dic = convert(codon_dic)
    aa_values = aa_dic.values()
    nb_of_aa = sum(aa_values)
    aa_marginal_pb = []
    aa_list = []
    for key, value in aa_dic.items():    
        aa_marginal_pb.append(value/nb_of_aa)
        aa_list.append(key)

    # Call the function to create a binary huffman code
    genome_code_out = huffman_procedure(aa_marginal_pb)

    #Compute the values asked for the project
    encoded_genome_length = 0
    expected_average_length = 0
    empirical_average_length = 0
    codeword_len = []

    print("\n" + star)
    print("* QUESTION 8 *")
    print(star)

    for i, pi in enumerate(aa_marginal_pb):
        li = len(genome_code_out[i])
        codeword_len.append(li)
        encoded_genome_length += li*aa_dic[aa_list[i]]
        expected_average_length += li*pi

    EncodedAaData = open("EAD.txt", "w")
    for index in range(len(aa_list)):
        EncodedAaData.write(str(aa_list[index]) + " " + str(aa_marginal_pb[index]) + " " + str(genome_code_out[index]) + " " + str(codeword_len[index]) + "\n")
    EncodedAaData.close()

    # Display the values asked for this project
    print("Genome length = ",genome_length,"\n")
    print("Number of AA = ",nb_of_aa,"\n")
    print("Encoded genome length = ",encoded_genome_length,"\n")
    compression_rate = (genome_length/encoded_genome_length)*math.log2(4)
    print("Compression rate = ",compression_rate,"\n")

    # -- QUESTION 9 --
    print("\n" + star)
    print("* QUESTION 9 *")
    print(star)

    dict, stream = online_LZ(genome_text)

    cnter = 0
    for elem in stream:
        if elem != '1' and elem != '0':
            cnter += 1

    CR = len(genome_text)/(len(stream)+cnter) * math.log2(4)/math.log2(2)

    print("Results:")
    print("\nLength of the encoded genome sequence = ", len(stream), " (non binarized letters) or ", len(stream)+cnter, " (binarized letters)")
    print("\nCompression rate = ", CR)

    # -- QUESTION 10 --
    print("\n" + star)
    print("* QUESTION 10 *")
    print(star)

    window = 22
    stream = LZ77(genome_text,window)
    nb_symbols = 3*len(stream)
    nb_commas = nb_symbols - 1
    encoded_genome_len = nb_symbols + nb_commas
    print(" Genome length = ",genome_length)
    print("length_encoded_lz77_genome = ", encoded_genome_len)
    compression_rate = (genome_length/encoded_genome_len)*(math.log2(4)/math.log2(26))
    print("Compression rate = ",compression_rate,"\n")


    # -- QUESTION 12 --
    print("\n" + star)
    print("* QUESTION 12 *")
    print(star)
    encoded_genome = LZ77Huffman(genome_text,7)
    print("\nResults:")
    print("\nGenome length = ", len(genome_text), "(alphabet size: 4)")
    print("\nEncoded genome length = ", len(encoded_genome), "(alphabet size: 2)")
    CR = len(genome_text)/len(encoded_genome) * math.log(4, 2)/math.log(2, 2)
    print('\nCompression rate = ', CR)

    # -- QUESTION 13 --
    print("\n" + star)
    print("* QUESTION 13 *")
    print(star)
    window_index = list(range(7,8))  # Choose a range of window size that you want to use !
    window = 0
    Q13Data = open("Q13.txt", "w")
    Q13Data.write("window size" + " " + "LZ77 encoded length" + " " + "LZ77 compression rate " + " " + "LZ77 Huffman encoded length" + " "+ "LZ77Huffman compression rate" + "\n")
    for elem in window_index:
        window = int(math.pow(2,elem))
        Lz77_stream = LZ77(genome_text,window)
        nb_tuples = len(Lz77_stream)
        Lz77_stream_len = nb_tuples*(2*elem + 2)
        Lz77_compression_rate = (genome_length/Lz77_stream_len)*math.log2(4)
        Lz77Huffman_stream = LZ77Huffman(genome_text,window)
        Lz77Huffman_stream_len = len(Lz77Huffman_stream)
        Lz77Huffman_compression_rate = (genome_length/Lz77Huffman_stream_len)*math.log2(4)
        Q13Data.write(str(window) + " " + str(Lz77_stream_len) + " " + str(Lz77_compression_rate) + " " + str(Lz77Huffman_stream_len) + " " +str(Lz77Huffman_compression_rate) + "\n")
    Q13Data.close()