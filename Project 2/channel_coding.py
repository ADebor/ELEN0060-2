from utils import load_wav, binarize, bin_to_dec, save_wav
from BSC import BSC
from Hamming import Hamming_7_4_cod, Hamming_7_4_dec

import matplotlib.pyplot as plt
import numpy as np
from playsound import playsound

if __name__ == "__main__":

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')


    # -- QUESTION 15 --

    # Playing the sound
    print("\nQUESTION 15")
    print("\nPlaying sound...")
    #playsound('sound.wav')
    print("Success.")

    # Generating the plot
    print("\nGenerating the plot...")
    rate, data = load_wav()
    nb_samples = len(data)
    fig = plt.figure()
    plt.plot(data)
    plt.xlabel(r'Time (s)',fontsize=16)
    plt.ylabel(r'(Quantized) amplitude',fontsize=16)
    plt.savefig("Q15_sound_signal.pdf")
    print("Success.")

    # -- QUESTION 16 --

    print("\nQUESTION 16")
    nb_bits = 8

    # Encoding
    print("\nEncoding the signal...")
    encoded_sound = []
    for sample in data:
        encoded_sound.append(binarize(sample, nb=nb_bits))

    # Concatenation
    stream = ''
    for word in encoded_sound:
        stream += str(word)
    stream = list(stream)
    print("Success.")

    # -- QUESTION 17 --

    print("\nQUESTION 17")

    # BSC simulation
    print("\nSimulating the channel effect...")
    p = 0.01
    bsc_stream = BSC(stream, p)
    print("Success.")

    # Decoding
    print("\nDecoding the signal...")
    decoded_data = np.empty(nb_samples, dtype=np.uint8)
    bsc_stream = [bsc_stream[k:k+nb_bits] for k in range(0, len(bsc_stream), nb_bits)]

    for i, word in enumerate(bsc_stream):
        if len(word) != nb_bits:
            exit("Aborted : corrupted signal")
        else:
            decoded_word = bin_to_dec(''.join(word))
            decoded_data[i] = decoded_word
    print("Success.")

    # Generating the plot
    print("\nGenerating the plot...")
    plt.figure()
    plt.plot(decoded_data)
    plt.xlabel(r'Time (s)',fontsize=16)
    plt.ylabel(r'(Quantized) amplitude',fontsize=16)
    plt.savefig("Q17_decoded_sound_signal.pdf")
    print("Success.")

    # Playing the sound
    print("\nSaving .wav file...")
    save_wav("sound_BSC.wav", rate, decoded_data)
    print("Success.")
    print("\nPlaying sound...")
    #playsound('sound_BSC.wav')
    print("Success.")

    # -- QUESTION 18 --

    print("\nQUESTION 18")

    # Encoding
    print("\nEncoding the signal with Hamming (7,4) ...")
    stream_Hamming = Hamming_7_4_cod(stream)

    # Concatenation
    stream_Hamming_str = ''
    for word in stream_Hamming:
        stream_Hamming_str += str(word)
    stream_Hamming = list(stream_Hamming_str)
    print("Success.")

    # -- QUESTION 19 --

    # BSC simulation
    print("\nSimulating the channel effect on the redundant signal...")
    p = 0.01
    bsc_stream_Hamming = BSC(stream_Hamming, p)
    print("Success.")

    # Decoding
    print("\nDecoding the signal...")
    decoded_data_Hamming = Hamming_7_4_dec(bsc_stream_Hamming)

    # Concatenation
    decoded_data_Hamming_str = ''
    for word_Hamming in decoded_data_Hamming:
        decoded_data_Hamming_str += ''.join(map(str, word_Hamming))
    decoded_data_Hamming_str = list(decoded_data_Hamming_str)
    print("Success.")

    # Decoding
    print("\nDecoding the signal...")
    decoded_data_Hamming_arr = np.empty(nb_samples, dtype=np.uint8)
    decoded_data_Hamming_str = [decoded_data_Hamming_str[k:k+nb_bits] for k in range(0, len(decoded_data_Hamming_str), nb_bits)]
    for i, word in enumerate(decoded_data_Hamming_str):
        if len(word) != nb_bits:
            exit("Aborted : corrupted signal")
        else:
            decoded_word_Hamming = bin_to_dec(''.join(word))
            decoded_data_Hamming_arr[i] = decoded_word_Hamming
    print("Success.")

    # Generating the plot
    print("\nGenerating the plot...")
    fig3 = plt.figure()
    plt.plot(decoded_data_Hamming_arr)
    plt.xlabel(r'Time (s)',fontsize=16)
    plt.ylabel(r'(Quantized) amplitude',fontsize=16)
    plt.savefig("Q19_decoded_sound_signal.pdf")
    print("Success.")

    # Playing the sound
    print("\nSaving .wav file...")
    save_wav("sound_BSC_Hamming.wav", rate, decoded_data_Hamming_arr)
    print("Success.")
    print("\nPlaying sound...")
    #playsound('sound_BSC_Hamming.wav')
    print("Success.")
