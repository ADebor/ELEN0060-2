import random

def BSC(stream, p):
    random.seed(0)

    """
    Simulates a binary symmetric channel with a probability of error p
    ---
    Input :
    - stream (list) : stream of bits at the input of the BSC
    - p (float) : probability of error
    ---
    Output :
    - stream (list) : stream of bits at the output of the BSC
    """
    new_stream = stream.copy()
    for i, bit in enumerate(stream):
        if random.random() <= p:
            new_stream[i] = str(int(not int(bit)))

    return new_stream
