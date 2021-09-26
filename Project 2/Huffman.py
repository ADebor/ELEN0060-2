class Node(object):
    """
    Node class used to build a Huffman tree.
    ---
    Attributes:
    - left and right : left and right children of a node, None if leaf
    - value : probability value associated to the node
    - word : index of the leaf if the node is a leaf, None otherwise
    ---
    Methods:
    - get methods for each of the attributes
    """

    def __init__(self, left=None, right=None, value=None, word=None):
        self.left = left
        self.right = right
        self.value = value
        self.word = word

    def get_children(self):
        return (self.left, self.right)

    def get_value(self):
        return self.value

    def get_word(self):
        return self.word

def huffman_code(node, code = ''):
    """
    Build a Huffman code given a root node
    ---
    Input:
    - node (Node object) : root node
    - code (string) : already generated binary code word
    ---
    Output:
    - tree (dictionary) : dictionary of the Huffman code words, with {key : value} = {index of a leaf : code word}
    """
    
    if node.get_word() != None:
        # Leaf reached
        return {node.get_word() : code}
    child1, child2 = node.get_children()
    tree = {}
    tree.update(huffman_code(child1, code + '1'))
    tree.update(huffman_code(child2, code + '0'))
    return tree

def huffman_procedure(p_dist):
    """
    Generates a binary Huffman code for a given probability distribution
    ---
    Input:
    - p_dist (list) : probability distribution
    ---
    Output:
    - ret (list) : binary Huffman code corresponding to the given probability distribution
    """

    nodes = [Node(value=p, word=str(i)) for i, p in enumerate(p_dist)]
    nodes = sorted(nodes, key=lambda node: node.get_value())

    while len(nodes) > 1:

        p1 = nodes[0].get_value()
        p2 = nodes[1].get_value()
        new_node = Node(left=nodes[1], right=nodes[0], value=p1+p2)

        nodes = nodes[2:]
        nodes.append(new_node)

        nodes = sorted(nodes, key=lambda node: node.get_value())

    code = huffman_code(nodes[0])

    ret = []
    for i in range(len(p_dist)):
        ret.append(code[str(i)])

    return ret
