import numpy as np
from collections import Counter
from heapq import heappush, heappop

def read_text_file(file_path):
    """
    Reads the input text file and returns a list of lines.
    
    Each line in the file is preserved as a string.
    """
    lines = []
    with open(file_path, 'r') as file:
        for line in file:
            lines.append(line)
    return lines


def run_text_analysis_tests():
    """
    Reads the input file 'input3.txt' and runs various entropy and Huffman coding analyses
    on each input string.
    """
    file_path = "input3.txt"
    input_strings = read_text_file(file_path)
    
    for text in input_strings:
        print("Input string:", text)
        print("First-order entropy:", calculate_first_order_entropy(text))
        print("Second-order entropy:", calculate_second_order_entropy(text))
        print("Average codeword length:", calculate_average_codeword_length(text))
        print("Joint average codeword length:", calculate_joint_average_codeword_length(text))
        print()


def calculate_first_order_entropy(text):
    """
    Calculates the first-order entropy of the provided text.
    
    Parameters:
        text (str): The input string.
        
    Returns:
        float: The first-order entropy (in bits per symbol).
    """
    entropy = 0
    n = len(text)
    counter = Counter(text)
    
    for symbol in counter.keys():
        prob = counter[symbol] / n
        entropy += (prob * np.log2(prob))
        
    return -entropy


def calculate_second_order_entropy(text):
    """
    Calculates the second-order entropy of the provided text.
    
    Parameters:
        text (str): The input string.
        
    Returns:
        float: The second-order entropy (in bits per symbol).
    """
    entropy = 0
    n = len(text)
    step_size = 2 
    pairs = [text[i:i+step_size] for i in range(0, n, step_size)]
    counter = Counter(pairs)
    
    for symbol in counter.keys(): 
        prob = counter[symbol] / len(pairs)
        entropy += (prob * np.log2(prob))

    return -entropy / 2


class Node:
    """
    A Node in the Huffman coding tree.
    
    Attributes:
        freq (int): Frequency of the symbol(s).
        symbol (str): The symbol or concatenated symbols.
        left (Node): Left child in the tree.
        right (Node): Right child in the tree.
        huff (str): Huffman code digit assigned ('0' or '1').
    """
    def __init__(self, freq, symbol, left=None, right=None): 
        self.freq = freq
        self.symbol = symbol
        self.left = left 
        self.right = right 
        self.huff = ''
    
    def __lt__(self, other): 
        return self.freq < other.freq


def extract_huffman_codes(node, code='', codes_list=[]):
    """
    Recursively extracts Huffman codes from the tree.
    
    Parameters:
        node (Node): The current node in the Huffman tree.
        code (str): The code accumulated so far.
        codes_list (list): List to store the (symbol, code) tuples.
        
    Returns:
        list: A list of tuples where each tuple contains a symbol and its Huffman code.
    """
    new_code = code + str(node.huff)
        
    if node.left is not None: 
        extract_huffman_codes(node.left, new_code, codes_list)
    if node.right is not None: 
        extract_huffman_codes(node.right, new_code, codes_list)
        
    if node.left is None and node.right is None: 
        codes_list.append((node.symbol, new_code))
        
    return codes_list


def generate_huffman_codes(text):
    """
    Generates Huffman codes for the symbols in the text.
    
    Parameters:
        text (str): The input string.
        
    Returns:
        list: A list of tuples containing each symbol and its Huffman code.
    """
    counter = Counter(text)
    nodes = []
    
    for symbol in counter: 
        heappush(nodes, Node(counter[symbol], symbol))
        
    while len(nodes) > 1: 
        left = heappop(nodes)
        right = heappop(nodes)
        
        left.huff = 0 
        right.huff = 1 
        
        new_node = Node(left.freq + right.freq, left.symbol + right.symbol, left, right)
        heappush(nodes, new_node)
        
    codes_list = []
    if nodes: 
        codes_list = extract_huffman_codes(nodes[0])
        
    return codes_list


def calculate_average_codeword_length(text):
    """
    Calculates the average Huffman codeword length for the provided text.
    
    Parameters:
        text (str): The input string.
        
    Returns:
        float: The average codeword length (in bits per symbol).
    """
    avg_length = 0
    n = len(text)
    counter = Counter(text) 
    codes = generate_huffman_codes(text) 
    
    for symbol, code in codes:
        avg_length += ((counter[symbol] / n) * len(code))
    
    return avg_length


def generate_joint_huffman_codes(text):
    """
    Generates Huffman codes based on joint (pairwise) symbols from the text.
    
    If the text length is odd, the last pair is padded with an underscore.
    
    Parameters:
        text (str): The input string.
        
    Returns:
        list: A list of tuples containing each combined symbol and its Huffman code.
    """
    n = len(text)
    step_size = 2
    pairs = [text[i:i+step_size] for i in range(0, n, step_size)]

    if n % step_size == 1:
        pairs[-1] += '_'

    counter = Counter(pairs)
    nodes = []
    for pair in counter:
        heappush(nodes, Node(counter[pair], pair))

    while len(nodes) > 1:
        left = heappop(nodes)
        right = heappop(nodes)

        left.huff = 0
        right.huff = 1

        new_node = Node(left.freq + right.freq, (left.symbol, right.symbol), left, right)
        heappush(nodes, new_node)

    codes_list = []
    if nodes:
        codes_list = extract_huffman_codes(nodes[0])

    return codes_list


def calculate_joint_average_codeword_length(text):
    """
    Calculates the average codeword length for joint Huffman coding (pairwise coding).
    
    Parameters:
        text (str): The input string.
        
    Returns:
        float: The joint average codeword length (in bits per symbol).
    """
    joint_avg_length = 0
    n = len(text)
    step_size = 2
    pairs = [text[i:i+step_size] for i in range(0, n, step_size)]
    
    if n % step_size == 1:
        pairs[-1] += '_'
        
    counter = Counter(pairs)
    codes = generate_joint_huffman_codes(text)
    
    for symbol, code in codes:
        joint_avg_length += ((counter[symbol] / len(pairs)) * len(code))
        
    return joint_avg_length


if __name__ == "__main__":
    run_text_analysis_tests()
