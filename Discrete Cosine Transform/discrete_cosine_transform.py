import math
import numpy as np

def read_dct_matrix_sizes(file_path):
    """
    Read DCT matrix sizes from a file.
    
    Each line in the file should contain an integer.
    """
    numbers = []
    with open(file_path, 'r') as file:
        for line in file:
            numbers.append(int(line.strip()))
    return numbers

def compute_alpha(i, N): 
    """
    Compute the scaling factor for the DCT.
    
    Returns sqrt(1/N) for i == 0, else sqrt(2/N).
    """
    if i == 0: 
        return math.sqrt(1/N)
    else: 
        return math.sqrt(2/N)
    
def compute_dct_matrix(n):
    """
    Compute and return the Discrete Cosine Transformation (DCT) matrix of size n x n.
    """
    dct_matrix = np.zeros((n, n))
    for i in range(n): 
        for j in range(n): 
            dct_matrix[i, j] = compute_alpha(i, n) * math.cos((2*j+1)*i*math.pi / (2*n))
    return dct_matrix

def test_dct_matrix():
    """
    Test the DCT matrix computation by reading sizes from a file.
    """
    file_path = "input2_1.txt"
    sizes = read_dct_matrix_sizes(file_path)
    for size in sizes:
        matrix = compute_dct_matrix(size)
        print("DCT of size", size, ":", matrix)
        print()

def read_vector_inputs(file_path):
    """
    Read vector inputs from a file.
    
    Each line in the file should be a string representation of a list, e.g., "[1,2,3]".
    """
    list_of_lists = []
    with open(file_path, 'r') as file:
        for line in file:
            numbers = line.strip()[1:-1].split(',')
            list_of_lists.append([int(number) for number in numbers])
    return list_of_lists

def compute_dct_transform(input_vector):
    """
    Compute and return the DCT transform result for the input vector.
    """
    N = len(input_vector)
    dct_matrix = compute_dct_matrix(N)
    coefficients = dct_matrix @ input_vector
    return coefficients

def test_dct_transform():
    """
    Test the DCT transform using vector inputs read from a file.
    """
    file_path = "input2_2.txt"
    vectors = read_vector_inputs(file_path)
    for vector in vectors:
        print("Input vector:", vector)
        results = compute_dct_transform(vector)
        print("DCT transformation results:", results)
        print()

def test_high_freq_vector():
    """
    Test the DCT transform on an input vector designed to have high energy in higher frequency components.
    
    Real-world example: an audio signal with sudden silence followed by a loud noise, or
    a high-contrast striped pattern in an image.
    """
    input_vector = np.array([(-1)**i for i in range(16)])
    results = compute_dct_transform(input_vector)
    print("DCT transformation results for high-frequency vector:", results)

if __name__ == '__main__':
    test_dct_matrix()
    test_dct_transform()
    test_high_freq_vector()
