def read_arithmetic_input(file_path):
    """
    Reads the input file and returns a list of strings (one per line).
    
    Parameters:
        file_path (str): Path to the input file.
    
    Returns:
        list: List of strings read from the file.
    """
    lines = []
    with open(file_path, 'r') as file:
        for line in file:
            lines.append(line)
    return lines


def run_arithmetic_encoding_tests():
    """
    Runs tests for the arithmetic encoding function.
    
    It processes each input string from 'input3.txt' twice:
      1. Without scaling operations.
      2. With scaling operations.
      
    The results are printed to the console.
    """
    file_path = "input1.txt"
    input_strings = read_arithmetic_input(file_path)
    
    # Test without scaling operations
    for input_string in input_strings:
        print("Input string:", input_string)
        lower_bound, upper_bound, _ = arithmetic_encode(input_string, enable_scaling=False)
        print("Bounds without E1/E2 scaling:", lower_bound, upper_bound)
        print()
        
    # Test with scaling operations enabled
    for input_string in input_strings:
        print("Input string:", input_string)
        lower_bound, upper_bound, operations = arithmetic_encode(input_string, enable_scaling=True)
        print("Bounds with E1/E2 scaling:", lower_bound, upper_bound)
        print("Scaling operations:", operations)
        print()


def arithmetic_encode(input_string, enable_scaling=False):
    """
    Performs arithmetic encoding on the input string for symbols 'A' and 'B'.
    
    The encoding calculates lower and upper bounds based on the cumulative
    distribution of symbols in the string. If scaling is enabled, the function
    applies scaling operations (E1 and E2) when the interval falls entirely
    in the lower or upper half of [0, 1].
    
    Parameters:
        input_string (str): The input string to encode.
        enable_scaling (bool): Whether to apply scaling operations.
    
    Returns:
        list: [lower_bound, upper_bound, operations]
              - lower_bound (float): The final lower bound of the encoding interval.
              - upper_bound (float): The final upper bound of the encoding interval.
              - operations (list): List of scaling operations performed.
    """
    lower_bound = 0.0
    upper_bound = 1.0
    operations = []

    # Calculate probabilities for each character
    prob_A = input_string.count('A') / len(input_string)
    prob_B = input_string.count('B') / len(input_string)
    
    # Cumulative distribution function values
    cdf_A = prob_A
    cdf_B = prob_A + prob_B  # Should be 1 if only 'A' and 'B' are present

    for char in input_string:
        current_range = upper_bound - lower_bound
        if char == "A":
            upper_bound = lower_bound + (current_range * cdf_A)
        else:  # Assuming the character is 'B'
            lower_bound = lower_bound + (current_range * cdf_A)
            upper_bound = lower_bound + (current_range * (cdf_B - cdf_A))

        if enable_scaling:
            # Apply scaling operations if conditions are met
            while True:
                if lower_bound < 0.5 and upper_bound < 0.5:
                    lower_bound *= 2
                    upper_bound *= 2
                    operations.append("E1 Scaling")
                elif lower_bound >= 0.5 and upper_bound >= 0.5:
                    lower_bound = 2 * (lower_bound - 0.5)
                    upper_bound = 2 * (upper_bound - 0.5)
                    operations.append("E2 Scaling")
                else:
                    operations.append("No Scaling")
                    break

    return [lower_bound, upper_bound, operations]


if __name__ == "__main__":
    run_arithmetic_encoding_tests()
