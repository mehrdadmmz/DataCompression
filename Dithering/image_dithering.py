import numpy as np

# Default dither matrix
DITHER_MATRIX = np.array([
    [0, 3],
    [2, 1]
])

def read_image_input(file_path):
    """
    Reads an image from a CSV text file, where each line represents a row of pixel values.
    
    Parameters:
        file_path (str): Path to the input file.
    
    Returns:
        np.array: 2D NumPy array representing the image.
    """
    image = []
    with open(file_path, 'r') as file:
        for line in file:
            row = []
            values = line.strip().split(',')
            for item in values:
                row.append(int(item))
            image.append(row)
    return np.array(image)


def run_dithering_tests():
    """
    Runs tests for both dithering and ordered dithering functions.
    
    The input image is read from 'input2.txt'.
    """
    input_file = "input2.txt"
    input_image = read_image_input(input_file)
    print("Input image:\n", input_image)
    
    dithered_image = apply_dithering(input_image, DITHER_MATRIX)
    ordered_dithered_image = apply_ordered_dithering(input_image, DITHER_MATRIX)
    
    print("Dithering result:\n", dithered_image)
    print("Ordered dithering result:\n", ordered_dithered_image)


def normalize_image(image, intensity_level, minimum=0, maximum=256):
    """
    Normalizes and quantizes the image.
    
    The intensity level should be equal to (n ** 2) + 1,
    where n is the size of the dither matrix.
    
    Parameters:
        image (np.array): Input image.
        intensity_level (int): The intensity level.
        minimum (int): Minimum intensity (default 0).
        maximum (int): Maximum intensity (default 256).
    
    Returns:
        np.array: Normalized and quantized image.
    """
    normalized_matrix = np.floor((image / maximum) * intensity_level).astype(int)
    return normalized_matrix


def apply_dithering(image, dither):
    """
    Applies dithering to the image using the provided dither matrix.
    
    The image is first normalized and then each pixel's intensity is compared
    to the entire dither matrix. The result is an expanded output image.
    
    Parameters:
        image (np.array): Input image.
        dither (np.array): Dither matrix.
    
    Returns:
        np.array: Dithered image with expanded dimensions.
    """
    # Normalize the input image
    intensity_level = dither.shape[0] * dither.shape[1] + 1
    normalized_image = normalize_image(image, intensity_level)
    
    # Calculate expanded output dimensions
    expanded_rows = normalized_image.shape[0] * dither.shape[0]
    expanded_cols = normalized_image.shape[1] * dither.shape[1]
    
    # Initialize the expanded output image
    output_image = np.zeros((expanded_rows, expanded_cols), dtype=int)
    
    # Apply the dithering process
    for i in range(normalized_image.shape[0]):
        for j in range(normalized_image.shape[1]):
            submatrix = (normalized_image[i, j] > dither).astype(int)
            output_image[i * dither.shape[0]:(i + 1) * dither.shape[0],
                         j * dither.shape[1]:(j + 1) * dither.shape[1]] = submatrix

    return output_image


def apply_ordered_dithering(image, dither):
    """
    Applies ordered dithering to the image using the provided dither matrix.
    
    The image is normalized and then processed in blocks the size of the dither matrix.
    The output image retains the same dimensions as the original image.
    
    Parameters:
        image (np.array): Input image.
        dither (np.array): Dither matrix.
    
    Returns:
        np.array: Image after applying ordered dithering.
    """
    # Normalize the input image
    intensity_level = dither.shape[0] * dither.shape[1] + 1
    normalized_image = normalize_image(image, intensity_level)
    
    # Ensure the dither matrix dimensions evenly divide the image dimensions
    assert normalized_image.shape[0] % dither.shape[0] == 0, "Image rows must be divisible by dither rows."
    assert normalized_image.shape[1] % dither.shape[1] == 0, "Image columns must be divisible by dither columns."
    
    # Create an output image of the same size
    output_image = np.zeros_like(normalized_image)
    
    dither_rows, dither_cols = dither.shape
    
    # Apply ordered dithering in blocks
    for i in range(0, normalized_image.shape[0], dither_rows):
        for j in range(0, normalized_image.shape[1], dither_cols):
            submatrix = normalized_image[i:i+dither_rows, j:j+dither_cols]
            result = (submatrix > dither).astype(int)
            output_image[i:i+dither_rows, j:j+dither_cols] = result

    return output_image


if __name__ == "__main__":
    run_dithering_tests()
