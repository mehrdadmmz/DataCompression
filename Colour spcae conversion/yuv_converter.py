import os
import math
import numpy as np
from heapq import heappush, heappop
from collections import Counter

def read_input_data(file_path):
    """Reads input data from a file and returns a list of (Y, Co, Cg) tuples."""
    data_points = []
    with open(file_path, 'r') as file:
        for line in file:
            values = line.strip().split(',')
            if len(values) == 3:
                try:
                    y = int(values[0])
                    co = int(values[1])
                    cg = int(values[2])
                    data_points.append((y, co, cg))
                except ValueError:
                    print(f"Invalid line: {line}")
    return data_points


def run_conversion_tests():
    """Runs a test conversion from YCoCg to YUV for each line in the input file."""
    input_file = "input1.txt"
    data_points = read_input_data(input_file)

    for y, co, cg in data_points:
        yuv = convert_ycocg_to_yuv(y, co, cg)
        print(f"YCoCg: ({y}, {co}, {cg}) -> YUV: (Y={yuv[0]}, U={yuv[1]}, V={yuv[2]})")


def convert_ycocg_to_yuv(y, co, cg):
    """
    Converts YCoCg values to YUV.
    
    Conversion formula:
      1. Rearrange (Cg, Y, Co) to form a matrix.
      2. Multiply by the first conversion matrix to obtain GBR.
      3. Rotate GBR to get RGB.
      4. Multiply by the second conversion matrix to compute YUV.
    """
    # Converting (Cg, Y, Co) to GBR
    cg_yco = np.array([[cg], [y], [co]])
    conversion_matrix1 = np.array([[1, 1, 0],
                                   [-1, 1, -1],
                                   [-1, 1, 1]])
    gbr_values = np.matmul(conversion_matrix1, cg_yco)
    
    # Converting GBR to RGB by rotating it
    rgb_values = np.roll(gbr_values, 1, axis=0)
    conversion_matrix2 = np.array([[0.299, 0.587, 0.114],
                                   [-0.299, -0.587, 0.886],
                                   [0.701, -0.587, -0.114]])
    yuv_matrix = np.matmul(conversion_matrix2, rgb_values)

    # Extracting Y, U, V from the YUV matrix
    y_result = yuv_matrix[0][0]
    u_result = yuv_matrix[1][0]
    v_result = yuv_matrix[2][0]

    return (y_result, u_result, v_result)


if __name__ == "__main__":
    run_conversion_tests()
