#!/usr/bin/env python3
""" Transpose function to check a new matrix transposed"""


def matrix_transpose(matrix):
    """ return a new matrix transposed

    Args:
        matrix: given list

    Return:
        new_matrix: Transposed matrix

    """
    if (not isinstance(matrix[0], list)):
        return [len(matrix)]
    else:
        new_matrix = [[matrix[j][i] for j in range(len(matrix))]
                      for i in range(len(matrix[0]))]
    return new_matrix
