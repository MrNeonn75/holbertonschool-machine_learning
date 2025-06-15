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
        
        trans_mat = []
        tmp =[]
        for i in range(len(matrix[0])):
            for list in matrix:
                tmp.append(list[i])
            trans_mat.append(tmp)
            tmp = []
    
    return trans_mat

