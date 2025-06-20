#!/usr/bin/env python3
"""function to add 2D matrix into a new one"""


def add_matrices2D(mat1, mat2):
    """ adding two matrix element wise

    Args:
        mat1, mat2: Given matrix

    Return:
        the sum of matrix: new matrix

    """
    if (len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0])):
        return None
    else:
        new_mat = []
        tmp = []
        for i in range(len(mat1)):
            for x in range(len(mat1[0])):
                tmp.append(mat1[i][x] + mat2[i][x])
            new_mat.append(tmp)
            tmp = []

        return new_mat
