#!/usr/bin/env python3
"""function to concatenate 2 matrices along
    a specific axis"""


def cat_matrices2D(mat1, mat2, axis=0):
    """ concatenate two matrices
     with an specific axis

    Args:
        mat1, mat2: Given matrices
        axis: Given axis

    Return:
        the new mat: new_mat

    """

    mat_new=[]
    if (axis==0):
        mat_new=mat1+mat2
    elif (axis==1):
        for i in range(len(mat1)):
            mat_new.append(mat1[i]+mat2[i])
    else:
        return None
   
    return mat_new
