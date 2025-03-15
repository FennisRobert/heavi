import numpy as np
import numba as nb

@nb.njit(cache=True)
def stack(matrix: np.ndarray, vector: np.ndarray, add: np.ndarray) -> np.ndarray:
    n1, n2 = matrix.shape
    n3 = vector.shape[0]
    newmat = np.zeros((n1, n2, n3), dtype=np.float64)
    for i in range(n3):
        newmat[:, :, i] = matrix * vector[i] + add
    return newmat

@nb.njit(cache=True)
def cstack(matrix: np.ndarray, vector: np.ndarray, add: np.ndarray) -> np.ndarray:
    n1, n2 = matrix.shape
    n3 = vector.shape[0]
    newmat = np.zeros((n1, n2, n3), dtype=np.complex128)
    for i in range(n3):
        newmat[:, :, i] = matrix * vector[i] + add
    return newmat