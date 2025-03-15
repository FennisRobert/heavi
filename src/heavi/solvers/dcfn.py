
########################################################################################
##
##    DC Solver for the MNA with Numba compilation
##    This file contains the DC solver for the MNA using Numba compilation
##
##    Author: Robert Fennis
##    Date: 2025
##
########################################################################################

#          __   __   __  ___  __  
# |  |\/| |__) /  \ |__)  |  /__` 
# |  |  | |    \__/ |  \  |  .__/ 
# -------------------------------------------


from numba import njit, prange, c16, i8, f8, typed, types, typeof
import numba_progress as nbp
from numba_progress.progress import ProgressBarType
import numpy as np

#  ___            __  ___    __        __  
# |__  |  | |\ | /  `  |  | /  \ |\ | /__` 
# |    \__/ | \| \__,  |  | \__/ | \| .__/ 
# -------------------------------------------


#placeholder decorator that instead of compiling with numba prints the type using typeof of the called arguments and then executes the function
# def njit(*args, **kwargs):
#     def wrapper(func):
#         def wrapped(*args, **kwargs):
#             print([typeof(arg) for arg in args])
#             return func(*args, **kwargs)
#         return wrapped
#     return wrapper


# Diode data structure
# Tuple[nodeid1, nodeid2, Idata, Vdata]

example = [(1,2,np.array([1.,2.,3.]),np.array([1.,2.,3.])), (4,5,np.array([1.,2.,3.]),np.array([1.,2.,3.]))]



@njit(types.Tuple((f8[:,:],f8[:]))(f8, f8, f8[:], f8[:]), cache=True, parallel=True, fastmath=True)
def compute_diode(v1, v2, Idata, Vdata):
    
    EPS = 1e-15
    dV = v1-v2
    I0 = np.interp(dV, Vdata, Idata)
    i2 = np.interp(dV+EPS, Vdata, Idata)
    i1 = np.interp(dV-EPS, Vdata, Idata)
    G0 = (i2-i1)/(2*EPS)
    G = np.array([[1,-1],[-1,1]]) * G0
    I = np.array([-1,1]) * (I0 - G0*dV)
    return G, I

@njit(f8[:,:](f8[:,:], f8[:,:], i8[:]), cache=True, fastmath=True)
def add_submatrix(larger_matrix, smaller_matrix, indices):
    n = indices.shape[0]
    for i in range(n):
        for j in range(n):
            larger_matrix[indices[i], indices[j]] += smaller_matrix[i, j]
    return larger_matrix

@njit(f8[:](f8[:,:], f8[:], i8, i8, f8[:], i8, typeof(example), f8, f8), 
      cache=True, parallel=True, fastmath=True)
def solve_MNA_DC(Alin, Xlin, N, M, rampsteps, Niter, diodes, eaps, erel):
    """"""

    NM = N+M

    # Initialize the S-parameter matrix
    v_data = np.zeros((N,))
    
    Amat = np.zeros_like(Alin)
    Xvec = np.zeros_like(Xlin)

    diodes = diodes[1:]
    for r in rampsteps:
        for i in range(Niter):
            Amat = Amat*0 + Alin
            Xvec = Xvec*0 + Xlin*r

            for diode in diodes:
                i1, i2 = diode[0], diode[1]
                (G, I) = compute_diode(v_data[i1], v_data[i2], diode[2], diode[3])
                Amat = add_submatrix(Amat, G, np.array([i1,i2]))
                Xvec[i1] = I[0]
                Xvec[i2] = I[1]

            # Reset voltage vector
            v_new = np.zeros((NM,))
            
            v_new[1:] = np.linalg.solve(Amat[1:,1:], Xvec[1:])

            
            # Check convergence
            maxchange = np.max(np.abs(v_new[:N] - v_data[:N]))
            maxV = np.max(np.abs(v_new[:N]))

            if maxchange < eaps + erel*maxV:
                v_data = v_new[:N]
                break
            v_data = v_new[:N]

    return v_data