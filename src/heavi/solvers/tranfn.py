
#            __   ___  __      __   ___       ___       __   __         ___      ___ 
# |  | |\ | |  \ |__  |__)    |  \ |__  \  / |__  |    /  \ |__)  |\/| |__  |\ |  |  
# \__/ | \| |__/ |___ |  \    |__/ |___  \/  |___ |___ \__/ |     |  | |___ | \|  |  
# -------------------------------------------

from numba import njit, prange, c16, i8, f8, typed, types, typeof, jit
import numba_progress as nbp
from numba_progress.progress import ProgressBarType
import numpy as np

#placeholder decorator that instead of compiling with numba prints the type using typeof of the called arguments and then executes the function
# def njit(*args, **kwargs):
#     def wrapper(func):
#         def wrapped(*args, **kwargs):
#             return func(*args, **kwargs)
#         return wrapped
#     return wrapper

# def jit(*args, **kwargs):
#     def wrapper(func):
#         def wrapped(*args, **kwargs):
#             return func(*args, **kwargs)
#         return wrapped
#     return wrapper

# Diode data structure
# Tuple[nodeid1, nodeid2, Idata, Vdata]

#### THESE ARE EXAMPLES OF THE DATA STRUCTURES USED IN THE FUNCTION AND ONLY INTENDED TO DEFINE NUMBA TYPE HINTS

_DIODE_EXAMPLE = [(1,2,np.array([1.,2.,3.]),np.array([1.,2.,3.])), (4,5,np.array([1.,2.,3.]),np.array([1.,2.,3.]))]
_LC_EXAMPLE = [(0,1,2,0,1e-12), (1,4,5,0,1e-12)]
_VSIGNALS_EXAMPLE = [(1,np.linspace(0,1,12),np.linspace(0,1,12)), (3,np.linspace(0,1,13),np.linspace(0,1,12))]

@njit(types.Tuple((f8[:,:],f8[:]))(f8, f8, f8[:], f8[:]), cache=True, fastmath=True)
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

@njit(types.Tuple((f8[:,:],f8[:]))(f8[:,:], f8[:], i8, i8, f8, f8, f8), cache=True, fastmath=True)
def implement_capacitor(A, x, i1, i2, C, Vold, dt):
    A[i1,i1] += C/dt
    A[i2,i2] += C/dt
    A[i1,i2] -= C/dt
    A[i2,i1] -= C/dt
    x[i1] += C*(Vold)/dt
    x[i2] -= C*(Vold)/dt
    return A, x

@njit(types.Tuple((f8[:,:],f8[:]))(f8[:,:], f8[:], i8, i8, i8, f8, f8, f8, f8), cache=True, fastmath=True)
def implement_inductor(A, x, i1, i2, vi, L, Vold, Iold, dt):
    A[i1, vi] += 1
    A[vi, i1] += 1
    A[i2, vi] += -1
    A[vi, i2] += -1
    A[vi, vi] += -L*2/dt
    x[vi] = -L*2*(Iold)/dt - Vold
    return A, x

@jit(f8[:,:](f8[:,:], f8[:], 
             i8, 
             i8, 
             typeof(_VSIGNALS_EXAMPLE), 
             f8[:], 
             i8, 
             typeof(_DIODE_EXAMPLE), 
             typeof(_LC_EXAMPLE),
             f8, 
             ProgressBarType), 
      cache=True, fastmath=True, nogil=True)
def solve_MNA_TRAN(Alin, Xlin, N, M, Vsignals, rampsteps, Niter, diodes, lcs,  alpha, pgb):
    """"""

    NM = N+M

    Vsignals = Vsignals[1:]

    time_series = Vsignals[0][2]
    dt = time_series[1] - time_series[0]

    #n_signal_sources = len(Vsignals)
    n_time_steps = Vsignals[0][2].shape[0]


    # Initialize the S-parameter matrix
    v_data_time = np.zeros((NM,n_time_steps))
    v_data = np.zeros((NM,))

    Amat = np.zeros_like(Alin)
    Xvec = np.zeros_like(Xlin)
    Xbase = np.zeros_like(Xlin)

    diodes = diodes[1:]
    
    extended_rampsteps = np.repeat(rampsteps, Niter)

    Xold = np.zeros_like(Xlin)
    Xnew = np.zeros_like(Xlin)

    for it in range(n_time_steps):
        Xold = 1*Xbase
        Xnew = 1*Xlin
        
        for source in Vsignals:
            Xnew[source[0]] = source[1][it]
        
        Xdiff = Xnew - Xold
        
        # Reset voltage vector
        v_new = np.zeros((NM,))
        
        for r in extended_rampsteps:
            #for i in range(Niter):
            Amat = Amat*0 + Alin
            Xvec = Xold + Xdiff*r

            for diode in diodes:
                i1, i2 = diode[0], diode[1]
                (G, I) = compute_diode(v_data[i1], v_data[i2], diode[2], diode[3])
                Amat = add_submatrix(Amat, G, np.array([i1,i2]))
                Xvec[i1] = I[0]
                Xvec[i2] = I[1]
            
            for lc in lcs[1:]:
                tp, i1, i2,vi,lcval = lc
                vold = v_data_time[i1,it-1] - v_data_time[i2,it-1]
                iold = v_data_time[vi,it-1]

                if tp == 0:
                    Amat, Xvec = implement_capacitor(Amat, Xvec, i1, i2, lcval, vold, dt)
                if tp == 1:
                    Amat, Xvec = implement_inductor(Amat, Xvec, i1, i2, vi, lcval, vold, iold, dt)
            
            Amat = Amat + np.eye(NM)*1e-15
            #Xvec = Xvec + np.random.randn(NM)*1e-15
            if np.all(Xvec[1:] == 0):
                v_new = v_data
            else:
                v_new[1:] = np.linalg.solve(Amat[1:,1:], Xvec[1:])
                #v_new[1:] = np.linalg.lstsq(Amat[1:,1:], Xvec[1:])[0]
                #v_new[1:] = np.linalg.pinv(Amat[1:,1:]) @ Xvec[1:]

            v_data = v_data + alpha*(v_new-v_data)

        v_data_time[:,it] = v_data
        if it % 2 == 0:
            pgb.update(2)
        #pgb.update(1)
    return v_data_time

@njit(f8[:,:](f8[:,:], f8[:], i8, i8, typeof(_VSIGNALS_EXAMPLE), f8[:], i8, typeof(_DIODE_EXAMPLE), f8), 
      cache=True, parallel=True, fastmath=True)
def solve_MNA_TRAN_nopgb(Alin, Xlin, N, M, Vsignals, rampsteps, Niter, diodes, alpha):
    """"""

    NM = N+M

    Vsignals = Vsignals[1:]
    #n_signal_sources = len(Vsignals)
    n_time_steps = Vsignals[0][2].shape[0]

    # Initialize the S-parameter matrix
    v_data_time = np.zeros((N,n_time_steps))
    v_data = np.zeros((N,))

    Amat = np.zeros_like(Alin)
    Xvec = np.zeros_like(Xlin)
    Xbase = np.zeros_like(Xlin)

    diodes = diodes[1:]

    for it in range(n_time_steps):
        Xbase = 0*Xbase + Xlin
        v_data = 0*v_data

        for source in Vsignals:
            i1 = source[0]
            Vdata = source[1]
            Xbase[i1] = Vdata[it]
            
        for r in rampsteps:
            for i in range(Niter):
                Amat = Amat*0 + Alin
                Xvec = Xvec*0 + Xbase*r

                for diode in diodes:
                    i1, i2 = diode[0], diode[1]
                    (G, I) = compute_diode(v_data[i1], v_data[i2], diode[2], diode[3])
                    Amat = add_submatrix(Amat, G, np.array([i1,i2]))
                    Xvec[i1] = I[0]
                    Xvec[i2] = I[1]

                # Reset voltage vector
                v_new = np.zeros((NM,))
                
                if np.all(Xvec[1:] == 0):
                    v_new = v_data
                else:
                    v_new[1:] = np.linalg.solve(Amat[1:,1:], Xvec[1:])

                
                # Check convergence
                maxchange = np.max(np.abs(v_new[:N] - v_data[:N]))
                maxV = np.max(np.abs(v_new[:N]))

                #if maxchange < eaps + erel*maxV:
                #    v_data = v_new[:N]
                #    break
                v_data = v_data + alpha*(v_new[:N] - v_data)
                #v_data = v_new[:N]
        v_data_time[:,it] = v_data
    return v_data_time