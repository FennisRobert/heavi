########################################################################################
##
##    Numba accelerated S-parameter calculation for RF networks
##    This code is part of the Heavi framework
##
##    Author: Robert Fennis
##    Date: 2025
##
########################################################################################

#          __   __   __  ___  __  
# |  |\/| |__) /  \ |__)  |  /__` 
# |  |  | |    \__/ |  \  |  .__/ 
# -------------------------------------------


from numba import njit, prange, c16, i8, f8
import numba_progress as nbp
from numba_progress.progress import ProgressBarType
import numpy as np


#  __   __             ___  __   __  
# /__` /  \ |    \  / |__  |__) /__` 
# .__/ \__/ |___  \/  |___ |  \ .__/ 
# -------------------------------------------


@njit(cache=True, parallel=True, fastmath=True)
def solve_MNA_RF(As, Zs, port_indices, frequencies, nnodes, nsources, progress_object):
    """
    Compute the S-parameter matrix for an RF network using Numba for acceleration.

    Parameters
    ----------
    Is : numpy.ndarray
        Current sources, complex-valued array of shape (n_nodes, n_ports, n_freqs).
    Ys : numpy.ndarray
        Admittance matrices, complex-valued array of shape (n_nodes, n_nodes, n_freqs).
    Zs : numpy.ndarray
        Source impedances, complex-valued array of shape (n_nodes, n_freqs).
    port_indices : numpy.ndarray
        Indices of the nodes corresponding to the ports of interest, integer array of shape (n_ports,).
    frequencies : numpy.ndarray
        Frequencies, float-valued array of shape (n_freqs,).

    Returns
    -------
    S_parameters : numpy.ndarray
        S-parameter matrix, complex-valued array of shape (n_ports, n_ports, n_freqs).
    """
    M = nsources
    N = nnodes
    nports = port_indices.shape[0]
    p_index = port_indices[:,0]
    p_out_nodes = port_indices[:,1]
    p_int_nodes = port_indices[:,2]
    p_gnd_nodes = port_indices[:,3]

    num_freqs = len(frequencies)
    NM = N+M
    
    # Initialize the S-parameter matrix
    v_data = np.zeros((N, num_freqs), dtype=np.complex128)
    S_parameters = np.zeros((nports, nports, num_freqs), dtype=np.complex128)
    # x vector placeholder

    for i in range(nports):

        active_port_index = p_index[i]

        for freq_idx in prange(num_freqs):
            # Reset voltage vector
            x = np.zeros((NM,), dtype=np.complex128)
            z = np.zeros((NM,), dtype=np.complex128)
            # Set 1V at the active port node
            z[N + active_port_index] = 1
            # Solve the system of equations for Vh[1:]
            x[1:] = np.linalg.lstsq(As[1:,1:,freq_idx], z[1:])[0].astype(np.complex128)
            #x[1:] = np.linalg.solve(As[1:,1:,freq_idx], z[1:])
            #x[1:] = np.linalg.pinv(subA[:,:,freq_idx]) @ z
            Z_in = Zs[active_port_index, freq_idx]

            v_data[:,freq_idx] = x[:N]

            Vi1 = x[p_out_nodes[i]]
            Vi2 = x[p_gnd_nodes[i]]
            Vi3 = x[p_int_nodes[i]]

            for j in range(nports):
                Z_out = Zs[p_index[j], freq_idx]
                # Compute numerator and denominator for S-parameter calculation
                Vo1 = x[p_out_nodes[j]]
                Vo2 = x[p_gnd_nodes[j]]
                Vo3 = x[p_int_nodes[j]]
                numerator = (Vo1 - Vo2) + np.conj(Z_out) * (Vo1 - Vo3) / Z_out
                denominator = (Vi1 - Vi2) -  Z_in * (Vi1-Vi3) / Z_in
                # Compute S-parameter
                S_parameters[j, i, freq_idx] = (numerator / denominator) * np.sqrt(np.abs(np.real(Z_in))) / np.sqrt(np.abs(np.real(Z_out)))
            
            progress_object.update(1)

    return v_data, S_parameters


@njit(cache=True, parallel=True, fastmath=True)
def solve_MNA_RF_nopgb(As, Zs, port_indices, frequencies, nnodes, nsources):
    """
    Compute the S-parameter matrix for an RF network using Numba for acceleration.

    Parameters
    ----------
    Is : numpy.ndarray
        Current sources, complex-valued array of shape (n_nodes, n_ports, n_freqs).
    Ys : numpy.ndarray
        Admittance matrices, complex-valued array of shape (n_nodes, n_nodes, n_freqs).
    Zs : numpy.ndarray
        Source impedances, complex-valued array of shape (n_nodes, n_freqs).
    port_indices : numpy.ndarray
        Indices of the nodes corresponding to the ports of interest, integer array of shape (n_ports,).
    frequencies : numpy.ndarray
        Frequencies, float-valued array of shape (n_freqs,).

    Returns
    -------
    S_parameters : numpy.ndarray
        S-parameter matrix, complex-valued array of shape (n_ports, n_ports, n_freqs).
    """
    M = nsources
    N = nnodes
    nports = port_indices.shape[0]
    p_index = port_indices[:,0]
    p_out_nodes = port_indices[:,1]
    p_int_nodes = port_indices[:,2]
    p_gnd_nodes = port_indices[:,3]

    num_freqs = len(frequencies)
    NM = N+M
    
    # Initialize the S-parameter matrix
    v_data = np.zeros((N, num_freqs), dtype=np.complex128)
    S_parameters = np.zeros((nports, nports, num_freqs), dtype=np.complex128)
    # x vector placeholder

    for i in range(nports):

        active_port_index = p_index[i]

        for freq_idx in prange(num_freqs):
            # Reset voltage vector
            x = np.zeros((NM,), dtype=np.complex128)
            z = np.zeros((NM,), dtype=np.complex128)
            # Set 1V at the active port node
            z[N + active_port_index] = 1
            # Solve the system of equations for Vh[1:]
            x[1:] = np.linalg.lstsq(As[1:,1:,freq_idx], z[1:])[0].astype(np.complex128)
            #x[1:] = np.linalg.solve(As[1:,1:,freq_idx], z[1:])
            #x[1:] = np.linalg.pinv(subA[:,:,freq_idx]) @ z
            Z_in = Zs[active_port_index, freq_idx]

            v_data[:,freq_idx] = x[:N]

            Vi1 = x[p_out_nodes[i]]
            Vi2 = x[p_gnd_nodes[i]]
            Vi3 = x[p_int_nodes[i]]

            for j in range(nports):
                Z_out = Zs[p_index[j], freq_idx]
                # Compute numerator and denominator for S-parameter calculation
                Vo1 = x[p_out_nodes[j]]
                Vo2 = x[p_gnd_nodes[j]]
                Vo3 = x[p_int_nodes[j]]
                numerator = (Vo1 - Vo2) + np.conj(Z_out) * (Vo1 - Vo3) / Z_out
                denominator = (Vi1 - Vi2) -  Z_in * (Vi1-Vi3) / Z_in
                # Compute S-parameter
                S_parameters[j, i, freq_idx] = (numerator / denominator) * np.sqrt(np.abs(np.real(Z_in))) / np.sqrt(np.abs(np.real(Z_out)))
            

    return v_data, S_parameters