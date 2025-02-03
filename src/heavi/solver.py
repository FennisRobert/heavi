from numba import njit, prange, c16, i8, f8
import numba_progress as nbp
from numba_progress.progress import ProgressBarType
import numpy as np

@njit(cache=True, parallel=True, fastmath=True)
def solve_single_frequency_c_compiled(Is, Ys, Zs, indices, frequencies, progprox):
    nT = len(indices)
    nF = len(frequencies)
    Ss = np.zeros((nT,nT,nF), dtype=np.complex128)
    Vdonor = np.zeros((Is.shape[0],), dtype=np.complex128)
    for it in range(nT):
        ind1 = indices[it]
        for i in prange(nF):
            Vh = 0*Vdonor
            Vh[1:] = np.linalg.solve(Ys[:,:,i],Is[1:,it,i])
            for itt in range(nT):
                ind2 = indices[itt]
                Q = np.sqrt(
                    np.abs(np.real(Zs[it,i]))
                ) / np.sqrt(np.abs(np.real(Zs[itt,i])))
                Ss[itt, it, i] = Q * (
                    (Vh[ind2] * 2 - Zs[itt,i]* Is[ind2,it,i])
                    / (Zs[it,i] * Is[ind1,it,i])
                )
            progprox.update(1)
    return Ss

@njit(cache=True, parallel=True, fastmath=True)
def compute_s_parameters(Is, Ys, Zs, port_indices, frequencies, progress_object):
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
    num_ports = len(port_indices)
    num_freqs = len(frequencies)
    num_nodes = Is.shape[0]

    # Initialize the S-parameter matrix
    S_parameters = np.zeros((num_ports, num_ports, num_freqs), dtype=np.complex128)

    # Voltage vector placeholder
    Vh = np.zeros((num_nodes,), dtype=np.complex128)

    for port_in_idx in range(num_ports):
        node_in = port_indices[port_in_idx]
        for freq_idx in prange(num_freqs):
            # Reset voltage vector
            Vh = 0*Vh

            # Solve the system of equations for Vh[1:]
            Vh[1:] = np.linalg.solve(
                Ys[:,:, freq_idx],
                Is[1:, port_in_idx, freq_idx]
            )

            Z_in = Zs[port_in_idx, freq_idx]

            for port_out_idx in range(num_ports):
                node_out = port_indices[port_out_idx]
                Z_out = Zs[port_out_idx, freq_idx]

                # Calculate scaling factor Q
                Q = np.sqrt(np.abs(np.real(Z_in))) / np.sqrt(np.abs(np.real(Z_out)))

                # Compute numerator and denominator for S-parameter calculation
                numerator = Vh[node_out] * 2 - Z_out * Is[node_out, port_in_idx, freq_idx]
                denominator = Z_in * Is[node_in, port_in_idx, freq_idx]

                # Compute S-parameter
                S_parameters[port_out_idx, port_in_idx, freq_idx] = Q * (numerator / denominator)
            progress_object.update(1)
    return S_parameters

@njit(cache=True, parallel=True, fastmath=True)
def compute_s_parameters_no_loadbar(Is, Ys, Zs, port_indices, frequencies):
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
    num_ports = len(port_indices)
    num_freqs = len(frequencies)
    num_nodes = Is.shape[0]

    # Initialize the S-parameter matrix
    S_parameters = np.zeros((num_ports, num_ports, num_freqs), dtype=np.complex128)

    # Voltage vector placeholder
    Vh = np.zeros((num_nodes,), dtype=np.complex128)

    for port_in_idx in range(num_ports):
        node_in = port_indices[port_in_idx]
        for freq_idx in prange(num_freqs):
            # Reset voltage vector
            Vh = 0*Vh

            # Solve the system of equations for Vh[1:]
            Vh[1:] = np.linalg.solve(
                Ys[:,:, freq_idx],
                Is[1:, port_in_idx, freq_idx]
            )

            Z_in = Zs[port_in_idx, freq_idx]

            for port_out_idx in range(num_ports):
                node_out = port_indices[port_out_idx]
                Z_out = Zs[port_out_idx, freq_idx]

                # Calculate scaling factor Q
                Q = np.sqrt(np.abs(np.real(Z_in))) / np.sqrt(np.abs(np.real(Z_out)))

                # Compute numerator and denominator for S-parameter calculation
                numerator = Vh[node_out] * 2 - Z_out * Is[node_out, port_in_idx, freq_idx]
                denominator = Z_in * Is[node_in, port_in_idx, freq_idx]

                # Compute S-parameter
                S_parameters[port_out_idx, port_in_idx, freq_idx] = Q * (numerator / denominator)
    return S_parameters