import numpy as np

def S11_to_VSWR(S11: np.ndarray) -> np.ndarray:
    """Convert S11 to VSWR"""
    return (1 + np.abs(S11)) / (1 - np.abs(S11))

def S11_to_impedance(S11: np.ndarray, Z0: float = 50) -> np.ndarray:
    """Convert S11 to impedance"""
    return Z0 * (1 + S11) / (1 - S11)

def Z_to_S11(Z: np.ndarray, Z0: float = 50) -> np.ndarray:
    """Convert impedance to S11"""
    return (Z - Z0) / (Z + Z0)

def VSWR_to_S11(VSWR: np.ndarray) -> np.ndarray:
    """Convert VSWR to S11"""
    return (VSWR - 1) / (VSWR + 1)