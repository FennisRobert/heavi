########################################################################################
##
##    Commonly used transformations in RF engineering
##    This module contains functions to convert between S11, VSWR and impedance
##
##    Author: Robert Fennis
##    Date: 2025
##
########################################################################################

#          __   __   __  ___  __  
# |  |\/| |__) /  \ |__)  |  /__` 
# |  |  | |    \__/ |  \  |  .__/ 
# -------------------------------------------

import numpy as np

#  ___            __  ___    __        __  
# |__  |  | |\ | /  `  |  | /  \ |\ | /__` 
# |    \__/ | \| \__,  |  | \__/ | \| .__/ 
# -------------------------------------------


def S11_to_VSWR(S11: np.ndarray) -> np.ndarray:
    """Convert S11 to VSWR

    Args:
        S11 (np.ndarray): The reflection coefficient

    Returns:
        np.ndarray: The Voltage Standing Wave Ratio
    """    
    return (1 + np.abs(S11)) / (1 - np.abs(S11))

def S11_to_Z(S11: np.ndarray, Z0: float = 50.) -> np.ndarray:
    """ Convert S11 to impedance

    Args:
        S11 (np.ndarray): The reflection coefficient
        Z0 (float, optional): The reference impedance. Defaults to 50.

    Returns:
        np.ndarray: The aparant impedance
    """    
    return Z0 * (1 + S11) / (1 - S11)

def Z_to_S11(Z: np.ndarray, Z0: float = 50) -> np.ndarray:
    """ Convert impedance to S11

    Args:
        Z (np.ndarray): The load impedance
        Z0 (float, optional): The reference impedance. Defaults to 50.

    Returns:
        np.ndarray: The reflection coefficient
    """    
    return (Z - Z0) / (Z + Z0)

def VSWR_to_S11(VSWR: np.ndarray) -> np.ndarray:
    """Convert VSWR to S11

    Args:
        VSWR (np.ndarray): The Voltage Standing Wave Ratio

    Returns:
        np.ndarray: The reflection coefficient
    """    
    return (VSWR - 1) / (VSWR + 1)

def VSWR_to_Z(VSWR: np.ndarray, Z0: float = 50) -> np.ndarray:
    """Convert VSWR to impedance

    Args:
        VSWR (np.ndarray): The Voltage Standing Wave Ratio
        Z0 (float, optional): The reference impedance. Defaults to 50.

    Returns:
        np.ndarray: The aparant impedance
    """    
    return S11_to_Z(VSWR_to_S11(VSWR), Z0)

def Z_to_VSWR(Z: np.ndarray, Z0: float = 50) -> np.ndarray:
    """Convert impedance to VSWR

    Args:
        Z (np.ndarray): The load impedance
        Z0 (float, optional): The reference impedance. Defaults to 50.

    Returns:
        np.ndarray: The Voltage Standing Wave Ratio
    """    
    return S11_to_VSWR(Z_to_S11(Z, Z0))