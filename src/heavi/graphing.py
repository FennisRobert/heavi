########################################################################################
##
##    Default plotting functions for the Heavi package
##    This file contains the default plotting functions for the Heavi package.
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
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
from matplotlib import pyplot as plt


#  __   ___  ___                ___  __      __   ___ ___ ___         __   __  
# |  \ |__  |__   /\  |  | |     |  /__`    /__` |__   |   |  | |\ | / _` /__` 
# |__/ |___ |    /~~\ \__/ |___  |  .__/    .__/ |___  |   |  | | \| \__> .__/ 
# -------------------------------------------

_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

ggplot_styles = {
    "axes.edgecolor": "000000",
    "axes.facecolor": "F2F2F2",
    "axes.grid": True,
    "axes.grid.which": "both",
    "axes.spines.left": True,
    "axes.spines.right": True,
    "axes.spines.top": True,
    "axes.spines.bottom": True,
    "grid.color": "A0A0A0",
    "grid.linewidth": "0.8",
    "xtick.color": "555555",
    "xtick.major.bottom": True,
    "xtick.minor.bottom": False,
    "ytick.color": "555555",
    "ytick.major.left": True,
    "ytick.minor.left": False,
    "lines.linewidth": 2,
}
plt.rcParams.update(ggplot_styles)

#  ___            __  ___    __        __  
# |__  |  | |\ | /  `  |  | /  \ |\ | /__` 
# |    \__/ | \| \__,  |  | \__/ | \| .__/ 
# -------------------------------------------


def _gen_grid(xs: tuple, ys: tuple, N = 201) -> list[np.ndarray]:
    """Generate a grid of lines for the Smith Chart

    Args:
        xs (tuple): Tuple containing the x-axis values
        ys (tuple): Tuple containing the y-axis values
        N (int, optional): Number Used. Defaults to 201.

    Returns:
        list[np.ndarray]: List of lines
    """    
    xgrid = np.arange(xs[0], xs[1]+xs[2], xs[2])
    ygrid = np.arange(ys[0], ys[1]+ys[2], ys[2])
    xsmooth = np.logspace(np.log10(xs[0]+1e-8), np.log10(xs[1]), N)
    ysmooth = np.logspace(np.log10(ys[0]+1e-8), np.log10(ys[1]), N)
    ones = np.ones((N,))
    lines = []
    for x in xgrid:
        lines.append((x*ones, ysmooth))
        lines.append((x*ones, -ysmooth))
    for y in ygrid:
        lines.append((xsmooth, y*ones))
        lines.append((xsmooth, -y*ones))
        
    return lines

def _generate_grids(orders = (0, 0.5, 1, 2, 5, 10, 50,1e5), N=201) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate the grid for the Smith Chart

    Args:
        orders (tuple, optional): Locations for Smithchart Lines. Defaults to (0, 0.5, 1, 2, 5, 10, 50,1e5).
        N (int, optional): N distrectization points. Defaults to 201.

    Returns:
        list[tuple[np.ndarray, np.ndarray]]: List of axes lines
    """    
    lines = []
    xgrids = orders
    for o1, o2 in zip(xgrids[:-1], xgrids[1:]):
        step = o2/10
        lines += _gen_grid((0, o2, step), (0, o2, step), N)   
    return lines

def _smith_transform(lines: list[tuple[np.ndarray, np.ndarray]]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Executes the Smith Transform on a list of lines

    Args:
        lines (list[tuple[np.ndarray, np.ndarray]]): List of lines

    Returns:
        list[tuple[np.ndarray, np.ndarray]]: List of transformed lines
    """    
    new_lines = []
    for line in lines:
        x, y = line
        z = x + 1j*y
        new_z = (z-1)/(z+1)
        new_x = new_z.real
        new_y = new_z.imag
        new_lines.append((new_x, new_y))
    return new_lines

def hintersections(x: np.ndarray, y: np.ndarray, level: float) -> list[float]:
    """Find the intersections of a line with a level

    Args:
        x (np.ndarray): X-axis values
        y (np.ndarray): Y-axis values
        level (float): Level to intersect

    Returns:
        list[float]: List of x-values where the intersection occurs
    """      
    y1 = y[:-1] - level
    y2 = y[1:] - level
    ycross = y1 * y2
    id1 = np.where(ycross < 0)[0]
    id2 = id1 + 1
    x1 = x[id1]
    x2 = x[id2]
    y1 = y[id1] - level
    y2 = y[id2] - level
    a = (y2 - y1) / (x2 - x1)
    b = y1 - a * x1
    xcross = list(-b / a)
    xlevel = list(x[np.where(y == level)])
    return xcross + xlevel


def plot(x: np.ndarray, y: np.ndarray) -> None:
    """Simple wrapper for an x-y plot

    Parameters
    ----------
    x : np.ndarray
        x-axis values
    y : np.ndarray
        y-axis values
    """

    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.grid()
    plt.show()


def smith(f: np.ndarray, S: np.ndarray) -> None:
    """ Plot the Smith Chart

    Args:
        f (np.ndarray): Frequency vector (Not used yet)
        S (np.ndarray): S-parameters to plot
    """     
    if not isinstance(S, list):
        Ss = [S]
    else:
        Ss = S
    
    fig, ax = plt.subplots()
    for line in _smith_transform(_generate_grids()):
        ax.plot(line[0], line[1], color='grey', alpha=0.3, linewidth=0.7)
    p = np.linspace(0,2*np.pi,101)
    ax.plot(np.cos(p), np.sin(p), color='black', alpha=0.5)
    # Add important numbers for the Impedance Axes
    # X and Y values (0, 0.5, 1, 2, 10, 50)
    for i in [0, 0.2, 0.5, 1, 2, 10]:
        z = i + 1j*0
        G = (z-1)/(z+1)
        ax.annotate(f"{i}", (G.real, G.imag), color='black', fontsize=8)
    for i in [0, 0.2, 0.5, 1, 2, 10]:
        z = 0 + 1j*i
        G = (z-1)/(z+1)
        ax.annotate(f"{i}", (G.real, G.imag), color='black', fontsize=8)       
        ax.annotate(f"{-i}", (G.real, -G.imag), color='black', fontsize=8)  
    for s in Ss:
        ax.plot(s.real, s.imag, color='blue')
    ax.grid(False)
    ax.axis('equal')
    plt.show()



def plot_s_parameters(f: np.ndarray, S: list[np.ndarray] | np.ndarray, 
                      dblim=[-80, 5], 
                    xunit="GHz", 
                    levelindicator: int | float =None, 
                    noise_floor=-150, 
                    fill_areas: list[tuple]= None, 
                    spec_area: list[tuple[float]] = None,
                    unwrap_phase=False, 
                    logx: bool = False,
                    labels: list[str] = None,
                    linestyles: list[str] = None,
                    colorcycle: list[int] = None,
                    filename: str = None,
                    show_plot: bool = True) -> None:
    """Plot S-parameters in dB and phase

    Args:
        f (np.ndarray): Frequency vector
        S (list[np.ndarray] | np.ndarray): S-parameters to plot (list or single array)
        dblim (list, optional): Decibel y-axis limit. Defaults to [-80, 5].
        xunit (str, optional): Frequency unit. Defaults to "GHz".
        levelindicator (int | float, optional): Level at which annotation arrows will be added. Defaults to None.
        noise_floor (int, optional): Artificial random noise floor level. Defaults to -150.
        fill_areas (list[tuple], optional): Regions to fill (fmin, fmax). Defaults to None.
        spec_area (list[tuple[float]], optional): _description_. Defaults to None.
        unwrap_phase (bool, optional): If or not to unwrap the phase data. Defaults to False.
        logx (bool, optional): Whether to use logarithmic frequency axes. Defaults to False.
        labels (list[str], optional): A lists of labels to use. Defaults to None.
        linestyles (list[str], optional): The linestyle to use (list or single string). Defaults to None.
        colorcycle (list[int], optional): A list of colors to use. Defaults to None.
        filename (str, optional): The filename (will automatically save). Defaults to None.
        show_plot (bool, optional): If or not to show the resulting plot. Defaults to True.
    """    
    if not isinstance(S, list):
        Ss = [S]
    else:
        Ss = S

    if linestyles is None:
        linestyles = ['-' for _ in S]

    if colorcycle is None:
        colorcycle = [i for i, S in enumerate(S)]

    unitdivider = {"MHz": 1e6, "GHz": 1e9, "kHz": 1e3}
    fnew = f / unitdivider[xunit]

    # Create two subplots: one for magnitude and one for phase
    fig, (ax_mag, ax_phase) = plt.subplots(2, 1, sharex=False, gridspec_kw={'height_ratios': [3, 1]})
    fig.subplots_adjust(hspace=0.3)

    minphase, maxphase = -180, 180

    maxy = 0
    for s, ls, cid in zip(Ss, linestyles, colorcycle):
        # Calculate and plot magnitude in dB
        SdB = 20 * np.log10(np.abs(s) + 10**(noise_floor/20) * np.random.rand(*s.shape) + 10**((noise_floor-30)/20))
        ax_mag.plot(fnew, SdB, label="Magnitude (dB)", linestyle=ls, color=_colors[cid % len(_colors)])
        if np.max(SdB) > maxy:
            maxy = np.max(SdB)
        # Calculate and plot phase in degrees
        phase = np.angle(s, deg=True)
        if unwrap_phase:
            phase = np.unwrap(phase, period=360)
            minphase = min(np.min(phase), minphase)
            maxphase = max(np.max(phase), maxphase)
        ax_phase.plot(fnew, phase, label="Phase (degrees)", linestyle=ls, color=_colors[cid % len(_colors)])

        # Annotate level indicators if specified
        if isinstance(levelindicator, (int, float)) and levelindicator is not None:
            lvl = levelindicator
            fcross = hintersections(fnew, SdB, lvl)
            for fs in fcross:
                ax_mag.annotate(
                    f"{str(fs)[:4]}{xunit}",
                    xy=(fs, lvl),
                    xytext=(fs + 0.08 * (max(f) - min(f)) / unitdivider[xunit], lvl),
                    arrowprops=dict(facecolor="black", width=1, headwidth=5),
                )
    if fill_areas is not None:
        for fmin, fmax in fill_areas:
            f1 = fmin / unitdivider[xunit]
            f2 = fmax / unitdivider[xunit]
            ax_mag.fill_between([f1, f2], dblim[0], dblim[1], color='grey', alpha= 0.2)
            ax_phase.fill_between([f1, f2], minphase, maxphase, color='grey', alpha= 0.2)
    if spec_area is not None:
        for fmin, fmax, vmin, vmax in spec_area:
            f1 = fmin / unitdivider[xunit]
            f2 = fmax / unitdivider[xunit]
            ax_mag.fill_between([f1, f2], vmin,vmax, color='red', alpha=0.2)
    # Configure magnitude plot (ax_mag)
    ax_mag.set_ylabel("Magnitude (dB)")
    ax_mag.set_xlabel(f"Frequency ({xunit})")
    ax_mag.axis([min(fnew), max(fnew), dblim[0], max(maxy*1.1,dblim[1])])
    ax_mag.axhline(y=0, color="k", linewidth=1)
    ax_mag.xaxis.set_minor_locator(tck.AutoMinorLocator(2))
    ax_mag.yaxis.set_minor_locator(tck.AutoMinorLocator(2))
    # Configure phase plot (ax_phase)
    ax_phase.set_ylabel("Phase (degrees)")
    ax_phase.set_xlabel(f"Frequency ({xunit})")
    ax_phase.axis([min(fnew), max(fnew), minphase, maxphase])
    ax_phase.xaxis.set_minor_locator(tck.AutoMinorLocator(2))
    ax_phase.yaxis.set_minor_locator(tck.AutoMinorLocator(2))
    if logx:
        ax_mag.set_xscale('log')
        ax_phase.set_xscale('log')
    if labels is not None:
        ax_mag.legend(labels)
        ax_phase.legend(labels)
    if show_plot:
        plt.show()
    if filename is not None:
        fig.savefig(filename)