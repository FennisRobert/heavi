import numpy as np
import matplotlib.pyplot as plt

def analyse_sparameter(sparam: list[np.ndarray], sparam_phase: list[np.ndarray], phase_reference: np.ndarray=None, n_bins: int = 51) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Analyse a set of S-parameter arrays, returning amplitude (dB) and phase statistics
    and plotting their histograms (PDF) and cumulative distributions (CDF), each on a
    separate y-axis.

    Parameters
    ----------
    sparam : list of np.ndarray
        Each element is a complex-valued S-parameter array (e.g. shape (F,)).
        We flatten them over frequency.
    *args, **kwargs : 
        Additional arguments (unused here, but provided for flexibility).

    Returns
    -------
    (amp_mean, amp_std), (phase_mean, phase_std) : tuple of tuples
        - amp_mean : float
            Mean of the magnitude in dB (flattened across frequency and Monte Carlo runs).
        - amp_std : float
            Standard deviation of the magnitude in dB.
        - phase_mean : float
            Mean of the phase error (degrees) after removing the mean complex phase.
        - phase_std : float
            Standard deviation of the phase error (degrees).
    """

    if sparam_phase is None:
        sparam_phase = sparam

    # ------------------
    # 1) Flatten S-parameters across all samples and frequencies
    # ------------------
    sparams_flat = np.concatenate(sparam)  # shape: (N*F,)
    
    # ------------------
    # 2) Compute amplitude in dB
    # ------------------
    magnitudes = np.abs(sparams_flat)
    magnitudes_db = 20.0 * np.log10(magnitudes + 1e-20)  # add small offset to avoid log(0)

    amp_mean = np.mean(magnitudes_db)
    amp_std = np.std(magnitudes_db)

    # ------------------
    # 3) Compute phase relative to average phase
    # ------------------
    # "Average complex S-parameter" means the mean of the complex samples.
    # NOTE: your code used: avg_sparam = sum(sparam)/len(sparam)
    # but that is effectively the average array. Then you took angles of ratio (S / avg_sparam).
    # The expression below just replicates that logic, but flattened. 
    # If you prefer the EXACT same logic as your snippet, do it the same way:
    if phase_reference is not None:
        phase_error_deg = 180 / np.pi * np.concatenate([
            np.angle(S / phase_reference) for S in sparam_phase
        ])
    else:
        phase_error_deg = 180 / np.pi * np.angle(np.concatenate(sparam_phase))
    
    phase_mean = np.mean(phase_error_deg)
    phase_std = np.std(phase_error_deg)

    # ------------------
    # 4) Plot results (PDF & CDF on separate y-axes)
    # ------------------
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    ax_amp, ax_phase = axes

    # Number of bins
    # === Amplitude Plot ===
    # PDF
    pdf_vals, bin_edges = np.histogram(magnitudes_db, bins=n_bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    # CDF (numerical integration of the PDF)
    cdf_vals = np.cumsum(pdf_vals) * np.diff(bin_edges)[0]
    
    # Create a twin axis for the CDF
    ax_amp_pdf = ax_amp
    ax_amp_cdf = ax_amp_pdf.twinx()

    pdf_line = ax_amp_pdf.plot(bin_centers, pdf_vals, 'b', label='PDF')
    cdf_line = ax_amp_cdf.plot(bin_centers, cdf_vals, 'r', label='CDF')

    ax_amp_pdf.set_title(f"Amplitude Distribution\nMean={amp_mean:.2f} dB, Std={amp_std:.2f} dB")
    ax_amp_pdf.set_xlabel("Amplitude [dB]")
    ax_amp_pdf.set_ylabel("PDF", color='b')
    ax_amp_cdf.set_ylabel("CDF", color='r')
    ax_amp_pdf.grid(True)
    ax_amp_cdf.grid(False)

    # Combine legends
    lines = pdf_line + cdf_line
    labels = [l.get_label() for l in lines]
    ax_amp_cdf.legend(lines, labels, loc='best')

    # === Phase Plot ===
    # PDF
    pdf_vals, bin_edges = np.histogram(phase_error_deg, bins=n_bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    # CDF
    cdf_vals = np.cumsum(pdf_vals) * np.diff(bin_edges)[0]

    # Create a twin axis for the CDF
    ax_phase_pdf = ax_phase
    ax_phase_cdf = ax_phase_pdf.twinx()

    pdf_line = ax_phase_pdf.plot(bin_centers, pdf_vals, 'b', label='PDF')
    cdf_line = ax_phase_cdf.plot(bin_centers, cdf_vals, 'r', label='CDF')

    ax_phase_pdf.set_title(f"Phase Error Distribution\nMean={phase_mean:.2f}°, Std={phase_std:.2f}°")
    ax_phase_pdf.set_xlabel("Phase Error [degrees]")
    ax_phase_pdf.set_ylabel("PDF", color='b')
    ax_phase_cdf.set_ylabel("CDF", color='r')
    ax_phase_pdf.grid(True)
    ax_phase_cdf.grid(False)

    # Combine legends
    lines = pdf_line + cdf_line
    labels = [l.get_label() for l in lines]
    ax_phase_cdf.legend(lines, labels, loc='best')

    plt.tight_layout()
    plt.show()

    # ------------------
    # 5) Return the stats
    # ------------------
    return (amp_mean, amp_std), (phase_mean, phase_std)


    