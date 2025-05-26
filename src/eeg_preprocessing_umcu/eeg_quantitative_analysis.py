import itertools
import logging
import math
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from multiprocessing import Pool, cpu_count
from typing import Dict, Optional, Tuple

import mne
import networkx as nx  # TODO: add to dependencies
import numpy as np
import pandas as pd
import psutil  # TODO: add to dependencies
from antropy import sample_entropy
from scipy import signal
from scipy.signal import hilbert
from scipy.sparse.csgraph import minimum_spanning_tree

PySimpleGUI_License = "ePycJVMLaeW5NflzbNn9NLlOVFHzl7w4ZaSLIk6MIYkvRWpncB3ORHyiauWyJB1gdyGQlPvXbgi7IAswIIk3xnpjYF2QVPuccn2aVNJdRhCnI96RMUTtcdzOMYzIM05eNLjWQjyxMQi8wGirTkGxl2jUZcWc5YzYZOUyRXllcQGExvvFeDWz1jlVbVnEROW3ZzXIJGzUalWp9huwImjmojivNhSr4Ew9ItiOwHikTXmpFxthZIUrZ8pfcCnFNQ0iIJjyokihWeWm95yfYymHVJu2I4iqwxi5T9mAFatLZaUkxphHcv3DQRi3OcipJZMBbN2ZR6lvbeWEEkiuL7CfJSDIbt2n1mwBY8WY555qIsj8oriYISiJwNiwQq3GVvzEdaGL9VtyZMXbJgJIRnCrI06II1jCQNxDNSj8YEyPICiNwyiaR1GVFw0VZOUeltz1cW3EVFlUZcC3IG6nIUj3ICwjM3jQQ6tWMUT5IAtLMADTUdi8L1ClJQERYEXfRelmR0XBhDwTa5XyJalIcXyoIX6aIvj3ICwSM3jcYrtcMYT2AFtgMMDTkNiNL4CtJKFIbDWmFQpNbUEFFckFZeHrJVlRcY3DM9isOmicJ258bD3qJjiKZVWp5Psab62lRPllb7WbFQAbcOHtJsv6dUGm9EueLXmv1ylIIliowFiAShVhBZBwZuGnRVyXZrXMNdzTI9j7osioM5T8QxztLBjCEoyPMvSI4XySMRzxk3umMeT5MeisfZQJ=c=02504c6fb7ca09721d288ae69f8237c96a99697e5b723e543938c4be810e2615f6fa037769c1edbd61ae40a244556b95fdfc2843df8e3807e955bc2c1d4be04c7022e2aa84c8eef696a9c6a61297e79cc4f465fb5e94513820c17814b2d35afadfa00653a9157afbad05ce088b890ca447c12c1df95d67e61ceed0b57d99ee7f26bfca445ad111393dab2dd1b6bee992510a1e973d0c6fae38f654816cc8de05ce7a79081d2029d636be38fb06ff7c68bfa0bdf080c7bb349a71ec74894e9f746bcbe58a67482485609109ec0a416582fc50f3500f55d5a021e7ea0ce4aafa6a207c77b80c2b48484e70314ef2b1a14970f110336f4c68eed12b49b4f3560b9e48eca892473d97b6ccb712cd086b0baa6aef3aa59be23f951a3476fbc5824402af301b988f410cf050f722fa3f2995ae68d4852645384eccec7841c10fe44b08102cc32a6d94a5854d0a148cecf8d25a51067db2e71842845dd715141ca15f1a5dd475bf4cba5afb23e794e77a53b89590ea0a37e638d46c73c869f4957c4a445d813a94167f3aaca7b58ce66ccb0c605e4820cc661c3d2ae832e41ee9fd46357fb40d26e103d4d747794f8548c27c363e096d495269740a6c08e5f936aec6c689a5a18694b24c37268c9c18760d063ad62b96d505b01074f81d7bb94d456c0d2bca0dd8b96b2246167bb1d0ce36a44a4ec051d22a72260ebbf910b375e511158"
import PySimpleGUI as sg  # noqa: E402

logger = logging.getLogger(__name__)

# Configuration
FOLDER_EXTENSION = "bdf"  # Change this to match your folder extension (e.g., 'bdf', 'edf', etc.)
MAX_MEMORY_PERCENT = 70  # Maximum memory usage percentage
MIN_WINDOW_SIZE = 100  # Minimum window size for spectral variability in ms

# Be careful, option to change frequency bands (both those recognized in the epoch file names
# and bands used for power and spectral variability calculations. Don't change the format. You can add additional
# bands in the same format.
FREQUENCY_BANDS = {
    "delta": {"pattern": r"0\.5-4\.0|delta", "range": (0.5, 4.0)},
    "theta": {"pattern": r"4\.0-8\.0|theta", "range": (4.0, 8.0)},
    "alpha": {"pattern": r"8\.0-13\.0|alpha", "range": (8.0, 13.0)},
    "beta1": {"pattern": r"13\.0-20\.0|beta1", "range": (13.0, 20.0)},
    "beta2": {"pattern": r"20\.0-30\.0|beta2", "range": (20.0, 30.0)},
    # Keep broadband (with this exact name) since this band is used for power and SV calculations.
    # It's fine if broadband refers to unfiltered epochs, power and SV calculations create a new PSD.
    "broadband": {"pattern": r"0\.5-47|broadband", "range": (0.5, 47.0)},
}


def validate_frequency_bands():
    """Validate FREQUENCY_BANDS configuration."""
    if not FREQUENCY_BANDS:
        msg = "FREQUENCY_BANDS dictionary is empty"
        raise ValueError(msg)

    for band_name, band_info in FREQUENCY_BANDS.items():
        if "pattern" not in band_info or "range" not in band_info:
            msg = f"Band {band_name} missing required keys (pattern, range)"
            raise ValueError(msg)

        fmin, fmax = band_info["range"]
        if not (isinstance(fmin, (int, float)) and isinstance(fmax, (int, float))):
            msg = f"Band {band_name} range values must be numeric"
            raise TypeError(msg)
        if fmin >= fmax:
            msg = f"Band {band_name} minimum frequency must be less than maximum"
            raise ValueError(msg)

        if not isinstance(band_info["pattern"], str):
            msg = f"Band {band_name} pattern must be a string"
            raise TypeError(msg)


BATCH_SIZE = 10  # Number of subjects to process in parallel
DEFAULT_THREADS = max(1, int(cpu_count() * 0.7))  # Use 80% of cores, no max limit


class MemoryMonitor:  # noqa: D101
    @staticmethod
    def get_memory_usage():
        """Get current memory usage percentage."""
        return psutil.Process().memory_percent()

    @staticmethod
    def check_memory():
        """Check if memory usage is too high."""
        if MemoryMonitor.get_memory_usage() > MAX_MEMORY_PERCENT:
            return True
        return False

    @staticmethod
    def check_concatenation_safety(data_size, num_epochs):
        """
        Check if concatenation is likely to exceed memory limits.

        Parameters
        ----------
        data_size : int
            Size of one epoch in bytes
        num_epochs : int
            Number of epochs to concatenate

        Returns
        -------
        bool : True if safe to proceed, False if likely to exceed memory
        """
        try:
            # Get system memory info
            system_memory = psutil.virtual_memory()
            available_memory = system_memory.available

            # Calculate estimated memory needed (add 20% buffer)
            estimated_memory = data_size * num_epochs * 1.2

            # Check if we'll exceed the threshold
            memory_threshold = (available_memory * MAX_MEMORY_PERCENT) / 100

            if estimated_memory > memory_threshold:
                logger.warning(
                    f"Concatenation may exceed memory limits. "
                    f"Estimated need: {estimated_memory / 1e9:.2f}GB, "
                    f"Available: {memory_threshold / 1e9:.2f}GB"
                )
                return False
            return True

        except Exception:
            logger.exception("Error checking memory for concatenation")
            return False


def setup_logging(folder_path):
    """Set up logging for the current analysis run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(folder_path, f"eeg_analysis_{timestamp}.log")

    # Clear any existing handlers
    logging.getLogger().handlers = []

    # Configure logging with both file and console output
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_filename, mode="w"), logging.StreamHandler()],
    )

    # Test the logging setup
    logger.info("Logging initialized")
    logger.info(f"Log file created at: {log_filename}")

    return log_filename


def create_gui():
    """Create the GUI layout for EEG analysis settings."""
    suggested_threads = DEFAULT_THREADS

    HEADER_BG = "#2C5784"
    HEADER_TEXT = "#FFFFFF"
    MAIN_BG = "#F0F2F6"
    BUTTON_COLOR = ("#FFFFFF", "#2C5784")

    sg.theme("Default1")
    sg.set_options(font=("Helvetica", 10))

    header = [
        [
            sg.Text(
                "EEG Quantitative Analysis",
                font=("Helvetica", 20, "bold"),
                text_color=HEADER_TEXT,
                background_color=HEADER_BG,
                pad=(10, 5),
            )
        ],
        [
            sg.Text(
                "Author: Yorben Lodema",
                font=("Helvetica", 10, "italic"),
                text_color=HEADER_TEXT,
                background_color=HEADER_BG,
                pad=(10, 5),
            )
        ],
    ]

    # Column 1: Input Settings and Matrix Export
    left_column = [
        [
            sg.Frame(
                "Input Settings",
                [
                    [sg.Text("Select data folder:", font=("Helvetica", 11, "bold"), background_color=MAIN_BG)],
                    [sg.Input(key="-FOLDER-", size=(25, 1)), sg.FolderBrowse(button_color=BUTTON_COLOR)],
                    [
                        sg.Text("Folder extension:", background_color=MAIN_BG),
                        sg.Input(FOLDER_EXTENSION, key="-EXTENSION-", size=(8, 1)),
                    ],
                    [
                        sg.Text("Processing threads:", background_color=MAIN_BG),
                        sg.Input(suggested_threads, key="-THREADS-", size=(5, 1)),
                    ],
                    [
                        sg.Checkbox(
                            "Epoch files have headers", key="-HAS_HEADERS-", default=True, background_color=MAIN_BG
                        )
                    ],
                ],
                background_color=MAIN_BG,
            )
        ],
        [
            sg.Frame(
                "Matrix Export",
                [
                    [
                        sg.Checkbox(
                            "Save connectivity matrices", key="-SAVE_MATRICES-", default=False, background_color=MAIN_BG
                        )
                    ],
                    [
                        sg.Text("Matrix folder:", background_color=MAIN_BG),
                        sg.Input("connectivity_matrices", key="-MATRIX_FOLDER-", size=(15, 1)),
                    ],
                    [sg.Checkbox("Save MST matrices", key="-SAVE_MST-", default=False, background_color=MAIN_BG)],
                    [
                        sg.Text("MST folder:", background_color=MAIN_BG),
                        sg.Input("mst_matrices", key="-MST_FOLDER-", size=(15, 1)),
                    ],
                    [
                        sg.Checkbox(
                            "Save channel-level averages",
                            key="-SAVE_CHANNEL_AVERAGES-",
                            default=False,
                            background_color=MAIN_BG,
                        )
                    ],
                ],
                background_color=MAIN_BG,
            )
        ],
    ]

    # Column 2: Complexity Measures
    middle_column = [
        [
            sg.Frame(
                "Complexity Measures",
                [
                    [sg.Checkbox("Calculate JPE/PE", key="-CALC_JPE-", default=False, background_color=MAIN_BG)],
                    [sg.Text("Time step (tau):", background_color=MAIN_BG), sg.Input("1", key="-JPE_ST-", size=(5, 1))],
                    [
                        sg.Checkbox(
                            "Convert to integers", key="-CONVERT_INTS_PE-", default=False, background_color=MAIN_BG
                        )
                    ],
                    [sg.Checkbox("Invert JPE (1-entropy)", key="-INVERT-", default=True, background_color=MAIN_BG)],
                    [
                        sg.Checkbox(
                            "Calculate Sample Entropy", key="-CALC_SAMPEN-", default=False, background_color=MAIN_BG
                        )
                    ],
                    [sg.Text("Order (m):", background_color=MAIN_BG), sg.Input("2", key="-SAMPEN_M-", size=(3, 1))],
                    [
                        sg.Checkbox(
                            "Calculate Approximate Entropy", key="-CALC_APEN-", default=False, background_color=MAIN_BG
                        )
                    ],
                    [sg.Text("Order (m):", background_color=MAIN_BG), sg.Input("1", key="-APEN_M-", size=(3, 1))],
                    [
                        sg.Text("Tolerance (r):", background_color=MAIN_BG),
                        sg.Input("0.25", key="-APEN_R-", size=(3, 1)),
                    ],
                ],
                background_color=MAIN_BG,
            )
        ],
    ]

    # Column 3: Spectral Analysis and Connectivity
    right_column = [
        [
            sg.Frame(
                "Spectral Analysis",
                [
                    [sg.Text("Sampling rate (Hz):", background_color=MAIN_BG), sg.Input(key="-POWER_FS-", size=(8, 1))],
                    [
                        sg.Text("PSD Method:", background_color=MAIN_BG),
                        sg.Combo(
                            ["Multitaper", "Welch", "FFT"], default_value="Multitaper", key="-PSD_METHOD-", size=(10, 1)
                        ),
                    ],
                    [
                        sg.Text(
                            "Welch parameters (only used if Welch method selected):",
                            background_color=MAIN_BG,
                            font=("Helvetica", 9, "italic"),
                        )
                    ],
                    [
                        sg.Text("Welch window (ms):", background_color=MAIN_BG),
                        sg.Input("1000", key="-WELCH_WINDOW-", size=(6, 1)),
                    ],
                    [
                        sg.Text("Welch overlap (%):", background_color=MAIN_BG),
                        sg.Input("50", key="-WELCH_OVERLAP-", size=(6, 1)),
                    ],
                    [sg.Checkbox("Calculate power bands", key="-CALC_POWER-", default=False, background_color=MAIN_BG)],
                    [
                        sg.Checkbox(
                            "Calculate peak frequency", key="-CALC_PEAK-", default=False, background_color=MAIN_BG
                        )
                    ],
                    [
                        sg.Text("Freq range:", background_color=MAIN_BG),
                        sg.Input("4", key="-PEAK_MIN-", size=(4, 1)),
                        sg.Text("-", background_color=MAIN_BG),
                        sg.Input("13", key="-PEAK_MAX-", size=(4, 1)),
                    ],
                    [
                        sg.Checkbox(
                            "Calculate spectral variability", key="-CALC_SV-", default=False, background_color=MAIN_BG
                        )
                    ],
                    [
                        sg.Text("Window (ms):", background_color=MAIN_BG),
                        sg.Input("2000", key="-SV_WINDOW-", size=(6, 1)),
                    ],
                ],
                background_color=MAIN_BG,
            )
        ],
        [
            sg.Frame(
                "Connectivity",
                [
                    [sg.Checkbox("Calculate PLI", key="-CALC_PLI-", default=False, background_color=MAIN_BG)],
                    [
                        sg.Checkbox(
                            "Calculate PLI MST measures", key="-CALC_PLI_MST-", default=False, background_color=MAIN_BG
                        )
                    ],
                    [sg.Checkbox("Calculate AEC", key="-CALC_AEC-", default=False, background_color=MAIN_BG)],
                    [
                        sg.Checkbox(
                            "Use orthogonalization (AECc)", key="-USE_AECC-", default=False, background_color=MAIN_BG
                        )
                    ],
                    [
                        sg.Checkbox(
                            "Concatenate epochs for AEC(c)",
                            key="-CONCAT_AECC-",
                            default=False,
                            background_color=MAIN_BG,
                        )
                    ],
                    [
                        sg.Checkbox(
                            "Calculate AEC(c) MST measures",
                            key="-CALC_AEC_MST-",
                            default=False,
                            background_color=MAIN_BG,
                        )
                    ],
                    [
                        sg.Checkbox(
                            "AEC make negative corr. zero",
                            key="-AEC_FORCE_POSITIVE-",
                            default=True,
                            background_color=MAIN_BG,
                        )
                    ],
                ],
                background_color=MAIN_BG,
            )
        ],
    ]

    progress_section = [
        [
            sg.Frame(
                "Progress",
                [
                    [
                        sg.Multiline(
                            size=(70, 6),
                            key="-LOG-",
                            autoscroll=True,
                            reroute_stdout=True,
                            disabled=True,
                            background_color="#FFFFFF",
                            text_color="#000000",
                        )
                    ],
                    [
                        sg.ProgressBar(
                            100, orientation="h", size=(60, 20), key="-PROGRESS-", bar_color=(HEADER_BG, MAIN_BG)
                        )
                    ],
                    [
                        sg.Column(
                            [
                                [
                                    sg.Button(
                                        "Process",
                                        size=(10, 1),
                                        button_color=BUTTON_COLOR,
                                        font=("Helvetica", 11, "bold"),
                                    ),
                                    sg.Button(
                                        "Exit",
                                        size=(8, 1),
                                        button_color=(HEADER_TEXT, "#AB4F4F"),
                                        font=("Helvetica", 11),
                                    ),
                                ]
                            ],
                            justification="center",
                            expand_x=True,
                            pad=(0, 5),
                            background_color=MAIN_BG,
                        )
                    ],
                ],
                background_color=MAIN_BG,
            )
        ],
    ]

    layout = [
        [sg.Column(header, background_color=HEADER_BG, expand_x=True)],
        [
            sg.Column(
                [
                    [
                        sg.Column(left_column, background_color=MAIN_BG, pad=(5, 5)),
                        sg.Column(middle_column, background_color=MAIN_BG, pad=(5, 5)),
                        sg.Column(right_column, background_color=MAIN_BG, pad=(5, 5)),
                    ],
                    [sg.Column(progress_section, background_color=MAIN_BG, pad=(0, 2))],
                ],
                background_color=MAIN_BG,
                pad=(5, 5),
            )
        ],
    ]

    window = sg.Window("EEG Analysis Tool", layout, background_color=MAIN_BG, finalize=True, margins=(0, 0))

    return window


def create_matrix_folder_structure(base_folder, matrix_folder_name, mst_folder_name=None):
    """Create folder structure with subject subfolders."""
    folders = {
        "jpe": os.path.join(base_folder, matrix_folder_name, "jpe"),
        "pli": os.path.join(base_folder, matrix_folder_name, "pli"),
        "aec": os.path.join(base_folder, matrix_folder_name, "aec"),
    }

    if mst_folder_name:
        folders.update(
            {
                "pli_mst": os.path.join(base_folder, mst_folder_name, "pli_mst"),
                "aec_mst": os.path.join(base_folder, mst_folder_name, "aec_mst"),
            }
        )

    # Create base folders
    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)

    return folders


def extract_freq_band(condition):
    """Parse the filename or condition string to identify frequency band.

    Based on the FREQUENCY_BANDS config.

    Parameters
    ----------
    condition : str
        The substring from the epoch filename (e.g., "8.0-13.0 Hz")
        or condition text that includes the frequency band.

    Returns
    -------
    str
        The band name (e.g., "alpha", "theta", "delta", etc.)
        or "unknown" if no match is found.
    """
    for band_name, band_info in FREQUENCY_BANDS.items():
        pattern = band_info["pattern"]
        # Add Hz to pattern if not already included
        if not pattern.endswith("Hz"):
            search_pattern = f"{pattern}(\s*Hz)?"
        else:
            search_pattern = pattern
        if re.search(search_pattern, condition, re.IGNORECASE):
            return band_name
    return "unknown"


def is_broadband_condition(condition):
    """
    Check if condition matches broadband pattern from FREQUENCY_BANDS config.

    Returns
    -------
    bool
        True if condition matches broadband pattern, False otherwise
    """
    if "broadband" not in FREQUENCY_BANDS:
        return False
    pattern = FREQUENCY_BANDS["broadband"]["pattern"]
    return bool(re.search(pattern, condition, re.IGNORECASE))


def save_connectivity_matrix(matrix, folder_path, subject, freq_band, feature, channel_names, level_type=None):
    """Save connectivity matrix to CSV with proper channel names."""
    # Create subject subfolder
    subject_folder = os.path.join(folder_path, subject)
    os.makedirs(subject_folder, exist_ok=True)

    # Include level type in filename for uniqueness
    filename = f"{level_type}_{freq_band}_{feature}.csv" if level_type else f"{freq_band}_{feature}.csv"
    filepath = os.path.join(subject_folder, filename)

    # Convert matrix to DataFrame with channel names
    df = pd.DataFrame(matrix)
    df.index = channel_names
    df.columns = channel_names

    df.to_csv(filepath)
    return filepath


def linear_detrend(data):
    """Apply linear detrending to each channel."""
    return signal.detrend(data, axis=0, type="linear")


def calculate_PSD(
    data: np.ndarray,
    fs: float,
    method: str = "multitaper",
    freq_range: Optional[Tuple[float, float]] = None,
    compute_spectrogram: bool = False,
    **kwargs,
) -> Dict[str, np.ndarray]:
    """
    Calculate Power Spectral Density (PSD) using specified method.

    Parameters
    ----------
    data : np.ndarray
        Time series data (samples x channels)
    fs : float
        Sampling frequency in Hz
    method : str
        Method to use for PSD calculation ('multitaper', 'welch', 'fft')
    freq_range : tuple, optional
        Frequency range to return (min_freq, max_freq)
    compute_spectrogram : bool
        Whether to compute and return spectrogram
    **kwargs : dict
        Method-specific parameters:
            Welch:
                window_length_ms : float (window length in milliseconds)
                overlap_percent : float (0 to 100)
            Multitaper:
                time_bandwidth : float (default 4)
                n_tapers : int (optional, computed from time_bandwidth)

    Returns
    -------
    dict
        Dictionary containing:
            'frequencies' : np.ndarray
                Frequency values
            'psd' : np.ndarray
                Power spectral density (frequencies x channels)
            'spectrogram' : np.ndarray, optional
                Time-frequency representation (only if compute_spectrogram=True)
    """
    if method not in ["multitaper", "welch", "fft"]:
        msg = f"Unknown method: {method}"
        raise ValueError(msg)

    # Input validation
    if not isinstance(data, np.ndarray):
        msg = "Data must be a numpy array"
        raise TypeError(msg)
    if data.ndim != 2:  # noqa: PLR2004
        msg = "Data must be 2D array (samples x channels)"
        raise ValueError(msg)
    if fs <= 0:
        msg = "Sampling frequency must be positive"
        raise ValueError(msg)

    # Initialize return dictionary
    result = {}

    # Calculate PSD based on method
    if method == "multitaper":
        try:
            frequencies, psd = _calculate_multitaper_psd(data, fs)
            result["frequencies"] = frequencies
            result["psd"] = psd

        except Exception:
            logger.exception("Error calculating multitaper PSD")
            raise

    elif method == "welch":
        try:
            window_length_ms = kwargs.get("window_length_ms", 1000)  # Default 1000ms
            overlap_percent = kwargs.get("overlap_percent", 50)  # Default 50%

            frequencies, psd = _calculate_welch_psd(
                data, fs, window_length_ms=window_length_ms, overlap_percent=overlap_percent
            )
            result["frequencies"] = frequencies
            result["psd"] = psd

        except Exception:
            logger.exception("Error calculating Welch PSD")
            raise

    elif method == "fft":
        try:
            frequencies, psd = _calculate_fft_psd(data, fs)
            result["frequencies"] = frequencies
            result["psd"] = psd

        except Exception:
            logger.exception("Error calculating FFT PSD")
            raise

    # Apply frequency range if specified
    if freq_range is not None:
        fmin, fmax = freq_range
        if not (0 <= fmin < fmax <= fs / 2):
            msg = f"Invalid frequency range: {freq_range}"
            raise ValueError(msg)

        freq_mask = (result["frequencies"] >= fmin) & (result["frequencies"] <= fmax)
        result["frequencies"] = result["frequencies"][freq_mask]
        result["psd"] = result["psd"][freq_mask]

    return result


def _calculate_welch_psd(
    data: np.ndarray, fs: float, window_length_ms: float = 1000, overlap_percent: float = 50
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate PSD using Welch's method.

    Parameters
    ----------
    data : np.ndarray
        Time series data (samples x channels)
    fs : float
        Sampling frequency in Hz
    window_length_ms : float
        Length of each segment in milliseconds
    overlap_percent : float
        Overlap between segments in percentage (0-100)

    Returns
    -------
    frequencies : np.ndarray
        Frequency values
    psd : np.ndarray
        Power spectral density (frequencies x channels)
    """
    # Convert window length from ms to samples
    nperseg = int((window_length_ms / 1000) * fs)

    # Convert overlap from percentage to samples
    noverlap = int(nperseg * (overlap_percent / 100))

    # Initialize array for PSD results
    n_channels = data.shape[1]

    # Calculate PSD for first channel to get frequency axis
    frequencies, temp_psd = signal.welch(
        data[:, 0], fs=fs, nperseg=nperseg, noverlap=noverlap, detrend=False, scaling="density"
    )

    # Initialize PSD array with correct dimensions
    psd = np.zeros((len(frequencies), n_channels))
    psd[:, 0] = temp_psd

    # Calculate for remaining channels
    for ch in range(1, n_channels):
        _, psd[:, ch] = signal.welch(
            data[:, ch],
            fs=fs,
            nperseg=nperseg,
            noverlap=noverlap,
            detrend=False,  # already done when loading data
            scaling="density",
        )

    return frequencies, psd


def _calculate_fft_psd(data: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    n_samples = data.shape[0]
    n_channels = data.shape[1]

    # Calculate frequency axis
    frequencies = np.fft.rfftfreq(n_samples, d=1 / fs)

    # Initialize PSD array
    psd = np.zeros((len(frequencies), n_channels))

    # Calculate PSD for each channel
    for ch in range(n_channels):
        # Apply Hanning window from scipy.signal.windows
        windowed_data = data[:, ch] * signal.windows.hann(n_samples)

        # Calculate FFT
        fft_data = np.fft.rfft(windowed_data)

        # Calculate power spectral density
        window_correction = np.mean(signal.windows.hann(n_samples) ** 2)
        psd[:, ch] = (np.abs(fft_data) ** 2) / (fs * n_samples * window_correction)

    return frequencies, psd


def _calculate_multitaper_psd(data: np.ndarray, fs: float):
    """Calculate PSD using MNE's multitaper implementation."""
    psds, freqs = mne.time_frequency.psd_array_multitaper(
        data.T,
        sfreq=fs,
        fmin=0,
        fmax=60,
        n_jobs=1,
        verbose=False,
    )

    # print(f"Multitaper PSD Frequency resolution: {freqs[1] - freqs[0]:.3f} Hz")  #noqa: ERA001

    return freqs, psds.T


def calculate_sampen_for_channels(data, m=2):
    """
    Calculate Sample Entropy for each channel using antropy.

    Parameters
    ----------
    data : numpy array (time points × channels)
    m : int
        Embedding dimension (order)

    Returns
    -------
    numpy.array : Sample Entropy values for each channel
    """
    n_channels = data.shape[1]
    sampen_values = np.zeros(n_channels)

    for ch in range(n_channels):
        try:
            sampen_values[ch] = sample_entropy(data[:, ch], order=m)

            if ch % 10 == 0:  # Log progress every 10 channels
                logger.info(f"Processed SampEn for {ch}/{n_channels} channels")

        except Exception:
            logger.exception(f"Error calculating SampEn for channel {ch}")
            sampen_values[ch] = np.nan

    return sampen_values


def calculate_apen_for_channels(data, m=1, r=0.25):
    """Calculate Approximate Entropy for each channel.

    Follows Pincus 1995, with optimized implementation using vectorization.

    Parameters
    ----------
    data : numpy array (time points * channels)
    m : int
        Embedding dimension (length of compared runs)
    r : float
        Tolerance (typically 0.25 * std of the data)

    Returns
    -------
    numpy.array : Approximate Entropy values for each channel
    """
    n_channels = data.shape[1]
    apen_values = np.zeros(n_channels)

    for ch in range(n_channels):
        try:
            # Get channel data
            x = data[:, ch]

            # Scale r by standard deviation of the data
            r_scaled = r * np.std(x)

            # Calculate phi(m) and phi(m+1)
            phi_m = _phi_vectorized(x, m, r_scaled)
            phi_m_plus_1 = _phi_vectorized(x, m + 1, r_scaled)

            # Calculate ApEn
            apen_values[ch] = phi_m - phi_m_plus_1

            if ch % 10 == 0:  # Log progress every 10 channels
                logger.info(f"Processed ApEn for {ch}/{n_channels} channels")

        except Exception:
            logger.exception(f"Error calculating ApEn for channel {ch}")
            apen_values[ch] = np.nan

    return apen_values


def _phi_vectorized(x, m, r):
    """
    Vectorized calculation of Φᵐ(r) following Pincus 1995.

    Parameters
    ----------
    x : array
        Time series data
    m : int
        Embedding dimension
    r : float
        Tolerance threshold

    Returns
    -------
    float : Φᵐ(r) value
    """
    N = len(x)
    N_m = N - m + 1

    # Create embedding matrix efficiently
    # Each row is a pattern of length m
    patterns = np.zeros((N_m, m))
    for i in range(m):
        patterns[:, i] = x[i : i + N_m]

    # Calculate distances using broadcasting
    # This computes the maximum absolute difference between all pairs of patterns
    diff = np.abs(patterns[:, None, :] - patterns[None, :, :])
    max_diff = np.max(diff, axis=2)

    # Count similar patterns (within tolerance r)
    # For each pattern, count how many other patterns are within distance r
    similar_patterns = np.sum(max_diff <= r, axis=1)

    # Normalize counts by N_m
    C = similar_patterns / N_m

    # Calculate Φᵐ(r) with small constant to avoid log(0)
    return np.mean(np.log(C + 1e-10))


def calculate_spectral_variability(data_values, fs, window_length=2000):
    """Calculate spectral variability per channel from concatenated broadband data.

    Uses FREQUENCY_BANDS for band definitions.

    - Expects pre-concatenated data with channel means already removed.
    - The "broadband" band is assumed to define total power reference.
    """
    try:
        num_samples, num_channels = data_values.shape
        samples_per_window = int(window_length * fs / 1000)

        # Require at least 3 windows for a meaningful coefficient of variation
        if num_samples < 3 * samples_per_window:
            logger.warning(
                f"Data length ({num_samples}) too short for meaningful "
                f"variability calculation with window length {samples_per_window} samples."
            )
            return None

        # 1) Identify the broadband range for total power
        if "broadband" not in FREQUENCY_BANDS:
            logger.exception("Broadband frequency range not defined in FREQUENCY_BANDS")
            return None
        broadband_min, broadband_max = FREQUENCY_BANDS["broadband"]["range"]

        # Prepare output dict of CV values
        cv_values = {}
        for band_name in FREQUENCY_BANDS:
            if band_name.lower() == "broadband":
                continue
            cv_values[band_name] = np.zeros(num_channels)

        # 2) Loop over channels and calculate spectrogram
        for channel in range(num_channels):
            try:
                # Compute spectrogram for this channel
                f, t, Sxx = signal.spectrogram(
                    data_values[:, channel],
                    fs=fs,
                    nperseg=samples_per_window,
                    noverlap=samples_per_window // 2,
                    detrend="constant",
                    window="hann",
                )

                # Create mask for broadband total power
                total_mask = (f >= broadband_min) & (f <= broadband_max)
                if not np.any(total_mask):
                    logger.exception("No frequencies found in the broadband range.")
                    for band_name in cv_values:
                        cv_values[band_name][channel] = np.nan
                    continue

                total_power = np.sum(Sxx[total_mask, :], axis=0)  # shape: (time_windows,)

                # 3) Loop over the user-defined frequency bands
                for band_name, band_info in FREQUENCY_BANDS.items():
                    if band_name.lower() == "broadband":
                        continue  # skip calculating a separate "broadband" measure

                    low_freq, high_freq = band_info["range"]
                    band_mask = (f >= low_freq) & (f < high_freq)
                    if not np.any(band_mask):
                        # If no frequencies found in this range, skip
                        cv_values[band_name][channel] = np.nan
                        continue

                    band_power = np.sum(Sxx[band_mask, :], axis=0)  # shape: (time_windows,)

                    # Compute relative power time series
                    with np.errstate(divide="ignore", invalid="ignore"):
                        relative_power = np.where(total_power > 0, band_power / total_power, 0)

                    # Remove NaN / Inf
                    valid_power = relative_power[np.isfinite(relative_power)]
                    if len(valid_power) > 0:
                        # Coefficient of Variation: std / mean
                        cv_values[band_name][channel] = np.std(valid_power) / np.mean(valid_power)
                    else:
                        cv_values[band_name][channel] = np.nan

            except Exception:
                logger.exception(f"Error processing channel {channel}")
                # Fill with NaN for all bands on this channel
                for band_name in cv_values:
                    cv_values[band_name][channel] = np.nan

        return cv_values

    except Exception:
        logger.exception("Error in spectral variability calculation")
        return None


def smooth_spectrum(frequencies, power_spectrum, smoothing_window=5):
    """Apply moving average smoothing to power spectrum."""
    return np.convolve(power_spectrum, np.ones(smoothing_window) / smoothing_window, mode="same")


def find_peaks(x, y, threshold_ratio=0.5):
    """Find significant peaks using relative maxima and prominence threshold."""
    peak_indices = signal.argrelextrema(y, np.greater)[0]
    prominences = signal.peak_prominences(y, peak_indices)[0]
    threshold = threshold_ratio * np.max(prominences)
    significant_peaks = peak_indices[prominences > threshold]

    return x[significant_peaks], y[significant_peaks]


def calculate_avg_peak_frequency(frequencies, psd, freq_range=(4, 13), smoothing_window=5):
    """
    Calculate peak frequency using pre-computed PSD with improved peak detection.

    Parameters
    ----------
    frequencies : numpy array
        Frequency values
    psd : numpy array
        Power spectral density (frequencies × channels)
    freq_range : tuple
        Frequency range to search for peaks (min_freq, max_freq)
    smoothing_window : int
        Window size for smoothing

    Returns
    -------
    numpy array: Peak frequencies for each channel
    """
    num_channels = psd.shape[1]
    peak_frequencies = np.zeros(num_channels)

    # Create frequency mask
    freq_mask = (frequencies >= freq_range[0]) & (frequencies <= freq_range[1])
    freq_range_idx = np.where(freq_mask)[0]

    if len(freq_range_idx) == 0:
        logger.warning(f"No frequencies found in range {freq_range[0]}-{freq_range[1]} Hz")
        return np.full(num_channels, np.nan)

    # Get masked frequencies and PSD
    frequencies_masked = frequencies[freq_mask]
    psd_masked = psd[freq_mask, :]

    for channel in range(num_channels):
        try:
            # Get channel-specific PSD
            channel_psd = psd_masked[:, channel]

            # Apply smoothing
            smoothed_psd = smooth_spectrum(frequencies_masked, channel_psd, smoothing_window)

            # Find all peaks
            peak_indices = signal.find_peaks(smoothed_psd)[0]

            if len(peak_indices) == 0:
                # No peaks found
                peak_frequencies[channel] = np.nan
                continue

            # Calculate peak properties
            peak_props = signal.peak_prominences(smoothed_psd, peak_indices)
            prominences = peak_props[0]

            # Sort peaks by prominence
            sorted_peak_indices = peak_indices[np.argsort(-prominences)]

            if len(sorted_peak_indices) > 0:
                # Get the frequency of the most prominent peak
                peak_frequencies[channel] = frequencies_masked[sorted_peak_indices[0]]
            else:
                peak_frequencies[channel] = np.nan

        except Exception:
            logger.exception(f"Error calculating peak frequency for channel {channel}")
            peak_frequencies[channel] = np.nan

    return peak_frequencies


def calculate_power_bands(frequencies, psd):
    """Calculate absolute and relative power.

    Uses pre-computed PSD for all defined frequency bands in FREQUENCY_BANDS, assuming
    'broadband' is always available in the dictionary for total power.

    Parameters
    ----------
    frequencies : np.ndarray
        1D array of frequency values.
    psd : np.ndarray
        2D array of power spectral density (frequencies x channels).

    Returns
    -------
    tuple
        (powers, channel_powers)

        - powers: dict with mean abs/rel power across channels per band
        - channel_powers: dict with channel-level arrays (one entry per band)
    """
    broadband_min, broadband_max = FREQUENCY_BANDS["broadband"]["range"]
    broadband_mask = (frequencies >= broadband_min) & (frequencies <= broadband_max)

    # Compute total power in the broadband range
    total_power = np.sum(psd[broadband_mask, :], axis=0)  # shape: (n_channels,)

    # Prepare output containers
    powers = {}
    channel_powers = {}

    # Loop over the frequency bands
    for band_name, band_info in FREQUENCY_BANDS.items():
        # Calculate for all bands including 'broadband'
        fmin, fmax = band_info["range"]
        band_mask = (frequencies >= fmin) & (frequencies <= fmax)

        # Compute absolute power in this band
        abs_power = np.sum(psd[band_mask, :], axis=0)

        # Compute relative power
        with np.errstate(divide="ignore", invalid="ignore"):
            # For broadband, relative power is 1.0 by definition
            if band_name.lower() == "broadband":
                rel_power = np.ones_like(abs_power)
            else:
                rel_power = abs_power / total_power

        # Store mean abs/rel power across channels
        powers[f"{band_name}_abs_power"] = np.nanmean(abs_power)
        powers[f"{band_name}_rel_power"] = np.nanmean(rel_power)

        # Also store channel-level arrays
        channel_powers[f"{band_name}_abs_power"] = np.nan_to_num(abs_power, nan=np.nan)
        channel_powers[f"{band_name}_rel_power"] = np.nan_to_num(rel_power, nan=np.nan)

    return powers, channel_powers


def calculate_mst_measures(connectivity_matrix, used_channels=None):
    """
    Calculate MST measures from a connectivity matrix with additional error handling for disconnected graphs.

    Args:
        connectivity_matrix (numpy.ndarray): Square connectivity matrix (e.g., PLI matrix)
        used_channels (numpy.ndarray, optional): Boolean array indicating which channels are used.
                                               If None, all channels are considered used.

    Returns
    -------
        tuple: (dict of MST measures, MST matrix, bool indicating success)
    """
    # Initialize used_channels if not provided
    if used_channels is None:
        used_channels = np.ones(len(connectivity_matrix), dtype=bool)

    # Get number of total channels (N) and used channels (M)
    n_total = len(connectivity_matrix)  # N in BrainWave
    n_used = np.sum(used_channels)  # M in BrainWave
    norm_factor = n_used - 1  # (M-1) for initial normalization

    # Create MST from connectivity matrix
    # Using -connectivity to get maximum spanning tree
    mst_matrix = minimum_spanning_tree(-connectivity_matrix).toarray()
    # Convert to NetworkX graph to check connectivity
    G = nx.from_numpy_array(-mst_matrix)

    # Check if the graph is connected
    if not nx.is_connected(G):
        return None, None, False

    # If connected, proceed with calculations
    # Negate back to get original weights
    mst_matrix = -mst_matrix
    G = nx.from_numpy_array(mst_matrix)

    measures = {}

    try:
        # 1. Maximum degree calculation
        degrees = defaultdict(float)
        for edge in G.edges():
            degrees[edge[0]] += 1.0 / norm_factor  # norm_factor is (M-1)
            degrees[edge[1]] += 1.0 / norm_factor

        measures["degree"] = max(degrees.values()) if degrees else 0

        # 2. Eccentricity - normalize by (M-1)
        eccentricity = nx.eccentricity(G)
        normalized_eccentricity = {node: ecc / norm_factor for node, ecc in eccentricity.items() if used_channels[node]}
        measures["eccentr"] = np.mean(list(normalized_eccentricity.values()))

        # 3. Betweenness centrality
        betweenness = nx.betweenness_centrality(G)
        measures["betweenness"] = max(list(betweenness.values()))

        # 4. Diameter - normalize by (M-1)
        raw_diameter = nx.diameter(G)
        measures["diameter"] = raw_diameter / norm_factor

        # 5. Leaf fraction
        leaf_nodes = sum(1 for node, deg in degrees.items() if abs(deg - 1.0 / norm_factor) < 1e-10)  # noqa: PLR2004
        measures["leaf"] = leaf_nodes / n_used

        max_betweenness = max(betweenness.values()) if betweenness else 0
        if max_betweenness > 0:
            measures["hierarchy"] = leaf_nodes / (2 * max_betweenness * norm_factor)
        else:
            measures["hierarchy"] = 0

        # 6. Kappa (degree divergence)
        sum_x = sum((n_total - 1) * deg for node, deg in degrees.items() if used_channels[node])
        sum_x2 = sum(((n_total - 1) * deg) ** 2 for node, deg in degrees.items() if used_channels[node])
        measures["kappa"] = sum_x2 / sum_x if sum_x > 0 else 0

        # 7. Tree hierarchy
        max_betweenness = max(betweenness.values()) if betweenness else 0
        if max_betweenness > 0:
            measures["hierarchy"] = leaf_nodes / (2 * max_betweenness * norm_factor)
        else:
            measures["hierarchy"] = 0

        # 8. Average shortest path (ASP)
        paths = dict(nx.all_pairs_shortest_path_length(G))
        sum_distances = 0
        for i in range(n_total):
            if used_channels[i]:
                node_distances = 0
                for j in range(n_total):
                    if used_channels[j] and i != j:
                        if i in paths and j in paths[i]:
                            node_distances += paths[i][j]
                sum_distances += node_distances

        measures["asp"] = sum_distances / (n_used * (n_used - 1)) if n_used > 1 else 0

        # 9. Tree efficiency (Teff)
        normalized_diam = raw_diameter / norm_factor
        measures["teff"] = 1.0 - (normalized_diam * (n_used - 1)) / (n_used - (n_used - 1) * measures["leaf"] + 1.0)

        # 10. R (degree correlation)
        degree_pairs = []
        for edge in G.edges():
            i, j = edge[0], edge[1]
            if used_channels[i] and used_channels[j]:
                degree_pairs.append((degrees[i], degrees[j]))

        if degree_pairs:
            deg_i, deg_j = zip(*degree_pairs)
            deg_i = np.array(deg_i)
            deg_j = np.array(deg_j)

            mean_i = np.mean(deg_i)
            mean_j = np.mean(deg_j)
            cov = np.mean((deg_i - mean_i) * (deg_j - mean_j))
            var_i = np.mean((deg_i - mean_i) ** 2)
            var_j = np.mean((deg_j - mean_j) ** 2)

            if var_i * var_j > 0:
                measures["r"] = cov / np.sqrt(var_i * var_j)
            else:
                measures["r"] = 0
        else:
            measures["r"] = 0

        # 11. Mean edge weight
        edge_weights = [abs(d.get("weight", 1.0)) for _, _, d in G.edges(data=True)]
        measures["mean"] = np.mean(edge_weights) if edge_weights else 0

        # 12. Reference value
        mst_sum = np.sum(abs(mst_matrix[used_channels][:, used_channels]))
        orig_sum = np.sum(connectivity_matrix[used_channels][:, used_channels])
        measures["ref"] = mst_sum / orig_sum if orig_sum > 0 else 0

        return measures, mst_matrix, True

    except Exception:
        logger.exception("Error in MST measures calculation")
        return None, None, False


def calculate_pli(data):
    """Optimized PLI calculation using vectorization."""
    analytic_signal = hilbert(data, axis=0)
    phases = np.angle(analytic_signal)

    n_channels = data.shape[1]
    pli = np.zeros((n_channels, n_channels))

    # Vectorized phase difference calculation
    for i in range(n_channels):
        phase_diffs = phases[:, i : i + 1] - phases[:, i:]
        signs = np.sign(np.sin(phase_diffs))
        means = np.abs(np.mean(signs, axis=0))
        pli[i, i:] = means
        pli[i:, i] = means

    return pli


def convert_to_integers(data):
    """Convert to integers using simple truncation."""
    return data.astype(int)


def calculate_aecc(data, orthogonalize=False, force_positive=True):
    """
    Calculate amplitude envelope correlation with optional orthogonalization.

    Parameters
    ----------
    data : numpy array (time points × channels)
        EEG data array
    orthogonalize : bool, optional
        Whether to perform orthogonalization
    force_positive : bool, optional
        Whether to force negative correlations to zero

    Returns
    -------
    numpy array (channels × channels)
        AEC(c) correlation matrix
    """

    def process_correlation(corr):
        """Process correlation based on force_positive setting."""
        return max(0.0, corr) if force_positive else corr

    n_channels = data.shape[1]
    correlation_matrix = np.zeros((n_channels, n_channels))

    if orthogonalize:
        # Process all channels pairwise
        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                # Orthogonalize in both directions
                d_orth_ij = data[:, j] - np.dot(data[:, j], data[:, i]) * data[:, i] / np.dot(data[:, i], data[:, i])
                d_orth_ji = data[:, i] - np.dot(data[:, i], data[:, j]) * data[:, j] / np.dot(data[:, j], data[:, j])

                # Calculate envelopes
                env_i = np.abs(hilbert(data[:, i]))
                env_j = np.abs(hilbert(data[:, j]))
                env_orth_ij = np.abs(hilbert(d_orth_ij))
                env_orth_ji = np.abs(hilbert(d_orth_ji))

                # Calculate correlations
                corr_ij = process_correlation(np.corrcoef(env_i, env_orth_ij)[0, 1])
                corr_ji = process_correlation(np.corrcoef(env_j, env_orth_ji)[0, 1])

                # Update correlation matrix
                correlation_matrix[i, j] = (corr_ij + corr_ji) / 2
                correlation_matrix[j, i] = correlation_matrix[i, j]

                # Clean up memory
                del env_i, env_j, env_orth_ij, env_orth_ji, d_orth_ij, d_orth_ji

    else:
        # Simple amplitude envelope correlation without orthogonalization
        envs = np.abs(hilbert(data, axis=0))
        correlation_matrix = np.corrcoef(envs.T)
        if force_positive:
            correlation_matrix = np.maximum(correlation_matrix, 0)

        del envs

    # Zero the diagonal
    np.fill_diagonal(correlation_matrix, 0)

    return correlation_matrix


def calculate_pe(data, n=4, st=1):
    """Calculate Permutation Entropy for each channel.

    Parameters
    ----------
        data : numpy array (time points × channels)
        n : int, embedding dimension
        st : int, time delay (should scale with sampling frequency)

    Returns
    -------
        numpy array : PE values for each channel
    """
    sz = data.shape[0]
    combinations = list(itertools.permutations(np.arange(0, n), n))

    PEs = []
    for ch in range(data.shape[1]):
        pattern_counts = np.zeros(len(combinations))

        # Step size for moving between patterns should be fixed (e.g., 1)
        # Only the sampling interval (st) within patterns should scale with frequency
        for i in range(0, sz - n * st, 1):
            dat_array = data[i : i + n * st : st, ch]
            if len(dat_array) < n:
                break
            dat_order = dat_array.argsort()
            rank = dat_order.argsort()
            pattern_idx = combinations.index(tuple(rank))
            pattern_counts[pattern_idx] += 1

        # Calculate PE
        total_patterns = np.sum(pattern_counts)
        if total_patterns > 0:
            prob = pattern_counts[pattern_counts > 0] / total_patterns
            entr = -np.sum(prob * np.log(prob))
            pe_norm = entr / np.log(math.factorial(n))
            PEs.append(pe_norm)
        else:
            PEs.append(np.nan)

    return np.array(PEs)


def find_mirror_patterns(combinations):
    """Create a lookup dictionary for mirror patterns (assumes 0-based ranks)."""
    if not combinations:
        return {}
    mirrors = {}
    n = len(combinations[0])  # Determine embedding dimension FROM the permutation length
    mirror_sum = n - 1  # Correct target sum for 0-based ranks

    for i, perm1 in enumerate(combinations):
        # Optimization: only need to check j > i
        for j in range(i + 1, len(combinations)):
            perm2 = combinations[j]
            # Check if perm2 is the mirror of perm1
            is_mirror = True
            for k in range(n):  # Iterate through elements of the permutations
                if perm1[k] + perm2[k] != mirror_sum:
                    is_mirror = False
                    break  # No need to check further elements
            if is_mirror:
                mirrors[i] = j
                mirrors[j] = i
    return mirrors


def is_volume_conduction(pattern1, pattern2, mirrors):
    """Check for volume conduction."""
    return pattern1 == pattern2 or pattern2 == mirrors.get(pattern1, -1)


def calculate_jpe(data, n=4, st=1, convert_ints=False, invert=True):
    """Calculate joint permutation entropy with corrected time delay handling.

    Parameters
    ----------
        data : numpy array (time points × channels)
        n : int, embedding dimension
        st : int, time delay (should scale with sampling frequency)
        convert_ints : bool, whether to convert data to integers
        invert : bool, whether to return 1-JPE
    """
    if convert_ints:
        data = convert_to_integers(data)

    data = np.asarray(data)
    sz = data.shape[0]
    combinations = list(itertools.permutations(np.arange(0, n), n))
    # mirrors = find_mirror_patterns(combinations, n-1) noqa: ERA001

    mirrors = find_mirror_patterns(combinations)

    # Modified to separate pattern step size from sampling interval
    rank_inds = []
    for i in range(0, sz - n * st, 1):  # Changed step size to 1
        dat_array = data[i : i + n * st : st, :]  # Keep st for within-pattern sampling
        if dat_array.shape[0] < n:  # Safety check
            break
        dat_order = dat_array.argsort(axis=0)
        rank = dat_order.argsort(axis=0)
        rank_inds.append([combinations.index(tuple(r)) for r in rank.T])

    rank_inds = np.array(rank_inds).T
    JPE = np.zeros((data.shape[1], data.shape[1]))

    for ch, x in enumerate(rank_inds):
        for ind, y in enumerate(rank_inds):
            if ind > ch:
                pattern_counts = 0
                jpe_mat = np.zeros((len(combinations), len(combinations)))

                for i, j in zip(x, y):
                    if not is_volume_conduction(i, j, mirrors):
                        pattern_counts += 1
                        jpe_mat[i, j] += 1

                if pattern_counts > 0:
                    jpe_mat = jpe_mat / pattern_counts
                    prob = jpe_mat[jpe_mat > 0]
                    entr = -np.sum(prob * np.log(prob))
                    jpe_norm = entr / np.log(math.factorial(n) * math.factorial(n) - 2 * math.factorial(n))
                    JPE[ch, ind] = 1 - jpe_norm if invert else jpe_norm

    JPE = JPE + JPE.T
    return JPE


def parse_epoch_filename(filename):
    """Parse epoch filename to extract components.

    Example: testjulia20231115kopie2_Source_level_4.0-8.0 Hz_Epoch_20.txt
    Alternative: 41_Source_level_broadband_Epoch1.txt.
    """
    # Extract the base name (everything before first underscore)
    base_name = filename.split("_")[0]

    # Extract level type (Source or Sensor)
    level_match = re.search(r"(Source|Sensor)_level", filename)
    level_type = level_match.group(1).lower() if level_match else "unknown"

    # Extract frequency band - try numerical range first
    freq_match = re.search(r"(\d+\.?\d*-\d+\.?\d*)\s*Hz", filename)
    if freq_match:
        freq_band = freq_match.group(1)
    else:
        # Try to extract broadband text
        parts = filename.split("_")
        for i, part in enumerate(parts):
            if part.lower() == "level" and i + 1 < len(parts):
                freq_band = parts[i + 1]
                break
        else:
            freq_band = "unknown"

    return {
        "base_name": base_name,
        "level_type": level_type,
        "freq_band": freq_band,
        "condition": f"{level_type}_{freq_band}",
    }


def process_subject_condition(args):
    """Process a single subject-condition combination."""
    (
        subject,
        condition,
        epoch_files,
        convert_ints_pe,
        invert,
        calc_jpe,
        calc_pli,
        calc_pli_mst,
        calc_aec,
        use_aecc,
        force_positive,
        jpe_st,
        calc_aec_mst,
        calc_power,
        power_fs,
        calc_peak,
        peak_min,
        peak_max,
        calc_sampen,
        sampen_m,
        calc_apen,
        apen_m,
        apen_r,
        calc_sv,
        sv_window,
        save_matrices,
        save_mst,
        save_channel_averages,
        concat_aecc,
        has_headers,
        psd_method,
        welch_window_ms,
        welch_overlap,
    ) = args

    try:
        channel_results = defaultdict(lambda: defaultdict(list))
        jpe_values = []
        pe_values = []
        pli_values = []
        aec_values = []
        pli_mst_values = defaultdict(list)
        aec_mst_values = defaultdict(list)
        power_values = defaultdict(list)
        apen_values = []
        sampen_values = []
        sv_values = {}
        channel_names = None

        # Initialize counter for successful MST calculations
        successful_mst_epochs = 0
        mst_calculation_attempted = False

        # Initialize matrices for connectivity
        avg_matrices = {
            "jpe": None,
            "pli": None,
            "aec": None,
        }

        logger.info(f"Processing {subject} - {condition} ({len(epoch_files)} epochs)")

        for i, file_path in enumerate(epoch_files):
            if MemoryMonitor.check_memory():
                logger.warning(f"High memory usage detected while processing {subject}")
                time.sleep(1)

            try:
                # Read data file
                if has_headers:
                    data = pd.read_csv(file_path, sep=None, engine="python")
                    if channel_names is None:
                        channel_names = data.columns.tolist()
                        logger.info(f"{subject} - {condition}: Found {len(channel_names)} channels from headers")
                else:
                    # First read the first row to check if it's non-numerical
                    first_row = pd.read_csv(file_path, sep=None, engine="python", header=None, nrows=1)
                    is_header = False

                    try:
                        first_row.astype(float)
                    except (ValueError, TypeError):
                        is_header = True
                        logger.info(f"Found non-numeric header in {os.path.basename(file_path)}, ignoring first row")

                    data = pd.read_csv(
                        file_path, sep=None, engine="python", header=None, skiprows=1 if is_header else 0
                    )

                    for col in data.columns:
                        data[col] = pd.to_numeric(data[col], errors="coerce")

                    if channel_names is None:
                        n_columns = len(data.columns)
                        channel_names = [f"Channel_{i + 1}" for i in range(n_columns)]
                        logger.info(f"{subject} - {condition}: Generated {len(channel_names)} channel names")
                    else:
                        current_columns = len(data.columns)
                        if current_columns != len(channel_names):
                            error_msg = (
                                f"Inconsistent number of columns in {os.path.basename(file_path)}. "
                                f"Expected {len(channel_names)}, found {current_columns}"
                            )
                            logger.exception(error_msg)
                            raise ValueError(error_msg)

                # Check for NaN values
                if data.isna().any().any():
                    logger.warning(
                        f"Found non-numeric values in {os.path.basename(file_path)} that were converted to NaN"
                    )

                data_values = data.values

                # Apply linear detrending to remove signal drift
                data_values = linear_detrend(data_values)

                del data

                # Determine if any spectral calculations are needed
                need_spectral = (calc_power or calc_peak or calc_sv) and is_broadband_condition(condition)

                if need_spectral:
                    try:
                        # Prepare PSD calculation parameters
                        psd_kwargs = {}
                        if psd_method == "welch":
                            psd_kwargs.update({"window_length_ms": welch_window_ms, "overlap_percent": welch_overlap})

                        # Calculate PSDs with appropriate settings
                        spectral_data = calculate_PSD(
                            data=data_values,
                            fs=power_fs,
                            method=psd_method,
                            compute_spectrogram=calc_sv,
                            window_length=sv_window if calc_sv else None,
                            overlap=0.5 if calc_sv else None,
                            **psd_kwargs,
                        )

                        if calc_power:
                            try:
                                logger.info(f"Starting power calculation for {subject} - {condition}, epoch {i + 1}")
                                powers, channel_powers = calculate_power_bands(
                                    frequencies=spectral_data["frequencies"], psd=spectral_data["psd"]
                                )

                                # Store whole-brain averages
                                for measure, value in powers.items():
                                    power_values[measure].append(value)
                                    logger.info(f"Power measure {measure}: {value}")

                                # Store channel-level results if requested
                                if save_channel_averages:
                                    for band_name in FREQUENCY_BANDS:
                                        for ch in range(len(channel_names)):
                                            channel_results[channel_names[ch]][f"{band_name}_abs_power"].append(
                                                channel_powers[f"{band_name}_abs_power"][ch]
                                            )
                                            channel_results[channel_names[ch]][f"{band_name}_rel_power"].append(
                                                channel_powers[f"{band_name}_rel_power"][ch]
                                            )

                            except Exception:
                                logger.exception(f"Error calculating power measures for epoch {i + 1}")
                                for band_name in FREQUENCY_BANDS:
                                    power_values[f"{band_name}_abs_power"].append(np.nan)
                                    power_values[f"{band_name}_rel_power"].append(np.nan)

                        # Calculate peak frequency if requested
                        if calc_peak:
                            try:
                                peak_freqs = calculate_avg_peak_frequency(
                                    frequencies=spectral_data["frequencies"],
                                    psd=spectral_data["psd"],
                                    freq_range=(peak_min, peak_max),
                                )

                                if save_channel_averages:
                                    for ch in range(len(channel_names)):
                                        channel_results[channel_names[ch]]["peak_frequency"].append(peak_freqs[ch])

                                n_channels_without_peak = np.sum(np.isnan(peak_freqs))
                                power_values["peak_frequency"].append(np.nanmean(peak_freqs))
                                power_values["channels_without_peak"].append(n_channels_without_peak)

                            except Exception:
                                logger.exception("Error calculating peak frequency")
                                power_values["peak_frequency"].append(np.nan)
                                power_values["channels_without_peak"].append(np.nan)

                        # Calculate spectral variability if requested
                        if calc_sv:
                            try:
                                # First load one epoch to check memory requirements
                                if epoch_files:
                                    try:
                                        test_data = pd.read_csv(
                                            epoch_files[0], sep=None, engine="python", header=0 if has_headers else None
                                        ).values
                                        if not MemoryMonitor.check_concatenation_safety(
                                            test_data.nbytes, len(epoch_files)
                                        ):
                                            msg = "Insufficient memory for safe concatenation of SV epochs"
                                            raise MemoryError(msg)
                                        del test_data
                                    except Exception:
                                        logger.exception("Error checking memory requirements for SV")
                                        raise

                                # Initialize list to store all epochs
                                all_data_sv = []
                                logger.info(f"Processing concatenated spectral variability for {subject} - {condition}")

                                # Read and store all epochs with offset correction
                                for file_path in epoch_files:
                                    try:
                                        if has_headers:
                                            data = pd.read_csv(file_path, sep=None, engine="python")
                                        else:
                                            data = pd.read_csv(file_path, sep=None, engine="python", header=None)

                                        epoch_data = data.values

                                        all_data_sv.append(epoch_data)
                                        del data, epoch_data
                                    except Exception:
                                        logger.exception(f"Error processing file {os.path.basename(file_path)} for SV")
                                        continue

                                if all_data_sv:
                                    try:
                                        # Concatenate along time axis
                                        concatenated_data = np.concatenate(all_data_sv, axis=0)
                                        # Clear the original list immediately
                                        all_data_sv = None

                                        if MemoryMonitor.check_memory():
                                            logger.warning("High memory usage detected after concatenation")
                                            time.sleep(1)

                                        # Calculate SV on concatenated data
                                        sv_results = calculate_spectral_variability(
                                            data_values=concatenated_data, fs=power_fs, window_length=sv_window
                                        )

                                        # Clear concatenated data immediately after use
                                        del concatenated_data

                                        if sv_results:
                                            if save_channel_averages:
                                                for band_name, values in sv_results.items():
                                                    band_key = f"sv_{band_name}"
                                                    for ch in range(len(channel_names)):
                                                        channel_results[channel_names[ch]][band_key] = values[ch]

                                            # Store single values for each band
                                            for band_name, values in sv_results.items():
                                                band_key = f"sv_{band_name}"
                                                sv_values[band_key] = np.nanmean(values)

                                    except Exception:
                                        logger.exception("Error calculating spectral variability on concatenated data")
                                        for band_name in FREQUENCY_BANDS:
                                            sv_values[f"sv_{band_name}"] = np.nan
                                else:
                                    logger.exception("No valid epochs could be processed for spectral variability")
                                    for band_name in FREQUENCY_BANDS:
                                        sv_values[f"sv_{band_name}"] = np.nan

                            except Exception:
                                logger.exception("Error in spectral variability processing")
                                for band_name in FREQUENCY_BANDS:
                                    sv_values[f"sv_{band_name}"] = np.nan

                            finally:
                                # Clean up interim data
                                if "all_data_sv" in locals():
                                    del all_data_sv

                        # Clean up spectral data
                        del spectral_data

                    except Exception:
                        logger.exception("Error in spectral calculations")
                        # Set all spectral measures to NaN
                        if calc_power:
                            for band_name in FREQUENCY_BANDS:
                                power_values[f"{band_name}_abs_power"].append(np.nan)
                                power_values[f"{band_name}_rel_power"].append(np.nan)
                        if calc_peak:
                            power_values["peak_frequency"].append(np.nan)
                            power_values["channels_without_peak"].append(np.nan)
                        if calc_sv:
                            for band_name in FREQUENCY_BANDS:
                                sv_values[f"sv_{band_name}"] = np.nan

                # Calculate JPE and PE
                if calc_jpe:
                    if convert_ints_pe:
                        data_values_pe = convert_to_integers(data_values)
                    else:
                        data_values_pe = data_values.copy()

                    jpe_matrix = calculate_jpe(data_values_pe, n=4, st=jpe_st, invert=invert)
                    if save_matrices:
                        if avg_matrices["jpe"] is None:
                            avg_matrices["jpe"] = jpe_matrix
                        else:
                            avg_matrices["jpe"] += jpe_matrix

                    mask = ~np.eye(jpe_matrix.shape[0], dtype=bool)
                    jpe_values.append(jpe_matrix[mask].mean())

                    pe_values_array = calculate_pe(data_values_pe, n=4, st=jpe_st)
                    pe_values.append(pe_values_array.mean())

                    if save_channel_averages:
                        for ch in range(len(channel_names)):
                            channel_results[channel_names[ch]]["pe"].append(pe_values_array[ch])
                            channel_jpe = np.mean(jpe_matrix, axis=1)
                            channel_results[channel_names[ch]]["jpe"].append(channel_jpe[ch])

                    if "data_values_pe" in locals():
                        del data_values_pe

                # Calculate PLI if requested
                if calc_pli:
                    try:
                        pli_matrix = calculate_pli(data_values)
                        if save_matrices or (save_mst and calc_pli_mst):
                            if avg_matrices["pli"] is None:
                                avg_matrices["pli"] = pli_matrix
                            else:
                                avg_matrices["pli"] += pli_matrix

                        mask = ~np.eye(pli_matrix.shape[0], dtype=bool)
                        pli_values.append(pli_matrix[mask].mean())

                        if save_channel_averages:
                            channel_pli = np.mean(pli_matrix, axis=1)
                            for ch in range(len(channel_names)):
                                channel_results[channel_names[ch]]["pli"].append(channel_pli[ch])

                        if calc_pli_mst:
                            try:
                                mst_measures, mst_matrix, success = calculate_mst_measures(pli_matrix)
                                if success:
                                    for measure, value in mst_measures.items():
                                        pli_mst_values[measure].append(value)
                            except Exception:
                                logger.exception(f"Error calculating PLI MST measures for epoch {i + 1}")
                                for measure in [
                                    "degree",
                                    "eccentr",
                                    "betweenness",
                                    "kappa",
                                    "r",
                                    "diameter",
                                    "leaf",
                                    "hierarchy",
                                    "teff",
                                    "asp",
                                    "ref",
                                    "mean",
                                ]:
                                    pli_mst_values[measure].append(np.nan)

                    except Exception:
                        logger.exception("Error calculating PLI")
                        pli_values.append(np.nan)

                # Calculate AEC/AECc if requested
                if calc_aec:
                    try:
                        if concat_aecc:
                            logger.info(
                                f"Processing concatenated AEC{'c' if use_aecc else ''} for {subject} - {condition}"
                            )
                            # Initialize list to store all epochs
                            all_data = []
                            data_values = None  # Initialize data_values in this scope

                            try:
                                # Read and store all epochs with offset correction
                                for file_path in epoch_files:
                                    try:
                                        data = pd.read_csv(file_path, sep=None, engine="python")
                                        epoch_data = data.values

                                        all_data.append(epoch_data)
                                        del data, epoch_data
                                    except Exception:
                                        logger.exception(f"Error processing file {os.path.basename(file_path)}")
                                        continue

                                if all_data:  # Check if we have any valid data
                                    # Concatenate along time axis
                                    data_values = np.concatenate(all_data, axis=0)
                                else:
                                    msg = "No valid epochs could be processed"
                                    raise ValueError(msg)

                            finally:
                                # Clean up interim data
                                del all_data

                            # Now use data_values for AEC calculation
                            aec_matrix = calculate_aecc(
                                data_values, orthogonalize=use_aecc, force_positive=force_positive
                            )

                            if save_matrices:
                                avg_matrices["aec"] = aec_matrix

                            mask = ~np.eye(aec_matrix.shape[0], dtype=bool)
                            aec_values = [aec_matrix[mask].mean()]  # Single value

                            # Calculate MST if requested - do it once and store results
                            mst_results = None
                            if calc_aec_mst:
                                try:
                                    logger.info(f"Calculating MST on concatenated AEC{'c' if use_aecc else ''} matrix")
                                    mst_measures, mst_matrix, success = calculate_mst_measures(aec_matrix)

                                    if success:
                                        successful_mst_epochs = 1
                                        mst_results = (mst_measures, mst_matrix, success)  # Store for later use

                                        # Store single values for MST measures
                                        for measure in [
                                            "degree",
                                            "eccentr",
                                            "betweenness",
                                            "kappa",
                                            "r",
                                            "diameter",
                                            "leaf",
                                            "hierarchy",
                                            "teff",
                                            "asp",
                                            "ref",
                                            "mean",
                                        ]:
                                            aec_mst_values[measure] = [mst_measures[measure]]

                                        # Store MST matrix if needed
                                        if save_mst:
                                            mst_matrix_symmetric = mst_matrix + mst_matrix.T
                                            avg_matrices["aec_mst"] = mst_matrix_symmetric
                                            logger.info("Stored MST matrix for concatenated data")
                                    else:
                                        successful_mst_epochs = 0
                                        logger.warning("Failed to calculate MST on concatenated matrix")
                                except Exception:
                                    successful_mst_epochs = 0
                                    logger.exception("Error calculating MST on concatenated matrix")

                            if save_channel_averages:
                                channel_aec = np.mean(aec_matrix, axis=1)
                                for ch in range(len(channel_names)):
                                    ch_name = channel_names[ch]
                                    # Initialize list if needed
                                    if "aec" not in channel_results[ch_name]:
                                        channel_results[ch_name]["aec"] = []
                                    # Always append to list (even for concatenated case)
                                    channel_results[ch_name]["aec"].append(channel_aec[ch])
                                logger.info("Added AEC channel data for concatenated processing")

                        else:
                            # Original epoch-by-epoch processing (keep existing code)
                            aec_matrix = calculate_aecc(
                                data_values, orthogonalize=use_aecc, force_positive=force_positive
                            )
                            if save_matrices or (save_mst and calc_aec_mst):
                                if avg_matrices["aec"] is None:
                                    avg_matrices["aec"] = aec_matrix
                                else:
                                    avg_matrices["aec"] += aec_matrix

                            mask = ~np.eye(aec_matrix.shape[0], dtype=bool)
                            aec_values.append(aec_matrix[mask].mean())

                            # Calculate MST if requested
                            if calc_aec_mst:
                                try:
                                    mst_measures, mst_matrix, success = calculate_mst_measures(aec_matrix)
                                    if success:
                                        for measure, value in mst_measures.items():
                                            aec_mst_values[measure].append(value)
                                        successful_mst_epochs += 1
                                except Exception:
                                    logger.exception("Error calculating MST measures for epoch")
                                    for measure in [
                                        "degree",
                                        "eccentr",
                                        "betweenness",
                                        "kappa",
                                        "r",
                                        "diameter",
                                        "leaf",
                                        "hierarchy",
                                        "teff",
                                        "asp",
                                        "ref",
                                        "mean",
                                    ]:
                                        aec_mst_values[measure].append(np.nan)

                            if save_channel_averages:
                                channel_aec = np.mean(aec_matrix, axis=1)
                                for ch in range(len(channel_names)):
                                    ch_name = channel_names[ch]
                                    # Initialize list if needed
                                    if "aec" not in channel_results[ch_name]:
                                        channel_results[ch_name]["aec"] = []
                                    # Always append to list
                                    channel_results[ch_name]["aec"].append(channel_aec[ch])

                    except Exception:
                        logger.exception(f"Error calculating AEC{'c' if use_aecc else ''}")
                        aec_values.append(np.nan)

                # Calculate complexity measures
                if calc_sampen:
                    try:
                        sampen_values_ch = calculate_sampen_for_channels(data_values, m=sampen_m)

                        if save_channel_averages:
                            for ch in range(len(channel_names)):
                                channel_results[channel_names[ch]]["sampen"].append(sampen_values_ch[ch])

                        sampen_values.append(np.nanmean(sampen_values_ch))  # Value per epoch
                        logger.info("Successfully calculated SampEn for epoch")
                    except Exception:
                        logger.exception("Error in SampEn calculation")
                        sampen_values.append(np.nan)

                if calc_apen:
                    try:
                        apen_values_ch = calculate_apen_for_channels(data_values, m=apen_m, r=apen_r)

                        if save_channel_averages:
                            for ch in range(len(channel_names)):
                                channel_results[channel_names[ch]]["apen"].append(apen_values_ch[ch])

                        apen_values.append(np.nanmean(apen_values_ch))  # Value per epoch
                        logger.info("Successfully calculated ApEn for epoch")
                    except Exception:
                        logger.exception("Error in ApEn calculation")
                        apen_values.append(np.nan)

            except Exception:
                logger.exception(f"Error processing file {os.path.basename(file_path)}")
                continue

        # Calculate channel averages across epochs if requested
        channel_averages = None
        if save_channel_averages:
            channel_averages = {}
            for channel in channel_names:
                channel_averages[channel] = {
                    measure: np.mean(values) for measure, values in channel_results[channel].items()
                }

        # First normalize averaged connectivity matrices
        n_epochs = len(epoch_files)
        for key in avg_matrices:
            if avg_matrices[key] is not None:
                # Only normalize if not using concatenation or if this isn't an AEC matrix
                if not (concat_aecc and key == "aec"):
                    avg_matrices[key] /= n_epochs
                    logger.info(f"Normalized {key} connectivity matrix by {n_epochs} epochs")

        # Now calculate MSTs from averaged connectivity matrices if needed
        if save_mst:
            # Handle PLI MST
            if calc_pli_mst and avg_matrices["pli"] is not None:
                try:
                    mst_measures, mst_matrix, success = calculate_mst_measures(avg_matrices["pli"])
                    if success:
                        mst_matrix_symmetric = mst_matrix + mst_matrix.T
                        avg_matrices["pli_mst"] = mst_matrix_symmetric
                        logger.info(f"Calculated PLI MST from averaged connectivity matrix for {subject}-{condition}")
                    else:
                        logger.warning(f"Could not calculate PLI MST from averaged matrix for {subject}-{condition}")
                except Exception:
                    logger.exception("Error calculating PLI MST from averaged matrix")

            # Handle AEC MST
            if calc_aec_mst and avg_matrices["aec"] is not None and not concat_aecc:
                try:
                    mst_measures, mst_matrix, success = calculate_mst_measures(avg_matrices["aec"])
                    if success:
                        mst_matrix_symmetric = mst_matrix + mst_matrix.T
                        avg_matrices["aec_mst"] = mst_matrix_symmetric
                        logger.info(f"Calculated AEC MST from averaged connectivity matrix for {subject}-{condition}")
                    else:
                        logger.warning(f"Could not calculate AEC MST from averaged matrix for {subject}-{condition}")
                except Exception:
                    logger.exception("Error calculating AEC MST from averaged matrix")

        # Prepare results dictionary
        results = {
            "avg_jpe": np.mean(jpe_values) if jpe_values and calc_jpe else np.nan,
            "avg_pe": np.mean(pe_values) if pe_values and calc_jpe else np.nan,
            "avg_pli": np.mean(pli_values) if pli_values and calc_pli else np.nan,
            "avg_aec": np.mean(aec_values) if aec_values and calc_aec else np.nan,
            "avg_sampen": np.mean(sampen_values) if sampen_values and calc_sampen else np.nan,
            "avg_apen": np.mean(apen_values) if apen_values and calc_apen else np.nan,
            "n_epochs": len(epoch_files),
            "channel_names": channel_names if channel_names else [],
            "matrices": avg_matrices if (save_matrices or save_mst) else None,
            "channel_averages": channel_averages,
        }

        # Add MST results and tracking metrics
        if calc_aec_mst:
            for measure in [
                "degree",
                "eccentr",
                "betweenness",
                "kappa",
                "r",
                "diameter",
                "leaf",
                "hierarchy",
                "teff",
                "asp",
                "ref",
                "mean",
            ]:
                if concat_aecc:
                    # For concatenated case, just store the single value
                    if aec_mst_values[measure]:
                        results[f"aec_mst_{measure}"] = aec_mst_values[measure][0]  # Single value
                        results[f"aec_mst_{measure}_valid_epochs"] = 1 if successful_mst_epochs else 0
                    else:
                        results[f"aec_mst_{measure}"] = np.nan
                        results[f"aec_mst_{measure}_valid_epochs"] = 0
                elif aec_mst_values[measure]:  # Only calculate mean if we have valid values
                    # Original epoch-by-epoch processing
                    results[f"aec_mst_{measure}"] = np.mean(aec_mst_values[measure])
                    results[f"aec_mst_{measure}_valid_epochs"] = len(aec_mst_values[measure])
                else:
                    results[f"aec_mst_{measure}"] = np.nan
                    results[f"aec_mst_{measure}_valid_epochs"] = 0

            # Add the tracking metrics
            results["aec_mst_successful_epochs"] = successful_mst_epochs
            results["aec_mst_total_epochs"] = 1 if concat_aecc else len(epoch_files)

            # Add concatenation info to results for reference
            results["aec_concatenated"] = concat_aecc

        if calc_pli_mst:
            for measure in [
                "degree",
                "eccentr",
                "betweenness",
                "kappa",
                "r",
                "diameter",
                "leaf",
                "hierarchy",
                "teff",
                "asp",
                "ref",
                "mean",
            ]:
                if pli_mst_values[measure]:
                    results[f"pli_mst_{measure}"] = np.mean(pli_mst_values[measure])
                    results[f"pli_mst_{measure}_valid_epochs"] = len(pli_mst_values[measure])
                else:
                    results[f"pli_mst_{measure}"] = np.nan
                    results[f"pli_mst_{measure}_valid_epochs"] = 0

        if calc_power and power_values:
            for band_name in FREQUENCY_BANDS:
                results[f"{band_name}_abs_power"] = np.mean(power_values[f"{band_name}_abs_power"])
                results[f"{band_name}_rel_power"] = np.mean(power_values[f"{band_name}_rel_power"])

        if calc_peak:
            results["peak_frequency"] = np.mean(power_values["peak_frequency"])
            results["channels_without_peak"] = np.mean(power_values["channels_without_peak"])

        if calc_sv:
            for band_name in FREQUENCY_BANDS:
                band_key = f"sv_{band_name}"
                # No averaging needed since we now have single values
                results[band_key] = sv_values.get(band_key, np.nan)

        return subject, condition, results

    except Exception:
        logger.exception(f"Error processing {subject} - {condition}")

        # Define error_result dictionary
        error_result = {
            "avg_jpe": np.nan,
            "avg_pe": np.nan,
            "avg_pli": np.nan,
            "avg_aec": np.nan,
            "avg_apen": np.nan,
            "avg_sampen": np.nan,
            "n_epochs": 0,
            "channel_names": [],
            "matrices": None if save_matrices else None,
            "channel_averages": None,
        }

        # Add MST measures to error result if needed
        if calc_aec_mst:
            for measure in [
                "degree",
                "eccentr",
                "betweenness",
                "kappa",
                "r",
                "diameter",
                "leaf",
                "hierarchy",
                "teff",
                "asp",
                "ref",
                "mean",
            ]:
                error_result[f"aec_mst_{measure}"] = np.nan
                error_result[f"aec_mst_{measure}_valid_epochs"] = 0
            error_result["aec_mst_successful_epochs"] = 0
            error_result["aec_mst_total_epochs"] = len(epoch_files)

        if calc_pli_mst:
            for measure in [
                "degree",
                "eccentr",
                "betweenness",
                "kappa",
                "r",
                "diameter",
                "leaf",
                "hierarchy",
                "teff",
                "asp",
                "ref",
                "mean",
            ]:
                error_result[f"pli_mst_{measure}"] = np.nan
                error_result[f"pli_mst_{measure}_valid_epochs"] = 0

        if calc_power:
            for band_name in FREQUENCY_BANDS:
                error_result[f"{band_name}_abs_power"] = np.nan
                error_result[f"{band_name}_rel_power"] = np.nan
        if calc_peak:
            error_result["peak_frequency"] = np.nan
            error_result["channels_without_peak"] = np.nan

        # Add spectral variability measures to error result if needed
        if calc_sv:
            for band_name in FREQUENCY_BANDS:
                error_result[f"sv_{band_name}"] = np.nan

        return subject, condition, error_result


def process_batch(batch_args, n_threads):
    """Process a batch of subjects using multiprocessing with fallback."""
    try:
        with Pool(processes=n_threads, maxtasksperchild=1) as pool:
            results = []
            for result in pool.imap_unordered(process_subject_condition, batch_args):
                results.append(result)
            return results
    except Exception:
        logger.exception("Pool processing failed, falling back to single thread")
        return [process_subject_condition(args) for args in batch_args]
    finally:
        if "pool" in locals():
            pool.terminate()
            pool.join()


def group_epochs_by_condition(folder_path, folder_ext):
    """Group epoch files by their base name and condition.

    Only processes folders containing valid epoch files
    Returns a dictionary: {base_name: {condition: [epoch_files]}}.
    """
    grouped_files = defaultdict(lambda: defaultdict(list))

    # Get immediate subdirectories
    try:
        subdirs = [
            d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d)) and d.endswith(folder_ext)
        ]
    except Exception:
        sg.popup_error("Error accessing directory")
        return grouped_files

    if not subdirs:
        sg.popup_error(f"No folders ending with '{folder_ext}' found in the selected directory.")
        return grouped_files

    for subdir in subdirs:
        subdir_path = os.path.join(folder_path, subdir)

        # Look for epoch files in this directory
        for file in os.listdir(subdir_path):
            if file.startswith(".") or file.startswith("._"):
                continue

            if not file.endswith(".txt"):
                continue

            # Check if file matches epoch pattern
            if "_level_" in file and ("_Epoch_" in file or "_Epoch" in file):
                try:
                    file_info = parse_epoch_filename(file)
                    full_path = os.path.join(subdir_path, file)
                    base_name = subdir.replace(folder_ext, "")  # Use folder name without extension
                    condition = file_info["condition"]
                    grouped_files[base_name][condition].append(full_path)
                except Exception:
                    print(f"Skipping file {file}")
                    continue

    # Print summary of what was found
    print("\nFound the following data:")
    for base_name, conditions in grouped_files.items():
        print(f"\nSubject: {base_name}")
        for condition, files in conditions.items():
            print(f"  {condition}: {len(files)} epochs")

    found_bands = set()
    unknown_conditions = set()
    has_broadband = False

    for conditions in grouped_files.values():
        for condition in conditions.keys():
            band = extract_freq_band(condition)
            if band != "unknown":
                found_bands.add(band)
                if band == "broadband":
                    has_broadband = True
            else:
                unknown_conditions.add(condition)

    logger.info("Found the following frequency bands in the data:")
    for band in sorted(found_bands):
        logger.info(f"  - {band} ({FREQUENCY_BANDS[band]['pattern']})")

    if has_broadband:
        logger.info("Broadband epochs are present - spectral calculations will be performed on these epochs")
    else:
        logger.warning("No broadband epochs found - spectral calculations will be skipped")

    if unknown_conditions:
        logger.warning("Found conditions with unrecognized frequency bands:")
        for cond in sorted(unknown_conditions):
            logger.warning(f"  - {cond}")

    return grouped_files


def process_all_subjects(  # noqa: D103
    grouped_files,
    convert_ints_pe,
    invert,
    n_threads,
    calc_jpe,
    calc_pli,
    calc_pli_mst,
    calc_aec,
    use_aecc,
    force_positive=True,
    jpe_st=1,
    calc_aec_mst=False,
    calc_power=False,
    power_fs=256,
    calc_peak=False,
    peak_min=3,
    peak_max=13,
    calc_sampen=False,
    sampen_m=2,
    calc_apen=False,
    apen_m=1,
    apen_r=0.25,
    calc_sv=False,
    sv_window=1000,
    save_matrices=False,
    save_mst=False,
    save_channel_averages=False,
    concat_aecc=False,
    has_headers=True,
    psd_method="multitaper",
    welch_window_ms=1000,
    welch_overlap=50,
    progress_callback=None,
):
    process_args = []
    for subject, conditions in grouped_files.items():
        for condition, epoch_files in conditions.items():
            process_args.append(
                (
                    subject,
                    condition,
                    epoch_files,
                    convert_ints_pe,
                    invert,
                    calc_jpe,
                    calc_pli,
                    calc_pli_mst,
                    calc_aec,
                    use_aecc,
                    force_positive,
                    jpe_st,
                    calc_aec_mst,
                    calc_power,
                    power_fs,
                    calc_peak,
                    peak_min,
                    peak_max,
                    calc_sampen,
                    sampen_m,
                    calc_apen,
                    apen_m,
                    apen_r,
                    calc_sv,
                    sv_window,
                    save_matrices,
                    save_mst,
                    save_channel_averages,
                    concat_aecc,
                    has_headers,
                    psd_method,
                    welch_window_ms,
                    welch_overlap,
                )
            )

    total_tasks = len(process_args)
    logger.info(f"Starting processing of {total_tasks} subject-condition combinations")
    logger.info(
        f"Processing options: JPE/PE calculation={calc_jpe}, JPE invert={invert}, "
        f"PE integer conversion={convert_ints_pe}, PLI calculation={calc_pli}, "
        f"PLI MST calculation={calc_pli_mst}, AEC calculation={calc_aec}, "
        f"AECc={use_aecc}, AEC concatenation={concat_aecc}, "
        f"Force positive AEC={force_positive}, Save MST matrices={save_mst}"
    )

    # Initialize results
    results = defaultdict(dict)
    completed = 0

    # Process in batches
    for i in range(0, len(process_args), BATCH_SIZE):
        batch = process_args[i : i + BATCH_SIZE]
        batch_size = len(batch)

        logger.info(
            f"Processing batch {i // BATCH_SIZE + 1} of {(total_tasks + BATCH_SIZE - 1) // BATCH_SIZE} "
            f"({batch_size} combinations)"
        )
        start_time = time.time()

        try:
            # Process batch
            batch_results = process_batch(batch, n_threads)

            # Update results and log MST statistics
            for subject, condition, result in batch_results:
                results[subject][condition] = result
                completed += 1

                # Log MST success rates if applicable
                if calc_aec_mst and "aec_mst_successful_epochs" in result:
                    success_rate = (
                        result["aec_mst_successful_epochs"] / result["aec_mst_total_epochs"] * 100
                        if result["aec_mst_total_epochs"] > 0
                        else 0
                    )
                    logger.info(
                        f"{subject} - {condition}: MST success rate: "
                        f"{success_rate:.1f}% ({result['aec_mst_successful_epochs']}/"
                        f"{result['aec_mst_total_epochs']} epochs)"
                    )

                if progress_callback:
                    progress_callback(completed / total_tasks * 100)

            batch_time = time.time() - start_time
            avg_time_per_combo = batch_time / batch_size
            remaining_time = (total_tasks - completed) * avg_time_per_combo

            logger.info(
                f"Batch completed in {batch_time:.1f} seconds "
                f"({avg_time_per_combo:.1f} sec/combination). "
                f"Estimated remaining time: {remaining_time / 60:.1f} minutes"
            )

        except Exception:
            logger.exception("Error processing batch")
            continue

        # Memory check and cleanup
        if MemoryMonitor.check_memory():
            logger.warning("High memory usage detected - triggering garbage collection")
            import gc

            gc.collect()
            time.sleep(1)

    # Final processing summary
    logger.info(f"Processing completed: {completed}/{total_tasks} combinations processed")
    if completed < total_tasks:
        logger.warning(f"Some combinations ({total_tasks - completed}) failed to process")

    return dict(results)


def save_results_to_excel(
    results_dict,
    output_path,
    invert,
    calc_pli_mst,
    calc_jpe=True,
    calc_pli=True,
    calc_aec=False,
    use_aecc=False,
    force_positive=True,
    calc_aec_mst=False,
    calc_power=False,
    power_fs=256,
    calc_peak=False,
    peak_min=None,
    peak_max=None,
    calc_sampen=False,
    calc_apen=False,
    calc_sv=False,
    save_channel_averages=False,
    concat_aecc=False,
    has_headers=True,
    sv_window=None,
    psd_method="multitaper",
    welch_window_ms=None,
    welch_overlap=None,
):
    """
    Save results to Excel with organized columns by condition.

    Keeps original logic:
      - Some features only generated if broadband epochs exist.
      - Broadband entropy/JPE measures are allowed, but separate
        broadband power columns are skipped.
      - Dynamically uses FREQUENCY_BANDS for non-broadband frequency bands.
    """
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        # 1) Gather all unique conditions
        all_conditions = set()
        for subject_data in results_dict.values():
            all_conditions.update(subject_data.keys())

        # 2) Build a list of primary columns; start with 'subject'
        columns = ["subject"]

        for condition in sorted(all_conditions):
            # --- Complexity Measures (JPE/PE) ---
            if calc_jpe:
                # e.g. "myCondition_avg_jpe", "myCondition_avg_pe"
                measure_name = "jpe_inv" if invert else "jpe"
                columns.append(f"{condition}_avg_{measure_name}")
                columns.append(f"{condition}_avg_pe")

            # --- PLI ---
            if calc_pli:
                columns.append(f"{condition}_avg_pli")

            # If you also have MST columns for PLI:
            if calc_pli_mst:
                # Example MST measures
                mst_measures = [
                    "degree",
                    "eccentr",
                    "betweenness",
                    "kappa",
                    "r",
                    "diameter",
                    "leaf",
                    "hierarchy",
                    "teff",
                    "asp",
                    "ref",
                    "mean",
                ]
                for mm in mst_measures:
                    columns.append(f"{condition}_pli_mst_{mm}")
                # Possibly also track valid-epoch counters
                columns.append(f"{condition}_pli_mst_successful_epochs")
                columns.append(f"{condition}_pli_mst_total_epochs")

            # --- AEC ---
            if calc_aec:
                columns.append(f"{condition}_avg_aec")
                if calc_aec_mst:
                    # Add MST columns for AEC if needed
                    mst_measures = [
                        "degree",
                        "eccentr",
                        "betweenness",
                        "kappa",
                        "r",
                        "diameter",
                        "leaf",
                        "hierarchy",
                        "teff",
                        "asp",
                        "ref",
                        "mean",
                    ]
                    for mm in mst_measures:
                        columns.append(f"{condition}_aec_mst_{mm}")
                    columns.append(f"{condition}_aec_mst_successful_epochs")
                    columns.append(f"{condition}_aec_mst_total_epochs")

            # --- SampEn & ApEn ---
            if calc_sampen:
                columns.append(f"{condition}_avg_sampen")
            if calc_apen:
                columns.append(f"{condition}_avg_apen")

            # Power & Peak Frequency ---
            def is_broadband_condition(condition):
                """Check if condition matches broadband pattern from FREQUENCY_BANDS."""
                if "broadband" not in FREQUENCY_BANDS:
                    return False
                pattern = FREQUENCY_BANDS["broadband"]["pattern"]
                return bool(re.search(pattern, condition, re.IGNORECASE))

            is_broadband_cond = is_broadband_condition(condition)

            # Power band measures - only for broadband conditions
            if calc_power and is_broadband_cond:
                for band_name in FREQUENCY_BANDS:
                    if band_name.lower() != "broadband":  # Skip broadband
                        columns.extend([f"{condition}_{band_name}_abs_power", f"{condition}_{band_name}_rel_power"])

            # Peak frequency - can be calculated independently
            if calc_peak and is_broadband_cond:
                columns.append(f"{condition}_peak_frequency")
                columns.append(f"{condition}_channels_without_peak")

            # --- Spectral Variability ---
            if calc_sv and is_broadband_cond:
                for band_name in FREQUENCY_BANDS:
                    if band_name.lower() == "broadband":
                        continue
                    columns.append(f"{condition}_sv_{band_name}")

            # Always add epoch count for each condition
            columns.append(f"{condition}_n_epochs")

        # 3) Build rows of data
        rows = []
        for subject, conditions in results_dict.items():
            row = {"subject": subject}

            for condition in sorted(all_conditions):
                data_for_condition = conditions[condition]

                # JPE/PE measures
                if calc_jpe:
                    measure_name = "jpe_inv" if invert else "jpe"
                    row[f"{condition}_avg_{measure_name}"] = data_for_condition.get("avg_jpe", np.nan)
                    row[f"{condition}_avg_pe"] = data_for_condition.get("avg_pe", np.nan)

                # PLI
                if calc_pli:
                    row[f"{condition}_avg_pli"] = data_for_condition.get("avg_pli", np.nan)

                # PLI MST
                if calc_pli_mst:
                    mst_measures = [
                        "degree",
                        "eccentr",
                        "betweenness",
                        "kappa",
                        "r",
                        "diameter",
                        "leaf",
                        "hierarchy",
                        "teff",
                        "asp",
                        "ref",
                        "mean",
                    ]
                    for mm in mst_measures:
                        row[f"{condition}_pli_mst_{mm}"] = data_for_condition.get(f"pli_mst_{mm}", np.nan)
                    row[f"{condition}_pli_mst_successful_epochs"] = data_for_condition.get(
                        "pli_mst_degree_valid_epochs", np.nan
                    )
                    row[f"{condition}_pli_mst_total_epochs"] = data_for_condition.get("pli_mst_total_epochs", np.nan)

                # AEC
                if calc_aec:
                    row[f"{condition}_avg_aec"] = data_for_condition.get("avg_aec", np.nan)
                    if calc_aec_mst:
                        mst_measures = [
                            "degree",
                            "eccentr",
                            "betweenness",
                            "kappa",
                            "r",
                            "diameter",
                            "leaf",
                            "hierarchy",
                            "teff",
                            "asp",
                            "ref",
                            "mean",
                        ]
                        for mm in mst_measures:
                            row[f"{condition}_aec_mst_{mm}"] = data_for_condition.get(f"aec_mst_{mm}", np.nan)
                        row[f"{condition}_aec_mst_successful_epochs"] = data_for_condition.get(
                            "aec_mst_successful_epochs", np.nan
                        )
                        row[f"{condition}_aec_mst_total_epochs"] = data_for_condition.get(
                            "aec_mst_total_epochs", np.nan
                        )

                # SampEn & ApEn
                if calc_sampen:
                    row[f"{condition}_avg_sampen"] = data_for_condition.get("avg_sampen", np.nan)
                if calc_apen:
                    row[f"{condition}_avg_apen"] = data_for_condition.get("avg_apen", np.nan)

                # Power & Peak Frequency
                is_broadband_cond = is_broadband_condition(condition)

                # Power band measures
                if calc_power and is_broadband_cond:
                    for band_name in FREQUENCY_BANDS:
                        if band_name.lower() == "broadband":
                            continue
                        abs_key = f"{band_name}_abs_power"
                        rel_key = f"{band_name}_rel_power"
                        row[f"{condition}_{band_name}_abs_power"] = data_for_condition.get(abs_key, np.nan)
                        row[f"{condition}_{band_name}_rel_power"] = data_for_condition.get(rel_key, np.nan)

                # Peak frequency
                if calc_peak and is_broadband_cond:
                    row[f"{condition}_peak_frequency"] = data_for_condition.get("peak_frequency", np.nan)
                    row[f"{condition}_channels_without_peak"] = data_for_condition.get("channels_without_peak", np.nan)

                # Spectral Variability
                if calc_sv and is_broadband_cond:
                    for band_name in FREQUENCY_BANDS:
                        if band_name.lower() == "broadband":
                            continue
                        sv_key = f"sv_{band_name}"
                        row[f"{condition}_sv_{band_name}"] = data_for_condition.get(sv_key, np.nan)

                # n_epochs
                row[f"{condition}_n_epochs"] = data_for_condition.get("n_epochs", 0)

            rows.append(row)

        # 4) Create the DataFrame, reorder columns, and export
        df = pd.DataFrame(rows)
        df = df[columns]  # Force the column order we built above
        df.to_excel(writer, sheet_name="Whole Brain Results", index=False)

        # Add analysis information sheet with sampling frequency info and spectral variability window
        info_data = {
            "Parameter": [
                "Analysis Date",
                "JPE Inversion",
                "PLI MST Calculated",
                "AEC Type",
                "AEC Concatenated Epochs",
                "AEC MST Calculated",
                "AEC Force Positive",
                "Power Bands Calculated",
                "Sampling Frequency (Hz)",
                "PSD Method",
                "Welch Window Length (ms)",
                "Welch Overlap (%)",
                "Peak Frequency Analysis",
                "Peak Frequency Range (Hz)",
                "Sample Entropy Calculated",
                "Approximate Entropy Calculated",
                "Spectral Variability Calculated",
                "Spectral Variability Window (ms)",
                "Channel Averages Calculated",
                "Channel Names Source",
            ],
            "Value": [
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Yes" if invert else "No",
                "Yes" if calc_pli_mst else "No",
                "AECc (orthogonalized)" if calc_aec and use_aecc else "AEC" if calc_aec else "Not calculated",
                "Yes" if concat_aecc else "No",
                "Yes" if calc_aec_mst else "No",
                "Yes" if force_positive else "No",
                "Yes" if calc_power else "No",
                str(power_fs),
                psd_method,  # New
                str(welch_window_ms) if psd_method == "welch" else "N/A",  # New
                str(welch_overlap) if psd_method == "welch" else "N/A",  # New
                "Yes" if calc_peak else "No",
                f"{calc_peak and f'{peak_min}-{peak_max}' or 'N/A'}",
                "Yes" if calc_sampen else "No",
                "Yes" if calc_apen else "No",
                "Yes" if calc_sv else "No",
                str(sv_window) if calc_sv else "N/A",
                "Yes" if save_channel_averages else "No",
                "File Headers" if has_headers else "Auto-generated",
            ],
        }

        info_df = pd.DataFrame(info_data)
        info_df.to_excel(writer, sheet_name="Analysis Information", index=False)

        # Save channel-level averages if requested
        if save_channel_averages:
            # First, gather all unique channels across all conditions
            all_channels = set()
            for subject, conditions in results_dict.items():
                for condition, result in conditions.items():
                    if result.get("channel_averages"):
                        all_channels.update(result["channel_averages"].keys())

            # Create rows with separate columns for each condition
            channel_rows = []
            # Keep track of which measures actually have data
            measures_with_data = set()

            for subject, conditions in results_dict.items():
                for channel in sorted(all_channels):
                    row = {"subject": subject, "channel": channel}

                    # For each condition, add all its measures
                    for condition in sorted(all_conditions):
                        if condition in conditions and conditions[condition].get("channel_averages"):
                            channel_data = conditions[condition]["channel_averages"].get(channel, {})
                            # Add each measure with condition prefix
                            for measure, value in channel_data.items():
                                column_name = f"{condition}_{measure}"
                                if measure.startswith("sv_"):  # Special handling for SV measures
                                    # SV values are now single values, not averaged
                                    row[column_name] = value
                                else:
                                    # Handle other measures as before
                                    row[column_name] = value

                                if not pd.isna(value):  # Only track columns that have actual data
                                    measures_with_data.add(column_name)

                    channel_rows.append(row)

            if channel_rows:
                df_channels = pd.DataFrame(channel_rows)

                # Keep only columns that have data
                base_cols = ["subject", "channel"]
                data_cols = sorted(measures_with_data, key=lambda x: (x.split("_")[0], x))

                # Final column order
                column_order = base_cols + data_cols
                df_channels = df_channels[column_order]
                df_channels.to_excel(writer, sheet_name="Channel Averages", index=False)

        # Save metadata about channels
        metadata_rows = []
        for subject, conditions in results_dict.items():
            for condition in sorted(all_conditions):
                if condition in conditions and "channel_names" in conditions[condition]:
                    metadata_rows.append(
                        {
                            "subject": subject,
                            "condition": condition,
                            "channels": ", ".join(conditions[condition]["channel_names"]),
                            "n_channels": len(conditions[condition]["channel_names"]),
                        }
                    )

        if metadata_rows:
            metadata_df = pd.DataFrame(metadata_rows)
            metadata_df.to_excel(writer, sheet_name="Channel Information", index=False)

    logger.info(f"Results saved to: {output_path}")
    print(f"\nResults saved to {output_path}")


def main():
    """Run the GUI and process EEG data analysis."""
    window = create_gui()

    while True:
        event, values = window.read()

        if event in (sg.WIN_CLOSED, "Exit"):
            break

        if event == "-PSD_METHOD-":  # When PSD method changes
            window.refresh()

        if event == "Process":
            folder_path = values["-FOLDER-"]
            folder_ext = values["-EXTENSION-"].strip()

            # Get PSD method parameters
            psd_method = values["-PSD_METHOD-"].lower()  # Convert to lowercase
            welch_window_ms = None
            welch_overlap = None

            if psd_method == "welch":
                try:
                    welch_window_ms = float(values["-WELCH_WINDOW-"])
                    welch_overlap = float(values["-WELCH_OVERLAP-"])
                    if welch_window_ms <= 0:
                        msg = "Welch window length must be greater than 0"
                        raise ValueError(msg)
                    if not 0 <= welch_overlap <= 100:  # noqa: PLR2004
                        msg = "Welch overlap must be between 0 and 100"
                        raise ValueError(msg)
                except ValueError:
                    sg.popup_error("Invalid Welch parameters")
                    continue

            # Setup logging first thing
            log_file = setup_logging(folder_path)
            logger.info("=== Starting new analysis run ===")
            logger.info(f"Folder path: {folder_path}")
            logger.info(f"Extension: {folder_ext}")
            logger.info(f"Processing files with{'out' if not values['-HAS_HEADERS-'] else ''} headers")
            if not values["-HAS_HEADERS-"]:
                logger.info("Channel names will be auto-generated")

            try:
                n_threads = int(values["-THREADS-"])
                if n_threads < 1 or n_threads > cpu_count():
                    msg = f"Number of threads must be between 1 and {cpu_count()}"
                    raise ValueError(msg)
            except ValueError:
                sg.popup_error("Invalid number of threads")
                continue

            if not folder_path or not folder_ext:
                sg.popup_error("Please select a folder and specify folder extension")
                continue

            try:
                validate_frequency_bands()
            except ValueError:
                sg.popup_error("Invalid frequency band configuration")
                sys.exit(1)

            # Get matrix saving options
            save_matrices = values["-SAVE_MATRICES-"]
            matrix_folder = values["-MATRIX_FOLDER-"]
            save_mst = values["-SAVE_MST-"]
            mst_folder = values["-MST_FOLDER-"]

            if save_matrices and not matrix_folder.strip():
                sg.popup_error("Please specify a folder name for saving matrices")
                continue

            if save_mst and not mst_folder.strip():
                sg.popup_error("Please specify a folder name for saving MST matrices")
                continue

            grouped_files = group_epochs_by_condition(folder_path, folder_ext)
            if not grouped_files:
                continue

            try:
                # Validate JPE time step
                try:
                    jpe_st = int(values["-JPE_ST-"])
                    if jpe_st < 1:
                        msg = "Time step must be greater than 0"
                        raise ValueError(msg)
                        # TODO: raising and immediately catching it seems redundant.
                        # Does it have to happen like this for PySimpleGUI to work properly?
                        # Simpler would be: `if condition: sg.popup_error("Message")`
                        # check throughout; if changed then de-ignore TRY301 in ruff settings
                except ValueError:
                    sg.popup_error("Invalid time step value")
                    continue

                # Validate power sampling frequency
                try:
                    power_fs = float(values["-POWER_FS-"])
                    if power_fs <= 0:
                        msg = "Sampling frequency must be greater than 0"
                        raise ValueError(msg)
                except ValueError:
                    sg.popup_error("Invalid sampling frequency value")
                    continue

                # Validate peak frequency range
                peak_min = peak_max = None
                if values["-CALC_PEAK-"]:
                    try:
                        peak_min = float(values["-PEAK_MIN-"])
                        peak_max = float(values["-PEAK_MAX-"])
                        if peak_min >= peak_max:
                            msg = "Minimum frequency must be less than maximum"
                            raise ValueError(msg)
                        if peak_min < 0 or peak_max > (power_fs / 2):
                            msg = f"Frequency range must be between 0 and {power_fs / 2} Hz"
                            raise ValueError(msg)
                    except ValueError:
                        sg.popup_error("Invalid peak frequency range")
                        continue

                # Validate SampEn parameters
                sampen_m = None
                if values["-CALC_SAMPEN-"]:
                    try:
                        sampen_m = int(values["-SAMPEN_M-"])
                        if sampen_m < 1:
                            msg = "Order m must be greater than 0"
                            raise ValueError(msg)
                    except ValueError:
                        sg.popup_error("Invalid SampEn order parameter")
                        continue

                # Validate ApEn parameters
                apen_m = apen_r = None

                if values["-CALC_APEN-"]:
                    try:
                        apen_m = int(values["-APEN_M-"])
                        if apen_m <= 0:
                            msg = "Order m must be greater than 0"
                            raise ValueError(msg)

                        apen_r = float(values["-APEN_R-"])
                        if apen_r <= 0:
                            msg = "Tolerance r must be greater than 0"
                            raise ValueError(msg)
                    except ValueError:
                        sg.popup_error("Invalid ApEn parameter")
                        continue

                # Validate spectral variability window
                sv_window = None
                if values["-CALC_SV-"]:
                    try:
                        sv_window = int(values["-SV_WINDOW-"])
                        if sv_window < MIN_WINDOW_SIZE:
                            msg = f"Window length must be at least {MIN_WINDOW_SIZE} ms"
                            raise ValueError(msg)
                    except ValueError:
                        sg.popup_error("Invalid spectral variability window")
                        continue

                def update_progress(value):
                    window["-PROGRESS-"].update(value)
                    window.refresh()

                results = process_all_subjects(
                    grouped_files,
                    convert_ints_pe=values["-CONVERT_INTS_PE-"],
                    invert=values["-INVERT-"],
                    n_threads=n_threads,
                    calc_jpe=values["-CALC_JPE-"],
                    calc_pli=values["-CALC_PLI-"],
                    calc_pli_mst=values["-CALC_PLI_MST-"],
                    calc_aec=values["-CALC_AEC-"],
                    use_aecc=values["-USE_AECC-"],
                    force_positive=values["-AEC_FORCE_POSITIVE-"],
                    concat_aecc=values["-CONCAT_AECC-"],
                    has_headers=values["-HAS_HEADERS-"],
                    jpe_st=jpe_st,
                    calc_aec_mst=values["-CALC_AEC_MST-"],
                    calc_power=values["-CALC_POWER-"],
                    power_fs=power_fs,
                    calc_peak=values["-CALC_PEAK-"],
                    peak_min=peak_min,
                    peak_max=peak_max,
                    calc_sampen=values["-CALC_SAMPEN-"],
                    sampen_m=sampen_m,
                    calc_apen=values["-CALC_APEN-"],
                    apen_m=apen_m,
                    apen_r=apen_r,
                    calc_sv=values["-CALC_SV-"],
                    sv_window=sv_window,
                    save_matrices=save_matrices,
                    save_mst=save_mst,
                    save_channel_averages=values["-SAVE_CHANNEL_AVERAGES-"],
                    psd_method=psd_method,
                    welch_window_ms=welch_window_ms if psd_method == "welch" else None,
                    welch_overlap=welch_overlap if psd_method == "welch" else None,
                    progress_callback=update_progress,
                )

                if results:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_path = os.path.join(folder_path, f"EEG_analysis_{timestamp}.xlsx")

                    try:
                        save_results_to_excel(
                            results,
                            output_path,
                            values["-INVERT-"],
                            values["-CALC_PLI_MST-"],
                            calc_jpe=values["-CALC_JPE-"],
                            calc_pli=values["-CALC_PLI-"],
                            calc_aec=values["-CALC_AEC-"],
                            use_aecc=values["-USE_AECC-"],
                            force_positive=values["-AEC_FORCE_POSITIVE-"],
                            calc_aec_mst=values["-CALC_AEC_MST-"],
                            calc_power=values["-CALC_POWER-"],
                            power_fs=power_fs,
                            calc_peak=values["-CALC_PEAK-"],
                            peak_min=peak_min,
                            peak_max=peak_max,
                            calc_sampen=values["-CALC_SAMPEN-"],
                            calc_apen=values["-CALC_APEN-"],
                            calc_sv=values["-CALC_SV-"],
                            sv_window=sv_window if values["-CALC_SV-"] else None,
                            save_channel_averages=values["-SAVE_CHANNEL_AVERAGES-"],
                            concat_aecc=values["-CONCAT_AECC-"],
                            has_headers=values["-HAS_HEADERS-"],
                            psd_method=psd_method,  # New
                            welch_window_ms=welch_window_ms if psd_method == "welch" else None,  # New
                            welch_overlap=welch_overlap if psd_method == "welch" else None,  # New
                        )
                        # Save matrices if requested
                        matrices_saved = 0
                        mst_matrices_saved = 0

                        logger.info("Starting matrix saving process...")

                        if save_matrices or save_mst:
                            folders = create_matrix_folder_structure(
                                folder_path, matrix_folder, mst_folder if save_mst else None
                            )

                            for subject, conditions in results.items():
                                for condition, result in conditions.items():
                                    if "matrices" in result and result["matrices"]:
                                        freq_band = extract_freq_band(condition)
                                        matrices = result["matrices"]
                                        current_channel_names = result["channel_names"]

                                        # Extract level type from condition
                                        level_type = "source" if "source" in condition.lower() else "sensor"

                                        # Save regular connectivity matrices
                                        if save_matrices:
                                            for feature, matrix in matrices.items():
                                                if matrix is not None and feature in ["jpe", "pli", "aec"]:
                                                    try:
                                                        if len(matrix) == len(current_channel_names):
                                                            filepath = save_connectivity_matrix(
                                                                matrix,
                                                                folders[feature],
                                                                subject,
                                                                freq_band,
                                                                feature,
                                                                current_channel_names,
                                                                level_type,
                                                            )
                                                            matrices_saved += 1
                                                            logger.info(f"Saved {feature} matrix to: {filepath}")
                                                        else:
                                                            logger.exception(
                                                                f"Matrix dimension ({len(matrix)}) doesn't match channel count ({len(current_channel_names)})"
                                                            )
                                                    except Exception:
                                                        logger.exception(f"Error saving {feature} matrix")

                                        # Save MST matrices
                                        if save_mst:
                                            for mst_type, matrix_key in [
                                                ("pli_mst", "pli_mst"),
                                                ("aec_mst", "aec_mst"),
                                            ]:
                                                if matrix_key in matrices and matrices[matrix_key] is not None:
                                                    try:
                                                        if len(matrices[matrix_key]) == len(current_channel_names):
                                                            filepath = save_connectivity_matrix(
                                                                matrices[matrix_key],
                                                                folders[mst_type],
                                                                subject,
                                                                freq_band,
                                                                matrix_key,
                                                                current_channel_names,
                                                                level_type,
                                                            )
                                                            mst_matrices_saved += 1
                                                            logger.info(f"Saved {matrix_key} matrix to: {filepath}")
                                                        else:
                                                            logger.exception(
                                                                f"MST matrix dimension ({len(matrices[matrix_key])}) doesn't match channel count ({len(current_channel_names)})"
                                                            )
                                                    except Exception:
                                                        logger.exception(f"Error saving {matrix_key} matrix")

                        if matrices_saved > 0:
                            logger.info(f"Saved {matrices_saved} connectivity matrices")
                        if mst_matrices_saved > 0:
                            logger.info(f"Saved {mst_matrices_saved} MST matrices")

                        logger.info(f"Results saved to: {output_path}")
                        success_message = f"Analysis complete!\nResults saved to:\n{output_path}"
                        if save_matrices:
                            success_message += f"\nConnectivity matrices saved in: {matrix_folder}"
                        if save_mst:
                            success_message += f"\nMST matrices saved in: {mst_folder}"
                        success_message += f"\nLog file: {log_file}"
                        sg.popup(success_message)

                    except Exception:
                        logger.exception("Error saving results")
                        sg.popup_error("Error saving results")
                else:
                    logger.warning("No results were generated")
                    sg.popup_error("No results were generated")

            except Exception:
                logger.exception("Error during processing")
                sg.popup_error("Error during processing")

            finally:
                logger.info("Analysis run completed")

    window.close()


if __name__ == "__main__":
    main()
