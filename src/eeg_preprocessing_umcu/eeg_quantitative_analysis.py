PySimpleGUI_License = "ePycJVMLaeW5NflzbNn9NLlOVFHzl7w4ZaSLIk6MIYkvRWpncB3ORHyiauWyJB1gdyGQlPvXbgi7IAswIIk3xnpjYF2QVPuccn2aVNJdRhCnI96RMUTtcdzOMYzIM05eNLjWQjyxMQi8wGirTkGxl2jUZcWc5YzYZOUyRXllcQGExvvFeDWz1jlVbVnEROW3ZzXIJGzUalWp9huwImjmojivNhSr4Ew9ItiOwHikTXmpFxthZIUrZ8pfcCnFNQ0iIJjyokihWeWm95yfYymHVJu2I4iqwxi5T9mAFatLZaUkxphHcv3DQRi3OcipJZMBbN2ZR6lvbeWEEkiuL7CfJSDIbt2n1mwBY8WY555qIsj8oriYISiJwNiwQq3GVvzEdaGL9VtyZMXbJgJIRnCrI06II1jCQNxDNSj8YEyPICiNwyiaR1GVFw0VZOUeltz1cW3EVFlUZcC3IG6nIUj3ICwjM3jQQ6tWMUT5IAtLMADTUdi8L1ClJQERYEXfRelmR0XBhDwTa5XyJalIcXyoIX6aIvj3ICwSM3jcYrtcMYT2AFtgMMDTkNiNL4CtJKFIbDWmFQpNbUEFFckFZeHrJVlRcY3DM9isOmicJ258bD3qJjiKZVWp5Psab62lRPllb7WbFQAbcOHtJsv6dUGm9EueLXmv1ylIIliowFiAShVhBZBwZuGnRVyXZrXMNdzTI9j7osioM5T8QxztLBjCEoyPMvSI4XySMRzxk3umMeT5MeisfZQJ=c=02504c6fb7ca09721d288ae69f8237c96a99697e5b723e543938c4be810e2615f6fa037769c1edbd61ae40a244556b95fdfc2843df8e3807e955bc2c1d4be04c7022e2aa84c8eef696a9c6a61297e79cc4f465fb5e94513820c17814b2d35afadfa00653a9157afbad05ce088b890ca447c12c1df95d67e61ceed0b57d99ee7f26bfca445ad111393dab2dd1b6bee992510a1e973d0c6fae38f654816cc8de05ce7a79081d2029d636be38fb06ff7c68bfa0bdf080c7bb349a71ec74894e9f746bcbe58a67482485609109ec0a416582fc50f3500f55d5a021e7ea0ce4aafa6a207c77b80c2b48484e70314ef2b1a14970f110336f4c68eed12b49b4f3560b9e48eca892473d97b6ccb712cd086b0baa6aef3aa59be23f951a3476fbc5824402af301b988f410cf050f722fa3f2995ae68d4852645384eccec7841c10fe44b08102cc32a6d94a5854d0a148cecf8d25a51067db2e71842845dd715141ca15f1a5dd475bf4cba5afb23e794e77a53b89590ea0a37e638d46c73c869f4957c4a445d813a94167f3aaca7b58ce66ccb0c605e4820cc661c3d2ae832e41ee9fd46357fb40d26e103d4d747794f8548c27c363e096d495269740a6c08e5f936aec6c689a5a18694b24c37268c9c18760d063ad62b96d505b01074f81d7bb94d456c0d2bca0dd8b96b2246167bb1d0ce36a44a4ec051d22a72260ebbf910b375e511158"
import PySimpleGUI as sg
import pandas as pd
import numpy as np
import os
import itertools
import math
import re
import psutil
import logging
from datetime import datetime
from collections import defaultdict
from multiprocessing import Pool, cpu_count
import time
from scipy.signal import hilbert
import networkx as nx
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy import signal
from scipy.stats import gaussian_kde
from antropy import sample_entropy
from typing import Dict, Optional, Tuple, Union

# Configuration
FOLDER_EXTENSION = 'bdf'  # Change this to match your folder extension (e.g., 'bdf', 'edf', etc.)
MAX_MEMORY_PERCENT = 70  # Maximum memory usage percentage

BATCH_SIZE = 10  # Number of subjects to process in parallel
DEFAULT_THREADS = max(1, int(cpu_count() * 0.8))  # Use 80% of cores, no max limit

class MemoryMonitor:
    @staticmethod
    def get_memory_usage():
        """Get current memory usage percentage"""
        return psutil.Process().memory_percent()

    @staticmethod
    def check_memory():
        """Check if memory usage is too high"""
        if MemoryMonitor.get_memory_usage() > MAX_MEMORY_PERCENT:
            return True
        return False

def setup_logging(folder_path):
    """Setup logging for the current analysis run"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(folder_path, f'eeg_analysis_{timestamp}.log')
    
    # Clear any existing handlers
    logging.getLogger().handlers = []
    
    # Configure logging with both file and console output
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, mode='w'),
            logging.StreamHandler()
        ]
    )
    
    # Test the logging setup
    logging.info("Logging initialized")
    logging.info(f"Log file created at: {log_filename}")
    
    return log_filename

def create_gui():
    suggested_threads = DEFAULT_THREADS
    
    HEADER_BG = '#2C5784'
    HEADER_TEXT = '#FFFFFF'
    MAIN_BG = '#F0F2F6'
    BUTTON_COLOR = ('#FFFFFF', '#2C5784')
    
    sg.theme('Default1')
    sg.set_options(font=('Helvetica', 11))
    
    header = [
        [sg.Text("EEG Quantitative Analysis", font=('Helvetica', 24, 'bold'), 
                text_color=HEADER_TEXT, background_color=HEADER_BG, pad=(20, 10))],
        [sg.Text("Author: Yorben Lodema", font=('Helvetica', 12, 'italic'), 
                text_color=HEADER_TEXT, background_color=HEADER_BG, pad=(20, 10))]
    ]
    
    left_column = [
        [sg.Frame('Input Settings', [
            [sg.Text("Select data folder:", font=('Helvetica', 11, 'bold'), background_color=MAIN_BG)],
            [sg.Input(key="-FOLDER-", size=(30, 1)), sg.FolderBrowse(button_color=BUTTON_COLOR)],
            [sg.Text("Folder extension:", background_color=MAIN_BG), sg.Input(FOLDER_EXTENSION, key="-EXTENSION-", size=(10, 1))],
            [sg.Text("Processing threads:", background_color=MAIN_BG), sg.Input(suggested_threads, key="-THREADS-", size=(5, 1))],
            [sg.Checkbox("Epoch files have headers", key="-HAS_HEADERS-", default=True, background_color=MAIN_BG)],
        ], background_color=MAIN_BG)],
                
        [sg.Frame('Complexity Measures', [
            [sg.Checkbox("Calculate JPE/PE", key="-CALC_JPE-", default=False, background_color=MAIN_BG)],
            [sg.Text("Time step (tau):", background_color=MAIN_BG), sg.Input("1", key="-JPE_ST-", size=(5, 1))],
            [sg.Checkbox("Convert to integers", key="-CONVERT_INTS_PE-", default=False, background_color=MAIN_BG)],
            [sg.Checkbox("Invert JPE (1-entropy)", key="-INVERT-", default=True, background_color=MAIN_BG)],
            [sg.Checkbox("Calculate Sample Entropy", key="-CALC_SAMPEN-", default=False, background_color=MAIN_BG)],
            [sg.Text("Order (m):", background_color=MAIN_BG), sg.Input("2", key="-SAMPEN_M-", size=(3, 1))],
            [sg.Checkbox("Calculate Approximate Entropy", key="-CALC_APEN-", default=False, background_color=MAIN_BG)],
            [sg.Text("Order (m):", background_color=MAIN_BG), sg.Input("1", key="-APEN_M-", size=(3, 1))],
            [sg.Text("Tolerance (r):", background_color=MAIN_BG), sg.Input("0.25", key="-APEN_R-", size=(3, 1))],
        ], background_color=MAIN_BG)],
                        
        [sg.Frame('Matrix Export', [
            [sg.Checkbox("Save connectivity matrices", key="-SAVE_MATRICES-", default=False, background_color=MAIN_BG)],
            [sg.Text("Matrix folder name:", background_color=MAIN_BG), 
             sg.Input("connectivity_matrices", key="-MATRIX_FOLDER-", size=(20, 1))],
            [sg.Checkbox("Save MST matrices", key="-SAVE_MST-", default=False, background_color=MAIN_BG)],
            [sg.Text("MST folder name:", background_color=MAIN_BG), 
             sg.Input("mst_matrices", key="-MST_FOLDER-", size=(20, 1))],
            [sg.Checkbox("Save channel-level averages", key="-SAVE_CHANNEL_AVERAGES-", default=False, background_color=MAIN_BG)],
        ], background_color=MAIN_BG)],
    ]
    
    right_column = [
        [sg.Frame('Spectral Analysis', [
            [sg.Text("Sampling rate (Hz):", background_color=MAIN_BG), sg.Input(key="-POWER_FS-", size=(8, 1))],
            [sg.Checkbox("Calculate power bands", key="-CALC_POWER-", default=False, background_color=MAIN_BG)],
            [sg.Checkbox("Calculate peak frequency", key="-CALC_PEAK-", default=False, background_color=MAIN_BG)],
            [sg.Text("Freq range (Hz):", background_color=MAIN_BG), 
             sg.Input("4", key="-PEAK_MIN-", size=(4, 1)), 
             sg.Text("-", background_color=MAIN_BG),
             sg.Input("13", key="-PEAK_MAX-", size=(4, 1))],
            [sg.Checkbox("Calculate spectral variability", key="-CALC_SV-", default=False, background_color=MAIN_BG)],
            [sg.Text("Window (ms):", background_color=MAIN_BG), sg.Input("2000", key="-SV_WINDOW-", size=(6, 1))],
        ], background_color=MAIN_BG)],
                
        [sg.Frame('Connectivity', [
            [sg.Checkbox("Calculate PLI", key="-CALC_PLI-", default=False, background_color=MAIN_BG)],
            [sg.Checkbox("Calculate PLI MST measures", key="-CALC_PLI_MST-", default=False, background_color=MAIN_BG)],
            [sg.Checkbox("Calculate AEC", key="-CALC_AEC-", default=False, background_color=MAIN_BG)],
            [sg.Checkbox("Use orthogonalization (AECc)", key="-USE_AECC-", default=False, background_color=MAIN_BG)],
            [sg.Checkbox("Concatenate epochs for AEC(c)", key="-CONCAT_AECC-", default=True, background_color=MAIN_BG)],
            [sg.Checkbox("Calculate AEC(c) MST measures", key="-CALC_AEC_MST-", default=False, background_color=MAIN_BG)],
            [sg.Checkbox("AEC make negative corr. zero", key="-AEC_FORCE_POSITIVE-", default=True, background_color=MAIN_BG)],
        ], background_color=MAIN_BG)]
    ]
    
    progress_section = [
        [sg.Frame('Progress', [
            [sg.Multiline(size=(70, 8), key='-LOG-', autoscroll=True, reroute_stdout=True,
                         disabled=True, background_color='#FFFFFF', text_color='#000000')],
            [sg.ProgressBar(100, orientation='h', size=(60, 20), key='-PROGRESS-',
                          bar_color=(HEADER_BG, MAIN_BG))],
            [sg.Column([[sg.Button("Process", size=(10, 1), button_color=BUTTON_COLOR, 
                                 font=('Helvetica', 11, 'bold')),
                        sg.Button("Exit", size=(8, 1), button_color=(HEADER_TEXT, '#AB4F4F'),
                                font=('Helvetica', 11))]], 
                      justification='center', expand_x=True, pad=(0, 10), background_color=MAIN_BG)],
        ], background_color=MAIN_BG)],
    ]
    
    layout = [
        [sg.Column(header, background_color=HEADER_BG, expand_x=True)],
        [sg.Column([
            [sg.Column(left_column, background_color=MAIN_BG, pad=(10, 5)),
             sg.Column(right_column, background_color=MAIN_BG, pad=(10, 5))],
            [sg.Column(progress_section, background_color=MAIN_BG, pad=(0, 5))]
        ], background_color=MAIN_BG, pad=(20, 20))]
    ]
    
    window = sg.Window("EEG Analysis Tool", 
                      layout,
                      background_color=MAIN_BG,
                      finalize=True,
                      margins=(0, 0))
    
    return window

def create_matrix_folder_structure(base_folder, matrix_folder_name, mst_folder_name=None):
    """Create folder structure with subject subfolders"""
    folders = {
        'jpe': os.path.join(base_folder, matrix_folder_name, 'jpe'),
        'pli': os.path.join(base_folder, matrix_folder_name, 'pli'),
        'aec': os.path.join(base_folder, matrix_folder_name, 'aec')
    }
    
    if mst_folder_name:
        folders.update({
            'pli_mst': os.path.join(base_folder, mst_folder_name, 'pli_mst'),
            'aec_mst': os.path.join(base_folder, mst_folder_name, 'aec_mst')
        })
    
    # Create base folders
    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)
    
    return folders

def extract_freq_band(condition):
    """Extract frequency band from condition string"""
    freq_patterns = {
        r'0.5-4.0': 'delta',
        r'4.0-8.0': 'theta',
        r'8.0-13.0': 'alpha',
        r'13.0-20.0': 'beta1',
        r'20.0-30.0': 'beta2',
        r'0.5-47': 'broadband'
    }
    
    for pattern, band_name in freq_patterns.items():
        if re.search(pattern, condition):
            return band_name
    return 'unknown'

def save_connectivity_matrix(matrix, folder_path, subject, freq_band, feature, channel_names, level_type=None):
    """Save connectivity matrix to CSV with proper channel names"""
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

def calculate_PSD(data: np.ndarray,
                 fs: float,
                 method: str = 'multitaper',
                 freq_range: Optional[Tuple[float, float]] = None,
                 compute_spectrogram: bool = False,
                 **kwargs) -> Dict[str, np.ndarray]:
    """
    Calculate Power Spectral Density (PSD) and optionally spectrogram using specified method.
    
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
            Multitaper:
                time_bandwidth : float (default 4)
                n_tapers : int (optional, computed from time_bandwidth)
            Spectrogram:
                window_length : int (in ms)
                overlap : float (0 to 1)
    
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
            'times' : np.ndarray, optional
                Time points for spectrogram (only if compute_spectrogram=True)
    """
    if method not in ['multitaper', 'welch', 'fft']:
        raise ValueError(f"Unknown method: {method}")
        
    # Input validation
    if not isinstance(data, np.ndarray):
        raise ValueError("Data must be a numpy array")
    if data.ndim != 2:
        raise ValueError("Data must be 2D array (samples x channels)")
    if fs <= 0:
        raise ValueError("Sampling frequency must be positive")
        
    # Initialize return dictionary
    result = {}
    
    # Calculate PSD based on method
    if method == 'multitaper':
        time_bandwidth = kwargs.get('time_bandwidth', 4.0)
        n_tapers = kwargs.get('n_tapers', int(2 * time_bandwidth - 1))
        
        try:
            frequencies, psd = _calculate_multitaper_psd(data, fs, time_bandwidth, n_tapers)
            result['frequencies'] = frequencies
            result['psd'] = psd
            logging.info(f"Calculated multitaper PSD with {n_tapers} tapers")
            
        except Exception as e:
            logging.error(f"Error calculating multitaper PSD: {str(e)}")
            raise
            
    elif method == 'welch':
        # Placeholder for future Welch implementation
        raise NotImplementedError("Welch's method not yet implemented")
        
    elif method == 'fft':
        # Placeholder for future FFT implementation
        raise NotImplementedError("FFT method not yet implemented")
    
    # Apply frequency range if specified
    if freq_range is not None:
        fmin, fmax = freq_range
        if not (0 <= fmin < fmax <= fs/2):
            raise ValueError(f"Invalid frequency range: {freq_range}")
        
        freq_mask = (result['frequencies'] >= fmin) & (result['frequencies'] <= fmax)
        result['frequencies'] = result['frequencies'][freq_mask]
        result['psd'] = result['psd'][freq_mask]
    
    # Calculate spectrogram if requested
    if compute_spectrogram:
        try:
            window_length = kwargs.get('window_length', 2000)  # default 2000ms
            overlap = kwargs.get('overlap', 0.5)  # default 50% overlap
            
            times, spect = _calculate_spectrogram(data, fs, window_length, overlap)
            result['spectrogram'] = spect
            result['times'] = times
            logging.info("Calculated spectrogram")
            
        except Exception as e:
            logging.error(f"Error calculating spectrogram: {str(e)}")
            raise
    
    return result

def _calculate_multitaper_psd(data: np.ndarray,
                            fs: float,
                            time_bandwidth: float,
                            n_tapers: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate PSD using multitaper method.
    
    Parameters
    ----------
    data : np.ndarray
        Time series data (samples x channels)
    fs : float
        Sampling frequency
    time_bandwidth : float
        Time-bandwidth product
    n_tapers : int
        Number of tapers to use
        
    Returns
    -------
    frequencies : np.ndarray
        Frequency values
    psd : np.ndarray
        Power spectral density estimates (frequencies x channels)
    """
    n_samples, n_channels = data.shape
    
    # Initialize output array
    psd = np.zeros((n_samples//2 + 1, n_channels))
    
    # Process each channel
    for ch in range(n_channels):
        frequencies, channel_psd = signal.multitaper.pmtm(
            data[:, ch],
            NW=time_bandwidth,
            k=n_tapers,
            fs=fs,
            return_onesided=True
        )
        psd[:, ch] = channel_psd
        
        if ch == 0:  # Save frequencies from first channel
            freq_values = frequencies
            
    return freq_values, psd

def _calculate_spectrogram(data: np.ndarray,
                         fs: float,
                         window_length: int,
                         overlap: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate spectrogram using multitaper method.
    
    Parameters
    ----------
    data : np.ndarray
        Time series data (samples x channels)
    fs : float
        Sampling frequency
    window_length : int
        Length of window in milliseconds
    overlap : float
        Overlap between windows (0 to 1)
        
    Returns
    -------
    times : np.ndarray
        Time points
    spectrogram : np.ndarray
        Time-frequency representation
    """
    # Convert window length from ms to samples
    nperseg = int(window_length * fs / 1000)
    noverlap = int(nperseg * overlap)
    
    # Calculate spectrogram for each channel
    n_channels = data.shape[1]
    first_run = True
    
    for ch in range(n_channels):
        f, t, sxx = signal.spectrogram(
            data[:, ch],
            fs=fs,
            nperseg=nperseg,
            noverlap=noverlap,
            window='hann',
            detrend='constant'
        )
        
        if first_run:
            # Initialize arrays with correct dimensions
            frequencies = f
            times = t
            spectrogram = np.zeros((len(f), len(t), n_channels))
            first_run = False
            
        spectrogram[:, :, ch] = sxx
        
    return times, spectrogram

def calculate_sampen_for_channels(data, m=2):
    """
    Calculate Sample Entropy for each channel using antropy.
    
    Parameters:
    data : numpy array (time points × channels)
    m : int
        Embedding dimension (order)
        
    Returns:
    numpy.array : Sample Entropy values for each channel
    """
    n_channels = data.shape[1]
    sampen_values = np.zeros(n_channels)
    
    for ch in range(n_channels):
        try:
            sampen_values[ch] = sample_entropy(data[:, ch], order=m)
            
            if ch % 10 == 0:  # Log progress every 10 channels
                logging.info(f"Processed SampEn for {ch}/{n_channels} channels")
                
        except Exception as e:
            logging.error(f"Error calculating SampEn for channel {ch}: {str(e)}")
            sampen_values[ch] = np.nan
            
    return sampen_values

def calculate_apen_for_channels(data, m=1, r=0.25):
    """
    Calculate Approximate Entropy for each channel following Pincus 1995,
    with optimized implementation using vectorization.
    
    Parameters:
    data : numpy array (time points × channels)
    m : int
        Embedding dimension (length of compared runs)
    r : float
        Tolerance (typically 0.25 * std of the data)
        
    Returns:
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
                logging.info(f"Processed ApEn for {ch}/{n_channels} channels")
                
        except Exception as e:
            logging.error(f"Error calculating ApEn for channel {ch}: {str(e)}")
            apen_values[ch] = np.nan
            
    return apen_values

def _phi_vectorized(x, m, r):
    """
    Vectorized calculation of Φᵐ(r) following Pincus 1995.
    
    Parameters:
    x : array
        Time series data
    m : int
        Embedding dimension
    r : float
        Tolerance threshold
        
    Returns:
    float : Φᵐ(r) value
    """
    N = len(x)
    N_m = N - m + 1
    
    # Create embedding matrix efficiently
    # Each row is a pattern of length m
    patterns = np.zeros((N_m, m))
    for i in range(m):
        patterns[:, i] = x[i:i+N_m]
    
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

def calculate_spectral_variability(times, spectrogram, frequencies):
    """
    Calculate spectral variability using pre-computed spectrogram.
    
    Parameters:
    times : numpy array
        Time points
    spectrogram : numpy array
        Time-frequency representation (frequencies × times × channels)
    frequencies : numpy array
        Frequency values
        
    Returns:
    dict : Dictionary with CV values for each band
    """
    # Define frequency bands
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta1': (13, 20),
        'beta2': (20, 30),
    }
    
    # Initialize results
    cv_values = {band: np.zeros(spectrogram.shape[2]) for band in bands}
    
    # Calculate total power mask
    total_mask = (frequencies >= 0.5) & (frequencies <= 47)
    
    for channel in range(spectrogram.shape[2]):
        try:
            # Calculate band powers over time
            for band_name, (low_freq, high_freq) in bands.items():
                band_mask = (frequencies >= low_freq) & (frequencies < high_freq)
                
                # Calculate band power over time
                band_power = np.sum(spectrogram[band_mask, :, channel], axis=0)
                total_power = np.sum(spectrogram[total_mask, :, channel], axis=0)
                
                # Calculate relative power over time
                with np.errstate(divide='ignore', invalid='ignore'):
                    relative_power = np.where(total_power > 0, band_power / total_power, 0)
                
                # Calculate coefficient of variation
                if np.all(relative_power == 0):
                    cv_values[band_name][channel] = np.nan
                else:
                    cv = np.std(relative_power) / np.mean(relative_power)
                    cv_values[band_name][channel] = cv
                    
        except Exception as e:
            logging.error(f"Error processing channel {channel}: {str(e)}")
            for band in bands:
                cv_values[band][channel] = np.nan
    
    return cv_values

def smooth_spectrum(frequencies, power_spectrum, smoothing_window=5):
    """Apply moving average smoothing to power spectrum"""
    return np.convolve(power_spectrum, np.ones(smoothing_window)/smoothing_window, mode='same')

def find_peaks(x, y, threshold_ratio=0.5):
    """Find significant peaks using relative maxima and prominence threshold"""
    peak_indices = signal.argrelextrema(y, np.greater)[0]
    prominences = signal.peak_prominences(y, peak_indices)[0]
    threshold = threshold_ratio * np.max(prominences)
    significant_peaks = peak_indices[prominences > threshold]
    
    return x[significant_peaks], y[significant_peaks]

def calculate_avg_peak_frequency(frequencies, psd, freq_range=(4, 13), smoothing_window=5):
    """
    Calculate peak frequency using pre-computed PSD.
    
    Parameters:
    frequencies : numpy array
        Frequency values
    psd : numpy array
        Power spectral density (frequencies × channels)
    freq_range : tuple
        Frequency range to search for peaks
    smoothing_window : int
        Window size for smoothing
        
    Returns:
    numpy array: Peak frequencies for each channel
    """
    num_channels = psd.shape[1]
    peak_frequencies = np.zeros(num_channels)
    
    # Create frequency mask
    freq_mask = (frequencies >= freq_range[0]) & (frequencies <= freq_range[1])
    frequencies = frequencies[freq_mask]
    psd = psd[freq_mask, :]
    
    for channel in range(num_channels):
        smoothed_spectrum = smooth_spectrum(frequencies, psd[:, channel], smoothing_window)
        peak_freqs, peak_powers = find_peaks(frequencies, smoothed_spectrum)
        
        if len(peak_freqs) > 1:
            kde = gaussian_kde(peak_freqs, weights=peak_powers)
            x_range = np.linspace(min(peak_freqs), max(peak_freqs), 1000)
            kde_values = kde(x_range)
            peak_frequencies[channel] = x_range[np.argmax(kde_values)]
        elif len(peak_freqs) == 1:
            peak_frequencies[channel] = peak_freqs[0]
        else:
            peak_frequencies[channel] = np.nan
            
    return peak_frequencies

def calculate_power_bands(frequencies, psd):
    """
    Calculate absolute and relative power using pre-computed PSD.
    
    Parameters:
    frequencies : numpy array
        Frequency values
    psd : numpy array
        Power spectral density (frequencies × channels)
    
    Returns:
    dict: Dictionary containing absolute and relative power values
    """
    # Define frequency bands
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta1': (13, 20),
        'beta2': (20, 30)
    }
    
    # Initialize results dictionary
    powers = {}
    
    # Calculate total power in 0.5-47 Hz range for relative power calculation
    total_mask = (frequencies >= 0.5) & (frequencies <= 47)
    total_power = np.sum(psd[total_mask, :], axis=0)
    
    # Calculate power in each band
    for band_name, (fmin, fmax) in bands.items():
        # Find frequencies corresponding to current band
        mask = (frequencies >= fmin) & (frequencies <= fmax)
        
        # Calculate absolute power
        abs_power = np.sum(psd[mask, :], axis=0)  # Sum over frequencies for each channel
        powers[f'{band_name}_abs_power'] = np.mean(abs_power)  # Mean over channels
        
        # Calculate relative power
        rel_power = abs_power / total_power  # Element-wise division
        powers[f'{band_name}_rel_power'] = np.mean(rel_power)  # Mean over channels
    
    return powers

def calculate_mst_measures(connectivity_matrix, used_channels=None):
    """
    Calculate MST measures from a connectivity matrix with additional error handling for disconnected graphs.
    
    Args:
        connectivity_matrix (numpy.ndarray): Square connectivity matrix (e.g., PLI matrix)
        used_channels (numpy.ndarray, optional): Boolean array indicating which channels are used.
                                               If None, all channels are considered used.
    
    Returns:
        tuple: (dict of MST measures, MST matrix, bool indicating success)
    """
    # Initialize used_channels if not provided
    if used_channels is None:
        used_channels = np.ones(len(connectivity_matrix), dtype=bool)
    
    # Get number of total channels (N) and used channels (M)
    n_total = len(connectivity_matrix)  # N in BrainWave
    n_used = np.sum(used_channels)      # M in BrainWave
    norm_factor = n_used - 1            # (M-1) for initial normalization
    
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
            degrees[edge[0]] += 1.0/norm_factor  # norm_factor is (M-1)
            degrees[edge[1]] += 1.0/norm_factor
        
        measures['degree'] = max(degrees.values()) if degrees else 0

        # 2. Eccentricity - normalize by (M-1)
        eccentricity = nx.eccentricity(G)
        normalized_eccentricity = {node: ecc/norm_factor 
                                 for node, ecc in eccentricity.items() 
                                 if used_channels[node]}
        measures['eccentr'] = np.mean(list(normalized_eccentricity.values()))
        
        # 3. Betweenness centrality
        betweenness = nx.betweenness_centrality(G)
        measures['betweenness'] = max(list(betweenness.values()))
        
        # 4. Diameter - normalize by (M-1)
        raw_diameter = nx.diameter(G)
        measures['diameter'] = raw_diameter / norm_factor
        
        # 5. Leaf fraction        
        leaf_nodes = sum(1 for node, deg in degrees.items() 
                        if abs(deg - 1.0/norm_factor) < 1e-10)
        measures['leaf'] = leaf_nodes / n_used
        
        max_betweenness = max(betweenness.values()) if betweenness else 0
        if max_betweenness > 0:
            measures['hierarchy'] = leaf_nodes / (2 * max_betweenness * norm_factor)
        else:
            measures['hierarchy'] = 0
        
        # 6. Kappa (degree divergence)
        sum_x = sum((n_total-1) * deg for node, deg in degrees.items() if used_channels[node])
        sum_x2 = sum(((n_total-1) * deg)**2 for node, deg in degrees.items() if used_channels[node])
        measures['kappa'] = sum_x2 / sum_x if sum_x > 0 else 0
        
        # 7. Tree hierarchy
        max_betweenness = max(betweenness.values()) if betweenness else 0
        if max_betweenness > 0:
            measures['hierarchy'] = leaf_nodes / (2 * max_betweenness * norm_factor)
        else:
            measures['hierarchy'] = 0
        
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
        
        measures['asp'] = sum_distances / (n_used * (n_used - 1)) if n_used > 1 else 0
        
        # 9. Tree efficiency (Teff)
        normalized_diam = raw_diameter / norm_factor
        measures['teff'] = 1.0 - (normalized_diam * (n_used-1)) / (n_used - (n_used-1)*measures['leaf'] + 1.0)
        
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
                measures['r'] = cov / np.sqrt(var_i * var_j)
            else:
                measures['r'] = 0
        else:
            measures['r'] = 0
        
        # 11. Mean edge weight
        edge_weights = [abs(d.get('weight', 1.0)) for _, _, d in G.edges(data=True)]
        measures['mean'] = np.mean(edge_weights) if edge_weights else 0
        
        # 12. Reference value
        mst_sum = np.sum(abs(mst_matrix[used_channels][:, used_channels]))
        orig_sum = np.sum(connectivity_matrix[used_channels][:, used_channels])
        measures['ref'] = mst_sum / orig_sum if orig_sum > 0 else 0
        
        return measures, mst_matrix, True
        
    except Exception as e:
        logging.error(f"Error in MST measures calculation: {str(e)}")
        return None, None, False
    
def calculate_pli(data):
    """Optimized PLI calculation using vectorization"""
    analytic_signal = hilbert(data, axis=0)
    phases = np.angle(analytic_signal)
    
    n_channels = data.shape[1]
    pli = np.zeros((n_channels, n_channels))
    
    # Vectorized phase difference calculation
    for i in range(n_channels):
        phase_diffs = phases[:, i:i+1] - phases[:, i:]
        signs = np.sign(np.sin(phase_diffs))
        means = np.abs(np.mean(signs, axis=0))
        pli[i, i:] = means
        pli[i:, i] = means
    
    return pli

def convert_to_integers(data):
    """Convert to integers using simple truncation"""
    return data.astype(int)

def calculate_aecc(data, orthogonalize=False, force_positive=True):
    """
    Calculate amplitude envelope correlation with optional orthogonalization.
    
    Parameters:
    data : numpy array (time points × channels)
        EEG data array
    orthogonalize : bool, optional
        Whether to perform orthogonalization
    force_positive : bool, optional
        Whether to force negative correlations to zero
        
    Returns:
    numpy array (channels × channels)
        AEC(c) correlation matrix
    """
    def process_correlation(corr):
        """Helper function to process correlation based on force_positive setting"""
        return max(0.0, corr) if force_positive else corr
    
    n_channels = data.shape[1]
    correlation_matrix = np.zeros((n_channels, n_channels))
    
    if orthogonalize:
        # Process all channels pairwise
        for i in range(n_channels):
            for j in range(i+1, n_channels):
                # Orthogonalize in both directions
                d_orth_ij = data[:, j] - np.dot(data[:, j], data[:, i]) * data[:, i] / np.dot(data[:, i], data[:, i])
                d_orth_ji = data[:, i] - np.dot(data[:, i], data[:, j]) * data[:, j] / np.dot(data[:, j], data[:, j])
                
                # Calculate envelopes
                env_i = np.abs(hilbert(data[:, i]))
                env_j = np.abs(hilbert(data[:, j]))
                env_orth_ij = np.abs(hilbert(d_orth_ij))
                env_orth_ji = np.abs(hilbert(d_orth_ji))
                
                # Calculate correlations
                corr_ij = process_correlation(np.corrcoef(env_i, env_orth_ij)[0,1])
                corr_ji = process_correlation(np.corrcoef(env_j, env_orth_ji)[0,1])
                
                # Update correlation matrix
                correlation_matrix[i,j] = (corr_ij + corr_ji) / 2
                correlation_matrix[j,i] = correlation_matrix[i,j]
                
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
    
    Parameters:
        data : numpy array (time points × channels)
        n : int, embedding dimension
        st : int, time delay (should scale with sampling frequency)
        
    Returns:
        numpy array : PE values for each channel
    """
    sz = data.shape[0]
    combinations = list(itertools.permutations(np.arange(0,n), n))
    
    PEs = []
    for ch in range(data.shape[1]):
        pattern_counts = np.zeros(len(combinations))
        
        # Step size for moving between patterns should be fixed (e.g., 1)
        # Only the sampling interval (st) within patterns should scale with frequency
        for i in range(0, sz-n*st, 1):
            dat_array = data[i:i+n*st:st, ch]  
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

def find_mirror_patterns(combinations, n):
    """Create a lookup dictionary for mirror patterns"""
    mirrors = {}
    for i, perm1 in enumerate(combinations):
        for j, perm2 in enumerate(combinations):
            if i != j:
                if all(a + b == n + 1 for a, b in zip(perm1, perm2)):
                    mirrors[i] = j
                    mirrors[j] = i
                    break
    return mirrors

def is_volume_conduction(pattern1, pattern2, mirrors):
    """Check for volume conduction"""
    return pattern1 == pattern2 or pattern2 == mirrors.get(pattern1, -1)

def calculate_jpe(data, n=4, st=1, convert_ints=False, invert=True):
    """Calculate joint permutation entropy with corrected time delay handling.
    
    Parameters:
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
    combinations = list(itertools.permutations(np.arange(0,n), n))
    mirrors = find_mirror_patterns(combinations, n-1)
    
    # Modified to separate pattern step size from sampling interval
    rank_inds = []
    for i in range(0, sz-n*st, 1):  # Changed step size to 1
        dat_array = data[i:i+n*st:st, :]  # Keep st for within-pattern sampling
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
    """
    Parse epoch filename to extract components
    Example: testjulia20231115kopie2_Source_level_4.0-8.0 Hz_Epoch_20.txt
    """
    # Extract the base name (everything before first underscore)
    base_name = filename.split('_')[0]
    
    # Extract level type (Source or Sensor)
    level_match = re.search(r'(Source|Sensor)_level', filename)
    level_type = level_match.group(1).lower() if level_match else "unknown"
    
    # Extract frequency band
    freq_match = re.search(r'(\d+\.?\d*-\d+\.?\d*)\s*Hz', filename)
    freq_band = freq_match.group(1) if freq_match else "unknown"
    
    return {
        'base_name': base_name,
        'level_type': level_type,
        'freq_band': freq_band,
        'condition': f"{level_type}_{freq_band}"
    }
                    
def process_subject_condition(args):
    """Process a single subject-condition combination"""    
    subject, condition, epoch_files, convert_ints_pe, invert, calc_jpe, calc_pli, calc_pli_mst, \
    calc_aec, use_aecc, force_positive, jpe_st, calc_aec_mst, calc_power, power_fs, calc_peak, \
    peak_min, peak_max, calc_sampen, sampen_m, calc_apen, apen_m, apen_r, calc_sv, sv_window, \
    save_matrices, save_mst, save_channel_averages, concat_aecc, has_headers = args
    
    try:
        # Initialize storage for channel-level and whole-brain results
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
        sv_values = defaultdict(list)
        channel_names = None
        
        # Initialize counter for successful MST calculations
        successful_mst_epochs = 0
        mst_calculation_attempted = False

        # Initialize matrices for connectivity
        avg_matrices = {
            'jpe': None,
            'pli': None,
            'aec': None,
        }
        
        logging.info(f"Processing {subject} - {condition} ({len(epoch_files)} epochs)")
        
        for i, file_path in enumerate(epoch_files):
            if MemoryMonitor.check_memory():
                logging.warning(f"High memory usage detected while processing {subject}")
                time.sleep(1)
            
            try:
                # Read data file
                if has_headers:
                    data = pd.read_csv(file_path, sep=None, engine='python')
                    if channel_names is None:
                        channel_names = data.columns.tolist()
                        logging.info(f"{subject} - {condition}: Found {len(channel_names)} channels from headers")
                else:
                    # First read the first row to check if it's non-numerical
                    first_row = pd.read_csv(file_path, sep=None, engine='python', header=None, nrows=1)
                    is_header = False
                    
                    try:
                        first_row.astype(float)
                    except (ValueError, TypeError):
                        is_header = True
                        logging.info(f"Found non-numeric header in {os.path.basename(file_path)}, ignoring first row")
                    
                    data = pd.read_csv(file_path, sep=None, engine='python', header=None, 
                                     skiprows=1 if is_header else 0)
                    
                    for col in data.columns:
                        data[col] = pd.to_numeric(data[col], errors='coerce')
                    
                    if channel_names is None:
                        n_columns = len(data.columns)
                        channel_names = [f"Channel_{i+1}" for i in range(n_columns)]
                        logging.info(f"{subject} - {condition}: Generated {len(channel_names)} channel names")
                    else:
                        current_columns = len(data.columns)
                        if current_columns != len(channel_names):
                            error_msg = (f"Inconsistent number of columns in {os.path.basename(file_path)}. "
                                       f"Expected {len(channel_names)}, found {current_columns}")
                            logging.error(error_msg)
                            raise ValueError(error_msg)
            
                # Check for NaN values
                if data.isna().any().any():
                    logging.warning(f"Found non-numeric values in {os.path.basename(file_path)} that were converted to NaN")
                
                data_values = data.values
                del data
                
                # Determine if any spectral calculations are needed
                need_spectral = (calc_power or calc_peak or calc_sv) and '0.5-47' in condition
                
                if need_spectral:
                    try:
                        # Calculate PSDs with appropriate settings
                        spectral_data = calculate_PSD(
                            data=data_values,
                            fs=power_fs,
                            method='multitaper',
                            compute_spectrogram=calc_sv,
                            window_length=sv_window if calc_sv else None,
                            overlap=0.5 if calc_sv else None
                        )
                        
                        # Calculate power bands if requested
                        if calc_power:
                            try:
                                powers = calculate_power_bands(
                                    frequencies=spectral_data['frequencies'],
                                    psd=spectral_data['psd']
                                )
                                
                                for measure, value in powers.items():
                                    power_values[measure].append(value)
                                
                                if save_channel_averages:
                                    total_mask = (spectral_data['frequencies'] >= 0.5) & (spectral_data['frequencies'] <= 47)
                                    total_power = np.sum(spectral_data['psd'][total_mask, :], axis=0)
                                    
                                    bands = {
                                        'delta': (0.5, 4),
                                        'theta': (4, 8),
                                        'alpha': (8, 13),
                                        'beta1': (13, 20),
                                        'beta2': (20, 30)
                                    }
                                    
                                    for band_name, (fmin, fmax) in bands.items():
                                        mask = (spectral_data['frequencies'] >= fmin) & (spectral_data['frequencies'] <= fmax)
                                        abs_power = np.sum(spectral_data['psd'][mask, :], axis=0)
                                        rel_power = abs_power / total_power
                                        
                                        for ch in range(len(channel_names)):
                                            channel_results[channel_names[ch]][f'{band_name}_abs_power'].append(abs_power[ch])
                                            channel_results[channel_names[ch]][f'{band_name}_rel_power'].append(rel_power[ch])
                                            
                            except Exception as e:
                                logging.error(f"Error calculating power measures for epoch {i+1}: {str(e)}")
                                for band in ['delta', 'theta', 'alpha', 'beta1', 'beta2']:
                                    power_values[f'{band}_abs_power'].append(np.nan)
                                    power_values[f'{band}_rel_power'].append(np.nan)
                        
                        # Calculate peak frequency if requested
                        if calc_peak:
                            try:
                                peak_freqs = calculate_avg_peak_frequency(
                                    frequencies=spectral_data['frequencies'],
                                    psd=spectral_data['psd'],
                                    freq_range=(peak_min, peak_max)
                                )
                                
                                if save_channel_averages:
                                    for ch in range(len(channel_names)):
                                        channel_results[channel_names[ch]]['peak_frequency'].append(peak_freqs[ch])
                                
                                n_channels_without_peak = np.sum(np.isnan(peak_freqs))
                                power_values['peak_frequency'].append(np.nanmean(peak_freqs))
                                power_values['channels_without_peak'].append(n_channels_without_peak)
                                
                            except Exception as e:
                                logging.error(f"Error calculating peak frequency: {str(e)}")
                                power_values['peak_frequency'].append(np.nan)
                                power_values['channels_without_peak'].append(np.nan)
                        
                        # Calculate spectral variability if requested
                        if calc_sv:
                            try:
                                sv_results = calculate_spectral_variability(
                                    times=spectral_data['times'],
                                    spectrogram=spectral_data['spectrogram'],
                                    frequencies=spectral_data['frequencies']
                                )
                                
                                if sv_results:
                                    if save_channel_averages:
                                        for band_name, values in sv_results.items():
                                            band_key = f'sv_{band_name}'
                                            for ch in range(len(channel_names)):
                                                channel_results[channel_names[ch]][band_key].append(values[ch])
                                    
                                    for band_name, values in sv_results.items():
                                        band_key = f'sv_{band_name}'
                                        if band_key not in sv_values:
                                            sv_values[band_key] = []
                                        sv_values[band_key].append(np.nanmean(values))
                                        
                            except Exception as e:
                                logging.error(f"Error calculating spectral variability: {str(e)}")
                                for band in ['delta', 'theta', 'alpha', 'beta1', 'beta2']:
                                    sv_values[f'sv_{band}'].append(np.nan)
                        
                        # Clean up spectral data
                        del spectral_data
                        
                    except Exception as e:
                        logging.error(f"Error in spectral calculations: {str(e)}")
                        # Set all spectral measures to NaN
                        if calc_power:
                            for band in ['delta', 'theta', 'alpha', 'beta1', 'beta2']:
                                power_values[f'{band}_abs_power'].append(np.nan)
                                power_values[f'{band}_rel_power'].append(np.nan)
                        if calc_peak:
                            power_values['peak_frequency'].append(np.nan)
                            power_values['channels_without_peak'].append(np.nan)
                        if calc_sv:
                            for band in ['delta', 'theta', 'alpha', 'beta1', 'beta2']:
                                sv_values[f'sv_{band}'].append(np.nan)
                
                # Calculate JPE and PE
                if calc_jpe:
                    if convert_ints_pe:
                        data_values_pe = convert_to_integers(data_values)
                    else:
                        data_values_pe = data_values.copy()
                    
                    jpe_matrix = calculate_jpe(data_values_pe, n=4, st=jpe_st, invert=invert)
                    if save_matrices:
                        if avg_matrices['jpe'] is None:
                            avg_matrices['jpe'] = jpe_matrix
                        else:
                            avg_matrices['jpe'] += jpe_matrix
                    
                    mask = ~np.eye(jpe_matrix.shape[0], dtype=bool)
                    jpe_values.append(jpe_matrix[mask].mean())
                    
                    pe_values_array = calculate_pe(data_values_pe, n=4, st=jpe_st)
                    pe_values.append(pe_values_array.mean())
                    
                    if save_channel_averages:
                        for ch in range(len(channel_names)):
                            channel_results[channel_names[ch]]['pe'].append(pe_values_array[ch])
                            channel_jpe = np.mean(jpe_matrix, axis=1)
                            channel_results[channel_names[ch]]['jpe'].append(channel_jpe[ch])
                    
                    if 'data_values_pe' in locals(): 
                        del data_values_pe
                
                # Calculate PLI if requested
                if calc_pli:
                    try:
                        pli_matrix = calculate_pli(data_values)
                        if save_matrices or (save_mst and calc_pli_mst):
                            if avg_matrices['pli'] is None:
                                avg_matrices['pli'] = pli_matrix
                            else:
                                avg_matrices['pli'] += pli_matrix
                                
                        mask = ~np.eye(pli_matrix.shape[0], dtype=bool)
                        pli_values.append(pli_matrix[mask].mean())
                        
                        if save_channel_averages:
                            channel_pli = np.mean(pli_matrix, axis=1)
                            for ch in range(len(channel_names)):
                                channel_results[channel_names[ch]]['pli'].append(channel_pli[ch])
                        
                        if calc_pli_mst:
                            try:
                                mst_measures, mst_matrix, success = calculate_mst_measures(pli_matrix)
                                if success:
                                    for measure, value in mst_measures.items():
                                        pli_mst_values[measure].append(value)
                            except Exception as e:
                                logging.error(f"Error calculating PLI MST measures for epoch {i+1}: {str(e)}")
                                for measure in ['degree', 'eccentr', 'betweenness', 'kappa', 'r', 
                                              'diameter', 'leaf', 'hierarchy', 'teff', 'asp', 
                                              'ref', 'mean']:
                                    pli_mst_values[measure].append(np.nan)
                                                
                    except Exception as e:
                        logging.error(f"Error calculating PLI: {str(e)}")
                        pli_values.append(np.nan)
                
                # Calculate AEC/AECc if requested
                if calc_aec:
                    try:
                        if concat_aecc:
                            logging.info(f"Processing concatenated AEC{'c' if use_aecc else ''} for {subject} - {condition}")
                            # Initialize list to store all epochs
                            all_data = []
                            data_values = None  # Initialize data_values in this scope
                                            
                            try:
                                # Read and store all epochs with offset correction
                                for file_path in epoch_files:
                                    try:
                                        data = pd.read_csv(file_path, sep=None, engine='python')
                                        epoch_data = data.values
                                        # Apply offset correction for each channel
                                        epoch_data = epoch_data - np.mean(epoch_data, axis=0)
                                        all_data.append(epoch_data)
                                        del data, epoch_data
                                    except Exception as e:
                                        logging.error(f"Error processing file {os.path.basename(file_path)}: {str(e)}")
                                        continue
                                
                                if all_data:  # Check if we have any valid data
                                    # Concatenate along time axis
                                    data_values = np.concatenate(all_data, axis=0)
                                else:
                                    raise ValueError("No valid epochs could be processed")
                            
                            finally:
                                # Clean up interim data
                                del all_data
                            
                            # Now use data_values for AEC calculation
                            aec_matrix = calculate_aecc(data_values, orthogonalize=use_aecc, force_positive=force_positive)
                
                            if save_matrices:
                                avg_matrices['aec'] = aec_matrix
                            
                            mask = ~np.eye(aec_matrix.shape[0], dtype=bool)
                            aec_values = [aec_matrix[mask].mean()]  # Single value
                            
                            # Calculate MST if requested - do it once and store results
                            mst_results = None
                            if calc_aec_mst:
                                try:
                                    logging.info(f"Calculating MST on concatenated AEC{'c' if use_aecc else ''} matrix")
                                    mst_measures, mst_matrix, success = calculate_mst_measures(aec_matrix)
                                    
                                    if success:
                                        successful_mst_epochs = 1
                                        mst_results = (mst_measures, mst_matrix, success)  # Store for later use
                                        
                                        # Store single values for MST measures
                                        for measure in ['degree', 'eccentr', 'betweenness', 'kappa', 'r', 
                                                      'diameter', 'leaf', 'hierarchy', 'teff', 'asp', 
                                                      'ref', 'mean']:
                                            aec_mst_values[measure] = [mst_measures[measure]]
                                        
                                        # Store MST matrix if needed
                                        if save_mst:
                                            mst_matrix_symmetric = mst_matrix + mst_matrix.T
                                            avg_matrices['aec_mst'] = mst_matrix_symmetric
                                            logging.info(f"Stored MST matrix for concatenated data")
                                    else:
                                        successful_mst_epochs = 0
                                        logging.warning(f"Failed to calculate MST on concatenated matrix")
                                except Exception as e:
                                    successful_mst_epochs = 0
                                    logging.error(f"Error calculating MST on concatenated matrix: {str(e)}")
                            
                            if save_channel_averages:
                                channel_aec = np.mean(aec_matrix, axis=1)
                                for ch in range(len(channel_names)):
                                    channel_results[channel_names[ch]]['aec'] = [channel_aec[ch]]
                
                        else:
                            # Original epoch-by-epoch processing (keep existing code)
                            aec_matrix = calculate_aecc(data_values, orthogonalize=use_aecc, force_positive=force_positive)
                            if save_matrices or (save_mst and calc_aec_mst):
                                if avg_matrices['aec'] is None:
                                    avg_matrices['aec'] = aec_matrix
                                else:
                                    avg_matrices['aec'] += aec_matrix
                                    
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
                                except Exception as e:
                                    logging.error(f"Error calculating MST measures for epoch: {str(e)}")
                                    for measure in ['degree', 'eccentr', 'betweenness', 'kappa', 'r', 
                                                  'diameter', 'leaf', 'hierarchy', 'teff', 'asp', 
                                                  'ref', 'mean']:
                                        aec_mst_values[measure].append(np.nan)
                
                    except Exception as e:
                        logging.error(f"Error calculating AEC{'c' if use_aecc else ''}: {str(e)}")
                        aec_values.append(np.nan)
                        
                # Calculate complexity measures                                                                                  
                if calc_sampen:
                    try:
                        sampen_values_ch = calculate_sampen_for_channels(data_values, m=sampen_m)
                        
                        if save_channel_averages:
                            for ch in range(len(channel_names)):
                                channel_results[channel_names[ch]]['sampen'].append(sampen_values_ch[ch])
                        
                        sampen_values.append(np.nanmean(sampen_values_ch))  # Value per epoch
                        logging.info(f"Successfully calculated SampEn for epoch")
                    except Exception as e:
                        logging.error(f"Error in SampEn calculation: {str(e)}")
                        sampen_values.append(np.nan)
                        
                if calc_apen:
                    try:
                        apen_values_ch = calculate_apen_for_channels(data_values, m=apen_m, r=apen_r)
                        
                        if save_channel_averages:
                            for ch in range(len(channel_names)):
                                channel_results[channel_names[ch]]['apen'].append(apen_values_ch[ch])
                        
                        apen_values.append(np.nanmean(apen_values_ch))  # Value per epoch
                        logging.info(f"Successfully calculated ApEn for epoch")
                    except Exception as e:
                        logging.error(f"Error in ApEn calculation: {str(e)}")
                        apen_values.append(np.nan)
                                                
            except Exception as e:
                logging.error(f"Error processing file {os.path.basename(file_path)}: {str(e)}")
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
                if not (concat_aecc and key == 'aec'):
                    avg_matrices[key] /= n_epochs
                    logging.info(f"Normalized {key} connectivity matrix by {n_epochs} epochs")
            
        # Now calculate MSTs from averaged connectivity matrices if needed
        if save_mst:
            # Handle PLI MST
            if calc_pli_mst and avg_matrices['pli'] is not None:
                try:
                    mst_measures, mst_matrix, success = calculate_mst_measures(avg_matrices['pli'])
                    if success:
                        mst_matrix_symmetric = mst_matrix + mst_matrix.T
                        avg_matrices['pli_mst'] = mst_matrix_symmetric
                
                        logging.info(f"Calculated PLI MST from averaged connectivity matrix for {subject}-{condition}")
                    else:
                        logging.warning(f"Could not calculate PLI MST from averaged matrix for {subject}-{condition}")
                except Exception as e:
                    logging.error(f"Error calculating PLI MST from averaged matrix: {str(e)}")
            
        # Prepare results dictionary
        results = {
            'avg_jpe': np.mean(jpe_values) if jpe_values and calc_jpe else np.nan,
            'avg_pe': np.mean(pe_values) if pe_values and calc_jpe else np.nan,
            'avg_pli': np.mean(pli_values) if pli_values and calc_pli else np.nan,
            'avg_aec': np.mean(aec_values) if aec_values and calc_aec else np.nan,
            'avg_sampen': np.mean(sampen_values) if sampen_values and calc_sampen else np.nan,
            'avg_apen': np.mean(apen_values) if apen_values and calc_apen else np.nan, 
            'n_epochs': len(epoch_files),
            'channel_names': channel_names if channel_names else [],
            'matrices': avg_matrices if (save_matrices or save_mst) else None,
            'channel_averages': channel_averages
        }
        
        # Add MST results and tracking metrics
        if calc_aec_mst:
            for measure in ['degree', 'eccentr', 'betweenness', 'kappa', 'r', 
                          'diameter', 'leaf', 'hierarchy', 'teff', 'asp', 
                          'ref', 'mean']:
                if concat_aecc:
                    # For concatenated case, just store the single value
                    if aec_mst_values[measure]:
                        results[f'aec_mst_{measure}'] = aec_mst_values[measure][0]  # Single value
                        results[f'aec_mst_{measure}_valid_epochs'] = 1 if successful_mst_epochs else 0
                    else:
                        results[f'aec_mst_{measure}'] = np.nan
                        results[f'aec_mst_{measure}_valid_epochs'] = 0
                else:
                    # Original epoch-by-epoch processing
                    if aec_mst_values[measure]:  # Only calculate mean if we have valid values
                        results[f'aec_mst_{measure}'] = np.mean(aec_mst_values[measure])
                        results[f'aec_mst_{measure}_valid_epochs'] = len(aec_mst_values[measure])
                    else:
                        results[f'aec_mst_{measure}'] = np.nan
                        results[f'aec_mst_{measure}_valid_epochs'] = 0
            
            # Add the tracking metrics
            results['aec_mst_successful_epochs'] = successful_mst_epochs
            results['aec_mst_total_epochs'] = 1 if concat_aecc else len(epoch_files)
            
            # Add concatenation info to results for reference
            results['aec_concatenated'] = concat_aecc
                    
        if calc_pli_mst:
            for measure in ['degree', 'eccentr', 'betweenness', 'kappa', 'r', 
                          'diameter', 'leaf', 'hierarchy', 'teff', 'asp', 
                          'ref', 'mean']:
                if pli_mst_values[measure]:
                    results[f'pli_mst_{measure}'] = np.mean(pli_mst_values[measure])
                    results[f'pli_mst_{measure}_valid_epochs'] = len(pli_mst_values[measure])
                else:
                    results[f'pli_mst_{measure}'] = np.nan
                    results[f'pli_mst_{measure}_valid_epochs'] = 0
        
        if calc_power and power_values:
            for band in ['delta', 'theta', 'alpha', 'beta1', 'beta2']:
                results[f'{band}_abs_power'] = np.mean(power_values[f'{band}_abs_power'])
                results[f'{band}_rel_power'] = np.mean(power_values[f'{band}_rel_power'])
        
        if calc_peak:
            results['peak_frequency'] = np.mean(power_values['peak_frequency'])
            results['channels_without_peak'] = np.mean(power_values['channels_without_peak'])
            
        if calc_sv and sv_values:
            for band in ['delta', 'theta', 'alpha', 'beta1', 'beta2']:
                band_key = f'sv_{band}'
                results[band_key] = np.mean(sv_values[band_key]) if band_key in sv_values and sv_values[band_key] else np.nan
                
        return subject, condition, results
            
    except Exception as e:
        logging.error(f"Error processing {subject} - {condition}: {str(e)}")
        
        # Define error_result dictionary
        error_result = {
            'avg_jpe': np.nan,
            'avg_pe': np.nan,
            'avg_pli': np.nan,
            'avg_aec': np.nan,
            'avg_apen': np.nan,
            'avg_sampen': np.nan,
            'n_epochs': 0,
            'channel_names': [],
            'matrices': None if save_matrices else None,
            'channel_averages': None
        }
        
        # Add MST measures to error result if needed
        if calc_aec_mst:
            for measure in ['degree', 'eccentr', 'betweenness', 'kappa', 'r', 
                          'diameter', 'leaf', 'hierarchy', 'teff', 'asp', 
                          'ref', 'mean']:
                error_result[f'aec_mst_{measure}'] = np.nan
                error_result[f'aec_mst_{measure}_valid_epochs'] = 0
            error_result['aec_mst_successful_epochs'] = 0
            error_result['aec_mst_total_epochs'] = len(epoch_files)
            
        if calc_pli_mst:
            for measure in ['degree', 'eccentr', 'betweenness', 'kappa', 'r', 
                          'diameter', 'leaf', 'hierarchy', 'teff', 'asp', 
                          'ref', 'mean']:
                error_result[f'pli_mst_{measure}'] = np.nan
                error_result[f'pli_mst_{measure}_valid_epochs'] = 0

        if calc_power:
            for band in ['delta', 'theta', 'alpha', 'beta1', 'beta2']:
                error_result[f'{band}_abs_power'] = np.nan
                error_result[f'{band}_rel_power'] = np.nan
        if calc_peak:
            error_result['peak_frequency'] = np.nan
            error_result['channels_without_peak'] = np.nan
        
        # Add spectral variability measures to error result if needed
        if calc_sv:
            for band in ['delta', 'theta', 'alpha', 'beta1', 'beta2']:
                error_result[f'sv_{band}'] = np.nan
        
        return subject, condition, error_result

def process_batch(batch_args, n_threads):
   """Process a batch of subjects using multiprocessing with fallback"""
   try:
       with Pool(processes=n_threads, maxtasksperchild=1) as pool:
           results = []
           for result in pool.imap_unordered(process_subject_condition, batch_args):
               results.append(result)
           return results
   except Exception as e:
       logging.error(f"Pool processing failed: {str(e)}, falling back to single thread")
       return [process_subject_condition(args) for args in batch_args]
   finally:
       if 'pool' in locals():
           pool.terminate()
           pool.join()
   
def group_epochs_by_condition(folder_path, folder_ext):
    """
    Group epoch files by their base name and condition
    Only processes folders containing valid epoch files
    Returns a dictionary: {base_name: {condition: [epoch_files]}}
    """
    grouped_files = defaultdict(lambda: defaultdict(list))
    
    # Get immediate subdirectories
    try:
        subdirs = [d for d in os.listdir(folder_path) 
                  if os.path.isdir(os.path.join(folder_path, d)) and 
                  d.endswith(folder_ext)]
    except Exception as e:
        sg.popup_error(f"Error accessing directory: {str(e)}")
        return grouped_files
    
    if not subdirs:
        sg.popup_warning(f"No folders ending with '{folder_ext}' found in the selected directory.")
        return grouped_files
    
    for subdir in subdirs:
        subdir_path = os.path.join(folder_path, subdir)
        
        # Look for epoch files in this directory
        for file in os.listdir(subdir_path):            
            if file.startswith('.') or file.startswith('._'):
                continue
                
            if not file.endswith('.txt'):
                continue
                
            # Check if file matches epoch pattern
            if '_level_' in file and '_Epoch_' in file:
                try:
                    file_info = parse_epoch_filename(file)
                    full_path = os.path.join(subdir_path, file)
                    base_name = subdir.replace(folder_ext, '')  # Use folder name without extension
                    condition = file_info['condition']
                    grouped_files[base_name][condition].append(full_path)
                except Exception as e:
                    print(f"Skipping file {file}: {str(e)}")
                    continue
    
    # Print summary of what was found
    print("\nFound the following data:")
    for base_name, conditions in grouped_files.items():
        print(f"\nSubject: {base_name}")
        for condition, files in conditions.items():
            print(f"  {condition}: {len(files)} epochs")
    
    return grouped_files
            
def process_all_subjects(grouped_files, convert_ints_pe, invert, n_threads, 
                        calc_jpe, calc_pli, calc_pli_mst, calc_aec, use_aecc, 
                        force_positive=True, jpe_st=1, calc_aec_mst=False, 
                        calc_power=False, power_fs=256, calc_peak=False, 
                        peak_min=3, peak_max=13, calc_sampen=False, sampen_m=2,
                        calc_apen=False, apen_m=1, apen_r=0.25,
                        calc_sv=False, sv_window=1000, 
                        save_matrices=False, save_mst=False, save_channel_averages=False,
                        concat_aecc=False, has_headers=True,
                        progress_callback=None):
    
    process_args = []
    for subject, conditions in grouped_files.items():
        for condition, epoch_files in conditions.items():                        
            process_args.append((subject, condition, epoch_files, 
                                convert_ints_pe, invert, calc_jpe, calc_pli, calc_pli_mst,
                                calc_aec, use_aecc, force_positive, jpe_st, calc_aec_mst, 
                                calc_power, power_fs, calc_peak, peak_min, peak_max,
                                calc_sampen, sampen_m, calc_apen, apen_m, apen_r,
                                calc_sv, sv_window, save_matrices, save_mst, 
                                save_channel_averages, concat_aecc, has_headers))

    total_tasks = len(process_args)
    logging.info(f"Starting processing of {total_tasks} subject-condition combinations")
    logging.info(f"Processing options: JPE/PE calculation={calc_jpe}, JPE invert={invert}, "
                    f"PE integer conversion={convert_ints_pe}, PLI calculation={calc_pli}, "
                    f"PLI MST calculation={calc_pli_mst}, AEC calculation={calc_aec}, "
                    f"AECc={use_aecc}, AEC concatenation={concat_aecc}, "
                    f"Force positive AEC={force_positive}, Save MST matrices={save_mst}")
    
    # Initialize results
    results = defaultdict(dict)
    completed = 0
    
    # Process in batches
    for i in range(0, len(process_args), BATCH_SIZE):
        batch = process_args[i:i + BATCH_SIZE]
        batch_size = len(batch)
        
        logging.info(f"Processing batch {i//BATCH_SIZE + 1} of {(total_tasks + BATCH_SIZE - 1)//BATCH_SIZE} "
                    f"({batch_size} combinations)")
        start_time = time.time()
        
        try:
            # Process batch
            batch_results = process_batch(batch, n_threads)
            
            # Update results and log MST statistics
            for subject, condition, result in batch_results:
                results[subject][condition] = result
                completed += 1
                
                # Log MST success rates if applicable
                if calc_aec_mst and 'aec_mst_successful_epochs' in result:
                    success_rate = (result['aec_mst_successful_epochs'] / 
                                  result['aec_mst_total_epochs'] * 100 
                                  if result['aec_mst_total_epochs'] > 0 else 0)
                    logging.info(f"{subject} - {condition}: MST success rate: "
                               f"{success_rate:.1f}% ({result['aec_mst_successful_epochs']}/"
                               f"{result['aec_mst_total_epochs']} epochs)")
                
                if progress_callback:
                    progress_callback(completed / total_tasks * 100)
            
            batch_time = time.time() - start_time
            avg_time_per_combo = batch_time / batch_size
            remaining_time = (total_tasks - completed) * avg_time_per_combo
            
            logging.info(f"Batch completed in {batch_time:.1f} seconds "
                        f"({avg_time_per_combo:.1f} sec/combination). "
                        f"Estimated remaining time: {remaining_time/60:.1f} minutes")
            
        except Exception as e:
            logging.error(f"Error processing batch: {str(e)}")
            continue
        
        # Memory check and cleanup
        if MemoryMonitor.check_memory():
            logging.warning("High memory usage detected - triggering garbage collection")
            import gc
            gc.collect()
            time.sleep(1)
    
    # Final processing summary
    logging.info(f"Processing completed: {completed}/{total_tasks} combinations processed")
    if completed < total_tasks:
        logging.warning(f"Some combinations ({total_tasks - completed}) failed to process")
    
    return dict(results)

def save_results_to_excel(results_dict, output_path, invert, calc_pli_mst, calc_jpe=True, 
                         calc_pli=True, calc_aec=False, use_aecc=False, force_positive=True, 
                         calc_aec_mst=False, calc_power=False, power_fs=256, calc_peak=False,
                         peak_min=None, peak_max=None, calc_sampen=False, calc_apen=False, calc_sv=False, 
                         save_channel_averages=False, concat_aecc=False, has_headers=True, sv_window=None):
    """Save results to Excel with organized columns by condition"""
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Save whole-brain averages
        all_conditions = set()
        for subject_data in results_dict.values():
            all_conditions.update(subject_data.keys())
        
        columns = ['subject']
        measure_name = "jpe_inv" if invert else "jpe"
        
        # Create column names for each condition
        for condition in sorted(all_conditions):
            if calc_jpe:
                columns.extend([
                    f'{condition}_avg_{measure_name}',
                    f'{condition}_avg_pe',
                ])
            
            if calc_pli:
                columns.append(f'{condition}_avg_pli')
            
            if calc_sampen:
                columns.append(f'{condition}_avg_sampen')
                
            if calc_apen:
                columns.append(f'{condition}_avg_apen')
            
            if calc_aec:
                columns.append(f'{condition}_avg_aec')
                
                # Add AEC MST measures columns
                if calc_aec_mst:
                    mst_measures = ['degree', 'eccentr', 'betweenness', 'kappa', 'r', 
                                  'diameter', 'leaf', 'hierarchy', 'teff', 'asp', 
                                  'ref', 'mean']
                    for measure in mst_measures:
                        columns.append(f'{condition}_aec_mst_{measure}')
                    # Add validation columns
                    columns.append(f'{condition}_aec_mst_successful_epochs')
                    columns.append(f'{condition}_aec_mst_total_epochs')
            
            columns.append(f'{condition}_n_epochs')
            
            # Add power-related columns for broadband condition
            if calc_power and '0.5-47' in condition:
                for band in ['delta', 'theta', 'alpha', 'beta1', 'beta2']:
                    columns.extend([
                        f'{condition}_{band}_abs_power',
                        f'{condition}_{band}_rel_power'
                    ])
            if calc_peak and '0.5-47' in condition:
                columns.extend([
                    f'{condition}_peak_frequency',
                    f'{condition}_channels_without_peak'
                ])
            if calc_sv and '0.5-47' in condition:
                for band in ['delta', 'theta', 'alpha', 'beta1', 'beta2']:
                    columns.append(f'{condition}_sv_{band}')
        
        # Create rows
        rows = []
        for subject, conditions in results_dict.items():
            row = {'subject': subject}
            for condition in sorted(all_conditions):
                if condition in conditions:
                    if calc_jpe:
                        row[f'{condition}_avg_{measure_name}'] = conditions[condition]['avg_jpe']
                        row[f'{condition}_avg_pe'] = conditions[condition]['avg_pe']
                    
                    if calc_pli:
                        row[f'{condition}_avg_pli'] = conditions[condition].get('avg_pli', np.nan)
                    
                    if calc_sampen:
                        row[f'{condition}_avg_sampen'] = conditions[condition].get('avg_sampen', np.nan)
                        
                    if calc_apen:
                        row[f'{condition}_avg_apen'] = conditions[condition].get('avg_apen', np.nan)
                    
                    if calc_aec:
                        row[f'{condition}_avg_aec'] = conditions[condition]['avg_aec']
                        
                        # Add AEC MST measures
                        if calc_aec_mst:
                            mst_measures = ['degree', 'eccentr', 'betweenness', 'kappa', 'r', 
                                          'diameter', 'leaf', 'hierarchy', 'teff', 'asp', 
                                          'ref', 'mean']
                            for measure in mst_measures:
                                key = f'aec_mst_{measure}'
                                row[f'{condition}_aec_mst_{measure}'] = conditions[condition].get(key, np.nan)
                            
                            # Add validation info
                            row[f'{condition}_aec_mst_successful_epochs'] = conditions[condition].get('aec_mst_successful_epochs', 0)
                            row[f'{condition}_aec_mst_total_epochs'] = conditions[condition].get('aec_mst_total_epochs', 0)
                    
                    row[f'{condition}_n_epochs'] = conditions[condition]['n_epochs']
                    
                    # Add power-related values for broadband condition
                    if calc_power and '0.5-47' in condition:
                        for band in ['delta', 'theta', 'alpha', 'beta1', 'beta2']:
                            row[f'{condition}_{band}_abs_power'] = conditions[condition].get(f'{band}_abs_power', np.nan)
                            row[f'{condition}_{band}_rel_power'] = conditions[condition].get(f'{band}_rel_power', np.nan)
                    if calc_peak and '0.5-47' in condition:
                        row[f'{condition}_peak_frequency'] = conditions[condition].get('peak_frequency', np.nan)
                        row[f'{condition}_channels_without_peak'] = conditions[condition].get('channels_without_peak', np.nan)
                    if calc_sv and '0.5-47' in condition:
                        for band in ['delta', 'theta', 'alpha', 'beta1', 'beta2']:
                            row[f'{condition}_sv_{band}'] = conditions[condition].get(f'sv_{band}', np.nan)
                else:
                    # Set NaN values for missing conditions
                    if calc_jpe:
                        row[f'{condition}_avg_{measure_name}'] = np.nan
                        row[f'{condition}_avg_pe'] = np.nan
                    
                    if calc_pli:
                        row[f'{condition}_avg_pli'] = np.nan
                    
                    if calc_sampen:
                        row[f'{condition}_avg_sampen'] = np.nan
                        
                    if calc_apen:
                        row[f'{condition}_avg_apen'] = np.nan
                    
                    if calc_aec:
                        row[f'{condition}_avg_aec'] = np.nan
                        
                        # Set NaN for MST measures
                        if calc_aec_mst:
                            for measure in ['degree', 'eccentr', 'betweenness', 'kappa', 'r', 
                                          'diameter', 'leaf', 'hierarchy', 'teff', 'asp', 
                                          'ref', 'mean']:
                                row[f'{condition}_aec_mst_{measure}'] = np.nan
                            row[f'{condition}_aec_mst_successful_epochs'] = 0
                            row[f'{condition}_aec_mst_total_epochs'] = 0
                    
                    row[f'{condition}_n_epochs'] = 0
                    
                    # Set NaN for power-related values
                    if calc_power and '0.5-47' in condition:
                        for band in ['delta', 'theta', 'alpha', 'beta1', 'beta2']:
                            row[f'{condition}_{band}_abs_power'] = np.nan
                            row[f'{condition}_{band}_rel_power'] = np.nan
                    if calc_peak and '0.5-47' in condition:
                        row[f'{condition}_peak_frequency'] = np.nan
                        row[f'{condition}_channels_without_peak'] = np.nan
                    if calc_sv and '0.5-47' in condition:
                        for band in ['delta', 'theta', 'alpha', 'beta1', 'beta2']:
                            row[f'{condition}_sv_{band}'] = np.nan
                            
            rows.append(row)
        
        # Create DataFrame and save
        df = pd.DataFrame(rows)
        df = df[columns]  # Reorder columns
        df.to_excel(writer, sheet_name='Whole Brain Results', index=False)
        
        # Add analysis information sheet with sampling frequency info and spectral variability window
        info_data = {
            'Parameter': [
                'Analysis Date',
                'JPE Inversion',
                'PLI MST Calculated',
                'AEC Type',
                'AEC Concatenated Epochs',
                'AEC MST Calculated',
                'AEC Force Positive',
                'Power Bands Calculated',
                'Sampling Frequency (Hz)',  
                'Peak Frequency Analysis', 
                'Peak Frequency Range (Hz)', 
                'Sample Entropy Calculated',
                'Approximate Entropy Calculated',
                'Spectral Variability Calculated',
                'Spectral Variability Window (ms)',
                'Channel Averages Calculated',
                'Channel Names Source'
            ],
            'Value': [
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'Yes' if invert else 'No',
                'Yes' if calc_pli_mst else 'No',
                'AECc (orthogonalized)' if calc_aec and use_aecc else 'AEC' if calc_aec else 'Not calculated',
                'Yes' if concat_aecc else 'No',
                'Yes' if calc_aec_mst else 'No',
                'Yes' if force_positive else 'No',
                'Yes' if calc_power else 'No',
                str(power_fs),
                'Yes' if calc_peak else 'No',
                f"{calc_peak and f'{peak_min}-{peak_max}' or 'N/A'}", 
                'Yes' if calc_sampen else 'No',
                'Yes' if calc_apen else 'No',
                'Yes' if calc_sv else 'No',
                str(sv_window) if calc_sv else 'N/A',
                'Yes' if save_channel_averages else 'No',
                'File Headers' if has_headers else 'Auto-generated'
            ]
        }
        
        info_df = pd.DataFrame(info_data)
        info_df.to_excel(writer, sheet_name='Analysis Information', index=False)
        
        # Save channel-level averages if requested
        if save_channel_averages:
            # First, gather all unique channels across all conditions
            all_channels = set()
            for subject, conditions in results_dict.items():
                for condition, result in conditions.items():
                    if result.get('channel_averages'):
                        all_channels.update(result['channel_averages'].keys())
            
            # Create rows with separate columns for each condition
            channel_rows = []
            # Keep track of which measures actually have data
            measures_with_data = set()
            
            for subject, conditions in results_dict.items():
                for channel in sorted(all_channels):
                    row = {
                        'subject': subject,
                        'channel': channel
                    }
                    
                    # For each condition, add all its measures
                    for condition in sorted(all_conditions):
                        if condition in conditions and conditions[condition].get('channel_averages'):
                            channel_data = conditions[condition]['channel_averages'].get(channel, {})
                            # Add each measure with condition prefix
                            for measure, value in channel_data.items():
                                column_name = f'{condition}_{measure}'
                                row[column_name] = value
                                if not pd.isna(value):  # Only track columns that have actual data
                                    measures_with_data.add(column_name)
                    
                    channel_rows.append(row)
            
            if channel_rows:
                df_channels = pd.DataFrame(channel_rows)
                
                # Keep only columns that have data
                base_cols = ['subject', 'channel']
                data_cols = sorted(measures_with_data, key=lambda x: (x.split('_')[0], x))
                
                # Final column order
                column_order = base_cols + data_cols
                df_channels = df_channels[column_order]
                df_channels.to_excel(writer, sheet_name='Channel Averages', index=False)
        
        # Save metadata about channels
        metadata_rows = []
        for subject, conditions in results_dict.items():
            for condition in sorted(all_conditions):
                if condition in conditions and 'channel_names' in conditions[condition]:
                    metadata_rows.append({
                        'subject': subject,
                        'condition': condition,
                        'channels': ', '.join(conditions[condition]['channel_names']),
                        'n_channels': len(conditions[condition]['channel_names'])
                    })
        
        if metadata_rows:
            metadata_df = pd.DataFrame(metadata_rows)
            metadata_df.to_excel(writer, sheet_name='Channel Information', index=False)
    
    logging.info(f"Results saved to: {output_path}")
    print(f"\nResults saved to {output_path}")
    
def main():
    window = create_gui()
    
    while True:
        event, values = window.read()
        
        if event == sg.WIN_CLOSED or event == "Exit":
            break
            
        if event == "Process":
            folder_path = values["-FOLDER-"]
            folder_ext = values["-EXTENSION-"].strip()
            
            # Setup logging first thing
            log_file = setup_logging(folder_path)
            logging.info("=== Starting new analysis run ===")
            logging.info(f"Folder path: {folder_path}")
            logging.info(f"Extension: {folder_ext}")
            logging.info(f"Processing files with{'out' if not values['-HAS_HEADERS-'] else ''} headers")
            if not values['-HAS_HEADERS-']:
                logging.info("Channel names will be auto-generated")
            
            try:
                n_threads = int(values["-THREADS-"])
                if n_threads < 1 or n_threads > cpu_count():
                    raise ValueError(f"Number of threads must be between 1 and {cpu_count()}")
            except ValueError as e:
                sg.popup_error(f"Invalid number of threads: {str(e)}")
                continue
            
            if not folder_path or not folder_ext:
                sg.popup_error("Please select a folder and specify folder extension")
                continue
            
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
                        raise ValueError("Time step must be greater than 0")
                except ValueError as e:
                    sg.popup_error(f"Invalid time step value: {str(e)}")
                    continue
                
                # Validate power sampling frequency
                try:
                    power_fs = float(values["-POWER_FS-"])
                    if power_fs <= 0:
                        raise ValueError("Sampling frequency must be greater than 0")
                except ValueError as e:
                    sg.popup_error(f"Invalid sampling frequency value: {str(e)}")
                    continue
                
                # Validate peak frequency range
                peak_min = peak_max = None
                if values["-CALC_PEAK-"]:
                    try:
                        peak_min = float(values["-PEAK_MIN-"])
                        peak_max = float(values["-PEAK_MAX-"])
                        if peak_min >= peak_max:
                            raise ValueError("Minimum frequency must be less than maximum")
                        if peak_min < 0 or peak_max > (power_fs/2):
                            raise ValueError(f"Frequency range must be between 0 and {power_fs/2} Hz")
                    except ValueError as e:
                        sg.popup_error(f"Invalid peak frequency range: {str(e)}")
                        continue
                    
                # Validate SampEn parameters
                sampen_m = None
                if values["-CALC_SAMPEN-"]:
                    try:
                        sampen_m = int(values["-SAMPEN_M-"])
                        if sampen_m < 1:
                            raise ValueError("Order m must be greater than 0")
                    except ValueError as e:
                        sg.popup_error(f"Invalid SampEn order parameter: {str(e)}")
                        continue
                            
                # Validate ApEn parameters
                apen_m = apen_r = None
                    
                if values["-CALC_APEN-"]:
                    try:
                        apen_m = int(values["-APEN_M-"])
                        if apen_m < 1:
                            raise ValueError("Order m must be greater than 0")
                            
                        apen_r = float(values["-APEN_R-"])
                        if apen_r <= 0:
                            raise ValueError("Tolerance r must be greater than 0")
                    except ValueError as e:
                        sg.popup_error(f"Invalid ApEn parameter: {str(e)}")
                        continue
                                
                # Validate spectral variability window
                sv_window = None
                if values["-CALC_SV-"]:
                    try:
                        sv_window = int(values["-SV_WINDOW-"])
                        if sv_window < 100:
                            raise ValueError("Window length must be at least 100ms")
                    except ValueError as e:
                        sg.popup_error(f"Invalid spectral variability window: {str(e)}")
                        continue
                
                def update_progress(value):
                    window['-PROGRESS-'].update(value)
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
                    progress_callback=update_progress
                )
                
                if results:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_path = os.path.join(folder_path, f'EEG_analysis_{timestamp}.xlsx')
                    
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
                        )
                        # Save matrices if requested
                        matrices_saved = 0
                        mst_matrices_saved = 0
                        
                        logging.info("Starting matrix saving process...")
                        
                        if save_matrices or save_mst:
                            folders = create_matrix_folder_structure(
                                folder_path,
                                matrix_folder,
                                mst_folder if save_mst else None
                            )
                            
                            for subject, conditions in results.items():
                                for condition, result in conditions.items():
                                    if 'matrices' in result and result['matrices']:
                                        freq_band = extract_freq_band(condition)
                                        matrices = result['matrices']
                                        current_channel_names = result['channel_names']
                                        
                                        # Extract level type from condition
                                        level_type = 'source' if 'source' in condition.lower() else 'sensor'
                                        
                                        # Save regular connectivity matrices
                                        if save_matrices:
                                            for feature, matrix in matrices.items():
                                                if matrix is not None and feature in ['jpe', 'pli', 'aec']:
                                                    try:
                                                        if len(matrix) == len(current_channel_names):
                                                            filepath = save_connectivity_matrix(
                                                                matrix,
                                                                folders[feature],
                                                                subject,
                                                                freq_band,
                                                                feature,
                                                                current_channel_names,
                                                                level_type
                                                            )
                                                            matrices_saved += 1
                                                            logging.info(f"Saved {feature} matrix to: {filepath}")
                                                        else:
                                                            logging.error(f"Matrix dimension ({len(matrix)}) doesn't match channel count ({len(current_channel_names)})")
                                                    except Exception as e:
                                                        logging.error(f"Error saving {feature} matrix: {str(e)}")
                                        
                                        # Save MST matrices
                                        if save_mst:
                                            for mst_type, matrix_key in [('pli_mst', 'pli_mst'), ('aec_mst', 'aec_mst')]:
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
                                                                level_type
                                                            )
                                                            mst_matrices_saved += 1
                                                            logging.info(f"Saved {matrix_key} matrix to: {filepath}")
                                                        else:
                                                            logging.error(f"MST matrix dimension ({len(matrices[matrix_key])}) doesn't match channel count ({len(current_channel_names)})")
                                                    except Exception as e:
                                                        logging.error(f"Error saving {matrix_key} matrix: {str(e)}")
                                
                        if matrices_saved > 0:
                            logging.info(f"Saved {matrices_saved} connectivity matrices")
                        if mst_matrices_saved > 0:
                            logging.info(f"Saved {mst_matrices_saved} MST matrices")
                        
                        logging.info(f"Results saved to: {output_path}")
                        success_message = f"Analysis complete!\nResults saved to:\n{output_path}"
                        if save_matrices:
                            success_message += f"\nConnectivity matrices saved in: {matrix_folder}"
                        if save_mst:
                            success_message += f"\nMST matrices saved in: {mst_folder}"
                        success_message += f"\nLog file: {log_file}"
                        sg.popup(success_message)
                        
                    except Exception as e:
                        logging.error(f"Error saving results: {str(e)}")
                        sg.popup_error(f"Error saving results: {str(e)}")
                else:
                    logging.warning("No results were generated")
                    sg.popup_error("No results were generated")
                    
            except Exception as e:
                logging.error(f"Error during processing: {str(e)}")
                sg.popup_error(f"Error during processing: {str(e)}")
            
            finally:
                logging.info("Analysis run completed")
    
    window.close()

if __name__ == "__main__":
    main()
    
    
