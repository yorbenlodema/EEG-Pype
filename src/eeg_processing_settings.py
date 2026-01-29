import matplotlib as mpl
import mne

PySimpleGUI_License = "e1yWJaMdasWkN4l4b7nYNllfVqHolVwwZUS5IA6pIekqRAp7cf3FRNyZakWgJy1ldnGXlZv7bhi9Ihs0Ifkvxbp2YA2KVOuKct2IVPJHRsC1IZ6tMETZcDyzOjDDQI2KMEzVIM3WMoSXw2izTBGmlgjvZrWx5GzDZHUvR0lZcOGCxBvSeXWR1OlFbsnKRlWGZkXNJYzXaLWC9TuQIljdo8iaNpSW4GwlIiiBwZi0TOmBFwtvZnUCZrpQc3ntNF0IIcj3ozi6W3Wj9QycYnmfV7u4ImirwWiLTimEFutRZ8Ujxih2c238QuirOCi9JAMabp29R9lzbBWSE9i2LWCrJkDfbX2A1vwVYAWY5y50IEjGojizI4itwdikQk3XVdzJdnGF94tYZjXDJnJhRpCNIr6RIijsQVxDNgjmYQyvIki5wWitRgGVFK0BZqUol9z8cU3cVIlyZaCkIY6uITj1IcwxMcjgQytEMtTZAztcMnD3k0ieLgCGJwEBYvXXRlldRuXLhgwwavX0J8lycMydIL6QI5j7IVwwMCjZYotyMgTmA6tQMmDUkZivLzCZJLFEbZWOFGp1biEHFjkAZ5HMJAlIcP3RMtinOoiaJJ5gbP3JJRimZIWn50sZbd2CRjlxbfWUFyAhcRHHJNv7d4Gf94uFLgmg1TlPIpilwAihS8VkBBBnZ8GER2yKZeX9NmzdIwjqociiMqTDQjz6LYj8Eky4MqSc4gydM5zvkfueMQTlEuiOf9Qt=e=R474cb6624d46e0ffc4738da48ec40ec6c752493664e4752ff53db807cace7e4621380eceb4d5de156b785a4403be2968b7a6a22be5c76e8b9cda0494edde848854d6e93a408dc85a76a78ee44989fdb316aafe12f99184914c3eec2accd1689a7983cb8f627bbf1c1ce62f546cc997b117824f4bed3d811de3d6eefd462b467e4bf7bd325190f51155d825c4ba5f300245d7b67550db63b79c8ffc6a34adf6fda39fcd06e2ab1406812358a35ac9f95eca70f2369b30c64b8b61a8e5ae61aa337084058d6616a62e06a4d4a75f10498e2d8a535e4f9dcc1ab389b8bb1a1528df10f2e8b9137f1d9b337c4dca8e20eec88414377e4e374e231b63e0eeae6d2490a0960db48c15809ff54ae57ae06fb1e9679b64dbba7458a9ae271203fa38d2582b5492c92269e8af8ec7cd3e88b50fbaa8a616fa3091ce0a1b5a90abe67666dc7c30d83f4c175d759481f7bda16854a7c1c52148763b845bba4303a8ea97104cdc0258b227c08f59d18db8b753b21f5caa0a47c28958d09ed5cd65c86741a5424a118cb0336ee21aa8e7caa2dc99a093c8d4ec1f77ebf0edebc4b4a59b2014bd44597b3a46b97b3471f8ef2314fe0cc2786e03a1c1881fe3a9c5fdf5b993cde580024846d9921808d77889b25eeea64761c94b44582e0b630a8b888e6d51574b89e1f4fa872f61d1a4842e09ea9db5cd5ae5ed40fc2a96e59b5c62c72d9734b0"  # noqa: E501

import PySimpleGUI as sg

# mpl.use('Qt5Agg')  # Set the backend to Qt5
mpl.use("TkAgg")  # Setting bakcend working best for Spyder
mne.set_config("MNE_BROWSER_BACKEND", "matplotlib")  # Setting for Spyder


# deaults for gui user input
sg.theme("Default1")
font = ("Ubuntu Medium", 14)
f_font = ("Courier New", 12)  # font filter frequency inputs
f_size = 5  # font size filter frequency inputs

sg.set_options(tooltip_font=(16))  # tootip size
settings = {}

EEG_version = "v4.4.2"

# script run defaults
settings["default_epoch_length"] = 8
settings["default_ica_components"] = 25
settings["default_downsample_factor"] = 1
settings["sample_frequencies"] = [250, 256, 500, 512, 1000, 1024, 1250, 2000, 2048, 4000, 5000]
settings["apply_average_ref"] = 1
settings["apply_epoch_selection"] = 0
settings["apply_output_filtering"] = 0
settings["epoch_length"] = 0.0
settings["apply_ica"] = 0
settings["rerun"] = 0
settings["apply_beamformer"] = 0
# settings["channels_to_be_dropped_selected"] = 0
settings["nr_ica_components"] = 0
settings["max_channels"] = 0
settings["skip_input_file"] = 0
settings["file_pattern"] = "-"
settings["input_file_pattern"] = "-"
settings["montage"] = "-"
settings["input_file_names"] = []
settings["input_file_paths"] = []
settings["channel_names"] = []
settings["sample_frequency"] = 250
settings["downsampled_sample_frequency"] = 250  # default, will be set in script
settings["config_file"] = " "
settings["log_file"] = " "
settings["previous_run_config_file"] = " "
settings["output_directory"] = " "
settings["batch_output_subdirectory"] = " "
settings["file_output_subdirectory"] = " "
settings["input_directory"] = " "
settings["batch_name"] = " "
settings["frequency_bands_modified"] = 0
settings["batch_prefix"] = " "
settings["header_rows"] = 1  # Skip ... header rows
settings["channel_names_row"] = 0  # Channel names are in row ... +1 (0-based counting)
# use channel_names_row = None if no header is present

# .txt reading settings
settings["txt_delimiter"] = "auto"  # Options: "auto", "tab", "comma", "semicolon", "space"
settings["txt_decimal_separator"] = "."  # Options: "." (international) or "," (European)
settings["txt_scaling_factor"] = 1e-6  # Default: µV to V conversion. Use 1.0 if already in Volts
settings["txt_encoding"] = "auto"  # Options: "auto", "utf-8", "latin-1", "cp1252"

# --- MONTAGE SETTINGS ---
settings["montage_options"] = {
    "Use Native Coordinates (Auto)": "native",
    "Standard 10-20 (94 channels)": "standard_1020",
    "Standard 10-05 (343 channels)": "standard_1005",
    "BioSemi 32-channel": "biosemi32",
    "BioSemi 64-channel": "biosemi64",
    "BioSemi 128-channel": "biosemi128",
    "BioSemi 160-channel": "biosemi160",
    "BioSemi 256-channel": "biosemi256",
    "EasyCap M1 (74 channels)": "easycap-M1",
    "EasyCap M10 (61 channels)": "easycap-M10",
    "GSN-HydroCel 32-channel": "GSN-HydroCel-32",
    "GSN-HydroCel 64-channel": "GSN-HydroCel-64_1.0",
    "GSN-HydroCel 128-channel": "GSN-HydroCel-128",
    "GSN-HydroCel 256-channel": "GSN-HydroCel-256",
    "MGH 60-channel": "mgh60",
    "MGH 70-channel": "mgh70",
}

# Special handling flags for specific file scenarios
settings["txt_import_options"] = {
    "Import .txt with BioSemi 64 names": "biosemi64",
    "Import .txt with BioSemi 32 names": "biosemi32", 
    "Import .txt with 10-20 names": "standard_1020",
    "Import .txt with 10-05 names": "standard_1005",
    "Import .txt for MEG data": "MEG",
    "Import .txt with generic names": "generic",
}

# Combined list for the dropdown (with visual separators)
settings["input_file_patterns"] = [
    # Auto-detect group
    "Use Native Coordinates (Auto)",
    # Standard layouts group  
    "Standard 10-20 (94 channels)",
    "Standard 10-05 (343 channels)",
    # BioSemi group
    "BioSemi 32-channel",
    "BioSemi 64-channel",
    "BioSemi 128-channel",
    "BioSemi 160-channel",
    "BioSemi 256-channel",
    # EasyCap group
    "EasyCap M1 (74 channels)",
    "EasyCap M10 (61 channels)",
    # GSN/EGI group
    "GSN-HydroCel 32-channel",
    "GSN-HydroCel 64-channel",
    "GSN-HydroCel 128-channel",
    "GSN-HydroCel 256-channel",
    # MGH group
    "MGH 60-channel",
    "MGH 70-channel",
    # Text import group
    "Import .txt with BioSemi 64 names",
    "Import .txt with BioSemi 32 names",
    "Import .txt with 10-20 names",
    "Import .txt with 10-05 names",
    "Import .txt for MEG data",
    "Import .txt with generic names",
]

# Text & Tooltips for the montage selection dialog
settings["input_file_patterns", "text"] = "Select Sensor Layout / Montage"
settings["input_file_patterns", "tooltip"] = (
    "Choose how to handle sensor positions:\n\n"
    "• 'Use Native Coordinates (Auto)':\n"
    "   Best for formats with built-in positions:\n"
    "   .vhdr (BrainVision), .fif (Neuromag), .set (EEGLAB),\n"
    "   .mff (EGI), .nxe (Nicolet)\n\n"
    "• Standard/BioSemi/GSN layouts:\n"
    "   Use for formats without positions:\n"
    "   .bdf, .edf, .cnt, or when you need to override\n\n"
    "• 'Import .txt ...':\n"
    "   For ASCII text file imports - select the naming\n"
    "   convention that matches your channel names"
)

# Legacy compatibility mappings (for old config files)
# Maps old-style keys to new montage names
settings["legacy_montage_map"] = {
    ("montage", "Use Native Coordinates (Auto)"): "native",
    ("montage", "Force Standard 10-20 Layout"): "standard_1020",
    ("montage", "Force Standard 10-05 Layout"): "standard_1005",
    ("montage", "Force Biosemi 32 Layout"): "biosemi32",
    ("montage", "Force Biosemi 64 Layout"): "biosemi64",
    ("montage", "Force Biosemi 128 Layout"): "biosemi128",
    ("montage", "Force GSN-Hydrocel 64 Layout"): "GSN-HydroCel-64_1.0",
    ("input_file_pattern", "Import .txt (Biosemi 64)"): "biosemi64",
    ("input_file_pattern", "Import .txt (Biosemi 32)"): "biosemi32",
    ("input_file_pattern", "Import .txt (10-20)"): "standard_1020",
    ("input_file_pattern", "Import .txt (10-05)"): "standard_1005",
    ("input_file_pattern", "Import .txt (MEG)"): "MEG",
}

settings["load_config_file", "text"] = "Select a previously created .pkl file"

# Allowed file extensions for the GUI file selector
settings["input_file_paths", "type_EEG"] = (
    ("All Supported Files", "*.*"),
    ("EEG .txt Files", "*.txt"),
    ("EEG .bdf Files", "*.bdf"),
    ("EEG .vhdr Files", "*.vhdr"),
    ("EEG .edf Files", "*.edf"),
    ("Fif", "*.fif"),
    ("CNT Files", "*.cnt"),
    ("SET Files (EEGLAB)", "*.set"),
    ("MFF Files (EGI)", "*.mff"),
)

settings["input_file_paths", "text"] = "Select input EEG file(s) - on Mac use 'Options' to filter file types"
settings["output_txt_decimals"] = 4  # used in np.round to round down exported txt files


# --- FREQUENCY BANDS ---

settings["frequency_bands"] = (
    "delta_low",
    "delta_high",
    "theta_low",
    "theta_high",
    "alpha_low",
    "alpha_high",
    "alpha1_low",
    "alpha1_high",
    "alpha2_low",
    "alpha2_high",
    "beta_low",
    "beta_high",
    "beta1_low",
    "beta1_high",
    "beta2_low",
    "beta2_high",
    "broadband_low",
    "broadband_high",
)

# Default filter settings
settings["cut_off_frequency", "delta_low"] = 0.5
settings["cut_off_frequency", "delta_high"] = 4.0
settings["cut_off_frequency", "theta_low"] = 4.0
settings["cut_off_frequency", "theta_high"] = 8.0
settings["cut_off_frequency", "alpha_low"] = 8.0
settings["cut_off_frequency", "alpha_high"] = 13.0
settings["cut_off_frequency", "beta1_low"] = 13.0
settings["cut_off_frequency", "beta1_high"] = 20.0
settings["cut_off_frequency", "beta2_low"] = 20.0
settings["cut_off_frequency", "beta2_high"] = 30.0
settings["cut_off_frequency", "broadband_low"] = 0.5
settings["cut_off_frequency", "broadband_high"] = 47.0

settings["cut_off_frequency", "alpha1_low"] = 8.0
settings["cut_off_frequency", "alpha1_high"] = 10.0
settings["cut_off_frequency", "alpha2_low"] = 10.0
settings["cut_off_frequency", "alpha2_high"] = 13.0
settings["cut_off_frequency", "beta_low"] = 13.0
settings["cut_off_frequency", "beta_high"] = 30.0

settings["use_split_alpha"] = False
settings["use_split_beta"] = True

settings["general_filt_low"] = 0.5
settings["general_filt_high"] = 47.0


# Possible MNE Montages:
# - ('standard_1005', 'Electrodes are named and positioned according to the international 10-05 system (343+3 locations)')
# - ('standard_1020', 'Electrodes are named and positioned according to the international 10-20 system (94+3 locations)')
# - ('standard_alphabetic', 'Electrodes are named with LETTER-NUMBER combinations (A1, B2, F4, …) (65+3 locations)')
# - ('standard_postfixed', 'Electrodes are named according to the international 10-20 system using postfixes for intermediate positions (100+3 locations)')
# - ('standard_prefixed', 'Electrodes are named according to the international 10-20 system using prefixes for intermediate positions (74+3 locations)')
# - ('standard_primed', "Electrodes are named according to the international 10-20 system using prime marks (' and '') for intermediate positions (100+3 locations)")
# - ('biosemi16', 'BioSemi cap with 16 electrodes (16+3 locations)')
# - ('biosemi32', 'BioSemi cap with 32 electrodes (32+3 locations)')
# - ('biosemi64', 'BioSemi cap with 64 electrodes (64+3 locations)')
# - ('biosemi128', 'BioSemi cap with 128 electrodes (128+3 locations)')
# - ('biosemi160', 'BioSemi cap with 160 electrodes (160+3 locations)')
# - ('biosemi256', 'BioSemi cap with 256 electrodes (256+3 locations)')
# - ('easycap-M1', 'EasyCap with 10-05 electrode names (74 locations)')
# - ('easycap-M10', 'EasyCap with numbered electrodes (61 locations)')
# - ('easycap-M43', 'EasyCap with numbered electrodes (64 locations)')
# - ('EGI_256', 'Geodesic Sensor Net (256 locations)')
# - ('GSN-HydroCel-32', 'HydroCel Geodesic Sensor Net and Cz (33+3 locations)')
# - ('GSN-HydroCel-64_1.0', 'HydroCel Geodesic Sensor Net (64+3 locations)')
# - ('GSN-HydroCel-65_1.0', 'HydroCel Geodesic Sensor Net and Cz (65+3 locations)')
# - ('GSN-HydroCel-128', 'HydroCel Geodesic Sensor Net (128+3 locations)')
# - ('GSN-HydroCel-129', 'HydroCel Geodesic Sensor Net and Cz (129+3 locations)')
# - ('GSN-HydroCel-256', 'HydroCel Geodesic Sensor Net (256+3 locations)')
# - ('GSN-HydroCel-257', 'HydroCel Geodesic Sensor Net and Cz (257+3 locations)')
# - ('mgh60', 'The (older) 60-channel cap used at MGH (60+3 locations)')
# - ('mgh70', 'The (newer) 70-channel BrainVision cap used at MGH (70+3 locations)')
# - ('artinis-octamon', 'Artinis OctaMon fNIRS (8 sources, 2 detectors)')
# - ('artinis-brite23', 'Artinis Brite23 fNIRS (11 sources, 7 detectors)')
# - ('brainproducts-RNP-BA-128', 'Brain Products with 10-10 electrode names (128 channels)')
