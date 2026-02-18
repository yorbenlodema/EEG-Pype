import matplotlib as mpl
import mne
import FreeSimpleGUI as sg

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

EEG_version = "v4.4.5"

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

# ICLabel settings
settings["use_icalabel"] = 0  # Default off, user can enable

# Beamformer atlas settings
settings["beamformer_atlases"] = {
    "Desikan-Killiany (68 cortical)": "desikan",
    "Brainnetome (246 regions)": "bna",
    "AAL2 (120 regions)": "aal2",
    "AAL3 (170 regions)": "aal3",
}

settings["beamformer_atlas_selection"] = ["desikan"]  # Default: DK only
settings["beamformer_include_subcortical"] = False  # Cortical only by default

# Atlas metadata: voxel files, label files, and whether atlas has subcortical regions
settings["atlas_config"] = {
    "desikan": {
        "voxels": "DesikanVox.xlsx",
        "labels": "DesikanVoxLabels.csv",
        "labels_cortical": "DesikanVoxLabels.csv",  # DK is cortical-only
        "has_subcortical": False,
        "n_regions_full": 68,
        "n_regions_cortical": 68,
    },
    "bna": {
        "voxels": "BNA_Vox.xlsx",
        "labels": "BNA_VoxLabels.csv",
        "labels_cortical": "BNA_VoxLabels_cortical.csv",
        "has_subcortical": True,
        "n_regions_full": 246,
        "n_regions_cortical": 210,
        # Subcortical regions are indices 211-246 (1-indexed) in BNA
        "subcortical_indices": list(range(210, 246)),  # 0-indexed
    },
    "aal2": {
        "voxels": "AAL2_Vox.xlsx",
        "labels": "AAL2_VoxLabels.csv",
        "labels_cortical": "AAL2_VoxLabels_cortical.csv",
        "has_subcortical": True,
        "n_regions_full": 120,
        "n_regions_cortical": 80,
        "subcortical_indices": [
            16, 17, 40, 41, 44, 45, 74, 75, 76, 77, 78, 79, 80, 81,
            94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105,
            106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116,
            117, 118, 119,
        ],
    },
    "aal3": {
        "voxels": "AAL3_Vox.xlsx",
        "labels": "AAL3_VoxLabels.csv",
        "labels_cortical": "AAL3_VoxLabels_cortical.csv",
        "has_subcortical": True,
        "n_regions_full": 166,
        "n_regions_cortical": 84,
        "subcortical_indices": [
            16, 17, 38, 39, 42, 43, 72, 73, 74, 75, 76, 77,
            90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101,
            102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
            112, 113, 114, 115, 116, 117, 118, 119, 120, 121,
            122, 123, 124, 125, 126, 127, 128, 129, 130, 131,
            132, 133, 134, 135, 136, 137, 138, 139, 140, 141,
            142, 143, 144, 145, 152, 153, 154, 155, 156, 157,
            158, 159, 160, 161, 162, 163, 164, 165,
        ],
    },
}

# Display order for GUI
settings["beamformer_atlas_order"] = ["desikan", "bna", "aal2", "aal3"]

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
