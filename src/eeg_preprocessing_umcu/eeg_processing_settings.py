"""@author: hvand."""

import matplotlib as mpl
import mne

PySimpleGUI_License = "e1yWJaMdasWkN4l4b7nYNllfVqHolVwwZUS5IA6pIekqRAp7cf3FRNyZakWgJy1ldnGXlZv7bhi9Ihs0Ifkvxbp2YA2KVOuKct2IVPJHRsC1IZ6tMETZcDyzOjDDQI2KMEzVIM3WMoSXw2izTBGmlgjvZrWx5GzDZHUvR0lZcOGCxBvSeXWR1OlFbsnKRlWGZkXNJYzXaLWC9TuQIljdo8iaNpSW4GwlIiiBwZi0TOmBFwtvZnUCZrpQc3ntNF0IIcj3ozi6W3Wj9QycYnmfV7u4ImirwWiLTimEFutRZ8Ujxih2c238QuirOCi9JAMabp29R9lzbBWSE9i2LWCrJkDfbX2A1vwVYAWY5y50IEjGojizI4itwdikQk3XVdzJdnGF94tYZjXDJnJhRpCNIr6RIijsQVxDNgjmYQyvIki5wWitRgGVFK0BZqUol9z8cU3cVIlyZaCkIY6uITj1IcwxMcjgQytEMtTZAztcMnD3k0ieLgCGJwEBYvXXRlldRuXLhgwwavX0J8lycMydIL6QI5j7IVwwMCjZYotyMgTmA6tQMmDUkZivLzCZJLFEbZWOFGp1biEHFjkAZ5HMJAlIcP3RMtinOoiaJJ5gbP3JJRimZIWn50sZbd2CRjlxbfWUFyAhcRHHJNv7d4Gf94uFLgmg1TlPIpilwAihS8VkBBBnZ8GER2yKZeX9NmzdIwjqociiMqTDQjz6LYj8Eky4MqSc4gydM5zvkfueMQTlEuiOf9Qt=e=R474cb6624d46e0ffc4738da48ec40ec6c752493664e4752ff53db807cace7e4621380eceb4d5de156b785a4403be2968b7a6a22be5c76e8b9cda0494edde848854d6e93a408dc85a76a78ee44989fdb316aafe12f99184914c3eec2accd1689a7983cb8f627bbf1c1ce62f546cc997b117824f4bed3d811de3d6eefd462b467e4bf7bd325190f51155d825c4ba5f300245d7b67550db63b79c8ffc6a34adf6fda39fcd06e2ab1406812358a35ac9f95eca70f2369b30c64b8b61a8e5ae61aa337084058d6616a62e06a4d4a75f10498e2d8a535e4f9dcc1ab389b8bb1a1528df10f2e8b9137f1d9b337c4dca8e20eec88414377e4e374e231b63e0eeae6d2490a0960db48c15809ff54ae57ae06fb1e9679b64dbba7458a9ae271203fa38d2582b5492c92269e8af8ec7cd3e88b50fbaa8a616fa3091ce0a1b5a90abe67666dc7c30d83f4c175d759481f7bda16854a7c1c52148763b845bba4303a8ea97104cdc0258b227c08f59d18db8b753b21f5caa0a47c28958d09ed5cd65c86741a5424a118cb0336ee21aa8e7caa2dc99a093c8d4ec1f77ebf0edebc4b4a59b2014bd44597b3a46b97b3471f8ef2314fe0cc2786e03a1c1881fe3a9c5fdf5b993cde580024846d9921808d77889b25eeea64761c94b44582e0b630a8b888e6d51574b89e1f4fa872f61d1a4842e09ea9db5cd5ae5ed40fc2a96e59b5c62c72d9734b0"  # noqa: E501
import PySimpleGUI as sg  # noqa: E402

# mpl.use('Qt5Agg')  # Set the backend to Qt5  #noqa: ERA001 #TODO: check ERA001 statements to see if you can remove the lines
mpl.use("TkAgg")  # Setting bakcend working best for Spyder
mne.set_config("MNE_BROWSER_BACKEND", "matplotlib")  # Setting for Spyder


# deaults for gui user input
sg.theme("Default1")
font = ("Ubuntu Medium", 14)
f_font = ("Courier New", 12)  # font filter frequency inputs
f_size = 5  # font size filter frequency inputs
my_image = sg.Image("UMC_logo.png", subsample=2, pad=(0, 0), background_color="#E6F3FF")  # UMC logo

sg.set_options(tooltip_font=(16))  # tootip size
settings = {}
filter_settings = {}

# script run defaults
settings["default_epoch_length"] = 8
settings["default_ica_components"] = 25
settings["default_downsample_factor"] = 1
settings["sample_frequencies"] = [250, 256, 500, 512, 1000, 1024, 1250, 2000, 2048, 5000]
settings["apply_average_ref"] = 1
settings["apply_epoch_selection"] = 0
settings["apply_output_filtering"] = 0
settings["epoch_length"] = 0.0
settings["apply_ica"] = 0
settings["rerun"] = 0
# settings['rerun_no_previous_epoch_selection'] = 0  #noqa: ERA001
settings["apply_beamformer"] = 0
settings["channels_to_be_dropped_selected"] = 0
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

settings["no_montage_patterns"] = ["*.vhdr", "*.fif", "*.cnt"]

settings["montage", ".txt_bio32"] = "biosemi32"
settings["montage", ".txt_bio64"] = "biosemi64"
settings["montage", ".txt_10-20"] = "standard_1020"
settings["montage", ".txt_MEG"] = "MEG"
settings["montage", ".bdf_32"] = "biosemi32"
settings["montage", ".bdf_64"] = "biosemi64"
settings["montage", ".bdf_128"] = "biosemi128"
settings["montage", ".edf_bio32"] = "biosemi32"
settings["montage", ".edf_bio64"] = "biosemi64"
settings["montage", ".edf_bio128"] = "biosemi128"
settings["montage", ".edf_10-20"] = "standard_1020"
settings["montage", ".edf_GSN-Hydrocel_64"] = "GSN-HydroCel-64_1.0"
# settings['montage',".eeg"] = "n/a"  #noqa: ERA001
# settings['montage',".fif"] = "n/a"  #noqa: ERA001
# settings['montage',".cnt"] = "standard_1005"  #noqa: ERA001

settings["input_file_pattern", ".txt_bio32"] = "*.txt"
settings["input_file_pattern", ".txt_bio64"] = "*.txt"
settings["input_file_pattern", ".txt_10-20"] = "*.txt"
settings["input_file_pattern", ".txt_MEG"] = "*.txt"
settings["input_file_pattern", ".bdf_32"] = "*.bdf"
settings["input_file_pattern", ".bdf_64"] = "*.bdf"
settings["input_file_pattern", ".bdf_128"] = "*.bdf"
settings["input_file_pattern", ".edf_bio32"] = "*.edf"  # Biosemi montage
settings["input_file_pattern", ".edf_bio64"] = "*.edf"  # Biosemi montage
settings["input_file_pattern", ".edf_bio128"] = "*.edf"  # Biosemi montage
settings["input_file_pattern", ".edf_10-20"] = "*.edf"  # Generic 10-20 montage
settings["input_file_pattern", ".edf_GSN-Hydrocel_64"] = "*.edf"
settings["input_file_pattern", ".eeg"] = "*.vhdr"
settings["input_file_pattern", ".fif"] = "*.fif"
settings["input_file_pattern", ".cnt"] = "*.cnt"

# defaults for frequency band filter settings
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
settings["cut_off_frequency", "delta_high"] = 4
settings["cut_off_frequency", "theta_low"] = 4
settings["cut_off_frequency", "theta_high"] = 8
settings["cut_off_frequency", "alpha_low"] = 8
settings["cut_off_frequency", "alpha_high"] = 13
settings["cut_off_frequency", "beta1_low"] = 13
settings["cut_off_frequency", "beta1_high"] = 20
settings["cut_off_frequency", "beta2_low"] = 20
settings["cut_off_frequency", "beta2_high"] = 30
settings["cut_off_frequency", "broadband_low"] = 0.5
settings["cut_off_frequency", "broadband_high"] = 47

settings["cut_off_frequency", "alpha1_low"] = 8
settings["cut_off_frequency", "alpha1_high"] = 10
settings["cut_off_frequency", "alpha2_low"] = 10
settings["cut_off_frequency", "alpha2_high"] = 13
settings["cut_off_frequency", "beta_low"] = 13
settings["cut_off_frequency", "beta_high"] = 30

settings["use_split_alpha"] = False
settings["use_split_beta"] = False

settings["input_file_patterns"] = [
    ".bdf_32",
    ".bdf_64",
    ".bdf_128",
    ".edf_bio32",
    ".edf_bio64",
    ".edf_bio128",
    ".edf_10-20",
    ".fif",
    ".eeg",
    ".edf_GSN-Hydrocel_64",
    ".txt_bio32",
    ".txt_bio64",
    ".txt_10-20",
    ".cnt",
    ".txt_MEG",
]
# text & tool tips
settings["input_file_patterns", "text"] = "Enter file type"
settings["input_file_patterns", "tooltip"] = (
    "Enter one filetype and electrode layout: .bdf 32ch, .bdf 64ch, .bdf 128ch, .edf biosemi 32 layout,\n .edf biosemi 64 layout, .edf biosemi 128 layout, .edf general 10-20 layout, .eeg, .txt biosemi 32 layout,\n .txt biosemi 64 layout, .txt general 10-20 layout, \nsee https://mne.tools/dev/auto_tutorials/intro/40_sensor_locations.html for the electrode layouts (montages) used"
)
settings["load_config_file", "text"] = "Select a previously created .pkl file"
settings["input_file_paths", "type_EEG"] = (
    ("EEG .txt Files", "*.txt"),
    ("EEG .bdf Files", "*.bdf"),
    ("EEG .vhdr Files", "*.vhdr"),
    ("EEG .edf Files", "*.edf"),
    ("Fif", "*.fif"),
    ("CNT Files", "*.cnt"),
)  # note the comma...
settings["input_file_paths", "text"] = "Select input EEG file(s) - on Mac use 'Options' to filter file types "

settings["output_txt_decimals"] = 4  # used in np.round to round down exported txt files
