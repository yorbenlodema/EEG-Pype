## What is this software?

With our software, we hope to provide users with an accessible way to preprocess resting-state EEG files, while still including powerful analysis tools. We do this by combining several functions from the MNE (MEG/EEG preprocessing) open-source project *(Gramfort et al., Frontiers in Neuroscience, 2013)*. By using an intuitive graphical user interface on top of a Python script, we hope that our software is easy to start using without coding experience, while still allowing more experienced users to adapt the software to their needs by altering the underlying code.

The software is currently able to:
- Open raw EEG files of type .txt, .bdf, .edf, .eeg and .fif.
- Open a single EEG or choose analysis settings for an entire batch of files.
- Apply a montage to the raw EEG (including electrode coordinates necessary for some analyses).
- Drop bad channels entirely.
- Interpolate bad channels after visual inspection.
- Apply an average reference.
- Apply independent component analysis to remove artefacts. For this, you can change the number of components that are calculated (please read up on this before use).
- Apply beamformer source reconstruction to the EEG (standard MNE LCMV beamformer with standard head model).
- Down sample the file to a lower sample frequency by specifying a downsample factor (like a foctor of 4: from 2048 Hz to 512 Hz for example).
- Perform interactive visual epoch selection.
- Perform filtering in different frequency bands and broadband output. These bands can be changed for the current batch in the GUI or more permanently in the settings file (see under tips and issues). The EEGs are filtered before cutting epochs, reducing edge artifacts.
- Split alpha and beta bands into sub-bands (alpha1/alpha2 and beta1/beta2) for more detailed frequency analysis.
- After performing analyses on a batch, rerun the batch with preservation of channel and epoch selection. To do this, select the previously created .pkl file.
- Log the chosen settings and performed analyses steps in a log file.
- Correct channel names to match expected montage names through an interactive find/replace interface.


The software is not (yet) able to:
- Analyse task EEG data.
- Open EEG files with data types not mentioned previously (you can put this in a new GitHub issue if you need to load another EEG filetype).

In addition, we later added a quantitative analysis script, which allows for the calculation of several commonly used quantitative measures on the resting-state EEG epochs that are created by our pre-processing software. See below for more details.

### Tips for use and known issues
When choosing the settings for the current analysis batch, most windows contain a "more info" button which will take you to an appropriate MNE documentation page.

When no raw EEG files show up in the file selection window, please choose a different file type in the dropdown menu on the right (it might be stuck on only showing .txt files for instance).

For the bad channel selection (for interpolation), you can select bad channels by clicking the channel names on the left side of the plot. The deselected (grey) channels will be interpolated. For ICA, this works the same but then artefact-containing components can be deselected in the graph plot of the ICA. These components will be filtered out of the EEG. For interactive epoch selection, epochs of insufficient quality can be deselected by clicking anywhere on the epoch, which will then turn red. This means the epoch will not be saved.

If the program glitches or stops working, we found that it works best to stop the Python process, for instance by clicking the red stop button or restarting the kernel in Spyder IDE or similar.

RESOLVED: ~~There is currently an unresolved problem where removing multiple ICA components and/or interpolating channels can result in a data rank that is too low to caculate the beamforming solution. See [here](https://mailman.science.ru.nl/pipermail/fieldtrip/2014-March/033565.html) for an explanation of this problem.~~

When using Spyder IDE to run the program, initially Spyder can prompt the user that it does not have the spyder-kernels module. Please follow the instructions provided in the console.

It is possible to change the underlying Python code (however, this is mostly unnecessary). Of the two main scripts, eeg_processing_script.py and eeg_processing_settings.py, the latter is the easiest to modify. Here, you can for instance rather easily change the standard output filter frequency bands (like delta, theta etc.). Note however, that it is currently not possible to increase or decrease the number of bands that the output is filtered in. In some IDE's, or with certain setups, it can also be necessary to change the matplotlib backend, for instance from TkAgg to Qt5Agg in the beginning of the settings script.

When loading EEG files, the software now includes a channel name correction feature. This helps when your EEG files have channel names that don't exactly match the expected montage (e.g., channels prefixed with "EEG" or having different capitalization). The interface shows you the current channel names versus the expected names for your chosen montage, and allows you to use find/replace to correct them. These corrections are then applied to each file separately. This way, there is a check for each file to see wether the channel names match the MNE montage.

When plotting the EEG channels side by side, note that ECG channels should be purple if recognized correctly:
![SCR-20250423-iyif](https://github.com/user-attachments/assets/b43621ec-3d84-44e4-a404-b1b499e9fe4d)
If this is not the case, these channels might be included in operations like average referencing.

The frequency band settings now include the option to split the alpha band (into alpha1: 8-10 Hz and alpha2: 10-13 Hz) and beta band (into beta1: 13-20 Hz and beta2: 20-30 Hz). You can toggle these splits when setting up your batch processing (under the change filter bands option). This allows for more detailed analysis of specific frequency ranges.

## Short video overviews
[![EEG-Pype Part 1](http://img.youtube.com/vi/NGzrvGhQj0g/0.jpg)](http://www.youtube.com/watch?v=NGzrvGhQj0g "EEG-Pype Part 1")

## Installation

This guide will walk you through the process of setting up the EEG Preprocessing Tool using Miniconda. Miniconda provides an easy way to create isolated Python environments and manage package dependencies.

### 1. Install Miniconda and Git

First, download and install Miniconda:

- For Windows: [Miniconda Windows Installer](https://docs.conda.io/en/latest/miniconda.html#windows-installers)
- For macOS: [Miniconda macOS Installer](https://docs.conda.io/en/latest/miniconda.html#macos-installers)
- For Linux: [Miniconda Linux Installer](https://docs.conda.io/en/latest/miniconda.html#linux-installers)

Follow the installation instructions provided on the Miniconda website.

If not done already, [install Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).

### 2. Clone the Repository

Clone the EEG Preprocessing Tool repository:

```bash
git clone https://github.com/yorbenlodema/EEG-Pype.git
cd EEG-Pype
```

### 3. Set up a Conda Environment

Open a terminal (or Anaconda Prompt on Windows) and run the following commands:

```bash
# Create a new conda environment with the required packages
conda env create -f Environment.yml

# Activate the new environment
conda activate EEG-Pype
```

### 4. Install PySimpleGUI separately

Please verify that the newly created EEG-Pype Conda environment is activated by checking if your Terminal/Command Prompt line starts with "(EEG-Pype)" and not "(base)". Then install the package using pip:

```bash
python -m pip install --force-reinstall --extra-index-url https://PySimpleGUI.net/install PySimpleGUI
```

### 5. Verify Installation

To verify that the installation was successful, you can try running the main script (eeg_processing_script.py) in your favorite way (we have used Spyder to run the script during development). For the first use, it is important to select your newly created Miniconda environment in your IDE. In Spyder this is done via: preferences > Python interpreter > use the following interpreter. Here you should be able to select your EEG-Pype Conda environment. When opening the script in an IDE like Spyder, you can simply press 'run' to start the script. If everything is set up correctly, the script should run without any import errors.

## Troubleshooting

If you encounter any issues during installation:

1. Make sure you have activated the conda environment (`conda activate EEG-Pype`).
2. Try updating pip: `python -m pip install --upgrade pip`
3. If you encounter any dependency conflicts, you can try installing dependencies manually:
   ```bash
   conda install numpy pandas matplotlib scikit-learn
   pip install PySimpleGUI mne
   ```

For any further issues, please open an issue on the [GitHub repository](https://github.com/snorben/EEG-Pype/issues).

## Updating the Software

When there's an update available on GitHub, follow these steps to update your local installation:
1. Navigate to the Project Directory
Open a terminal (or Anaconda Prompt on Windows) and navigate to your project directory:
```bash
cd path/to/EEG-Pype
```
3. Activate the Conda Environment
Ensure you're using the correct environment:
```bash
conda activate EEG-Pype
```
5. Pull the Latest Changes
Fetch and merge the latest changes from the GitHub repository:
```bash
git pull origin main
```
7. Update Dependencies
If there are any changes to the dependencies, reinstall the package:
```bash
python -m pip install . --upgrade
```
This command will update the package and any new or updated dependencies.

### If you encounter issues after updating:

Ensure your conda environment is up to date:
```bash
conda update --all
```

If you're still having problems, you can try creating a fresh environment:
```bash
conda deactivate
conda remove --name EEG-Pype --all
conda create -n EEG-Pype python=3.11
conda activate EEG-Pype
python -m pip install .
```

## Quantitative analysis module (separate script)
### Overview
The EEG Quantitative Analysis Tool is a GUI-based application for calculating various quantitative features from preprocessed EEG epochs. Different from the preprocessing software, this program is best run from the command line due to compatibility issues of the parallel processing implementation with IDEs like Spyder. To do this, simply change directory to the folder containing the Python script and use (similar to): ```python eeg_quantitative_analysis.py```. Then, the GUI should load automatically.

Depending on your setup, it is probably advisable to not run too many EEGs in one go in the analysis script, since this can cause problems (probably memory-related) when saving the analysis output. Amounts of around 100-200 EEGs should work.

In the GUI, the number of threads should be specified. This number means that the calculations will be spread over multiple CPU cores. It is advisable to leave one, or even two, of your available cores free for other tasks your computer has to perform to prevent freezes. If the script runs into memory problems, especially when calculating entropy measures, it can be necessary to lower the number of threads the analyses run on.

### Data Requirements
- Input data should be organized in folders ending with a specified extension (e.g., 'bdf', 'edf'). This should be the standard output from the preprocessing script. This extension is specified in the "Folder extension" field.
- Each folder should contain epoch files in .txt format.
- Epoch files should follow the naming convention: `[subject]_[level]_level_[frequency]Hz_Epoch_[number].txt`.
- Data can be loaded with or without headers:
  - With headers (default): First row contains channel names.
  - Without headers: Channel names will be auto-generated as "Channel_1", "Channel_2", etc.

### Features
#### Connectivity Measures  
1. **Phase Lag Index (PLI)** *(Stam et al., Human Brain Mapping, 2007)*
   - Measures the asymmetry of the distribution of phase differences between two signals.
   - Key Benefit: It ignores zero-lag synchronization (common in volume conduction) by exclusively quantifying non-zero phase lag.
   - Range: 0 (no coupling or zero-lag coupling) to 1 (perfect non-zero lag synchronization).
     
2. **Phase Lag Time (PLT)** *Based on BrainWave software: https://github.com/CornelisStam/BrainWave*  
- Concept: Unlike PLI, which looks at the distribution over the whole epoch, PLT measures the temporal stability of the phase relationship. It quantifies the average duration that one signal consistently leads or lags the other before the relationship flips.
- Interpretation: A value close to 1.0 indicates a stable leading/lagging relationship throughout the epoch. A value close to 0.0 indicates frequent, unstable switching between leading and lagging.
- Custom Implementation: This version includes a noise threshold parameter. This acts as a refractory period to filter out rapid, high-frequency phase slips (noise), ensuring that only significant changes in the phase relationship affect the score.

3. **Amplitude Envelope Correlation (AEC)**
   - Measures correlation between amplitude envelopes of band-filtered signals
   - Options:
     - Standard AEC: Direct correlation of Hilbert envelopes.
     - Orthogonalized AEC (AECc): Removes zero-lag correlations through orthogonalization.
     - Epoch concatenation: Recommended for short epochs to improve reliability.
     - Force positive: Often used as negative correlations may not be physiologically meaningful.

#### Complexity Measures
1. **Joint Permutation Entropy (JPE/PE)** *(Scheijbeler et al., Network Neuroscience, 2022) and (Bandt and Pompe, Pysical Review Letters, 2002)*
   - Quantifies complexity through ordinal patterns in the signal.
   - Options:
     - Time step (tau): Determines temporal scale of patterns (should be adjusted based on sampling rate).
     - Integer conversion: Can improve detection of equal values.
     - Inversion: 1-JPE provides a measure of similarity rather than complexity.
   - PE calculated per channel, JPE for channel pairs.

2. **Sample Entropy (SampEn)**
   - Measures signal regularity/predictability.
   - Less sensitive to data length than ApEn.
   - Higher values indicate more complexity/randomness.
   - Order m: Length of compared sequences (typically 2 or 3).

3. **Approximate Entropy (ApEn)**
   - Similar to SampEn but with self-matches.
   - Options:
     - Order m: Pattern length (typically 2).
     - Tolerance r: Similarity criterion (typically 0.1-0.25 * SD).

#### Spectral Analysis Details
1. **Peak Frequency Analysis**
   - Calculated using Welch's method with smoothing.
   - For multiple peaks:
     - Uses kernel density estimation to find the dominant frequency.
     - Considers peak prominence to filter noise.
   - Reports channels without clear peaks separately.

2. **Spectral Variability**
   - Tracks temporal fluctuations in relative band power.
   - Uses sliding window approach.
   - Coefficient of variation calculated per frequency band.
   - Window length affects temporal resolution vs. stability.

#### MST Measures (for PLI and AEC(c))
- Degree: Maximum node degree normalized by (M-1).
- Eccentricity: Longest path from each node.
- Betweenness centrality: Node importance in network.
- Leaf fraction: Proportion of nodes with degree 1.
- Tree hierarchy: Balance between network integration and overload prevention.
- Additional measures: Diameter, kappa (degree divergence), mean edge weight.

### Spectral Analysis Methods
The tool now supports multiple methods for spectral analysis:
1. Multitaper Method (Default)
- Provides optimal frequency resolution.
- Best for detecting narrow-band signals.
- Uses MNE's implementation.
- Parameters automatically optimized.

2. Welch's Method
- Reduces noise through averaging.
- Configurable parameters:
   - Window length (ms)
   - Overlap percentage
- Better for smooth spectra.
- Good for longer recordings.

3. FFT Method
- Direct Fast Fourier Transform.
- Uses Hanning window.
- Fastest computation.

Method selection and parameters can be configured through the GUI:
- Choose method from dropdown.
- Welch parameters appear when selected:
   - Window length in milliseconds (default: 1000ms)
   - Overlap percentage (default: 50%)

The selected method affects (this PSD method is used for):
- Power band calculations.
- Peak frequency detection.

### Custom Frequency Bands
The tool allows customization of frequency bands used for both epoch recognition and spectral analysis. Bands are configured in the FREQUENCY_BANDS dictionary in the main script. Please be careful when changing these since doing so can easily break a lot of the logic in the code.

Each band requires:

- A unique name (e.g., "delta", "theta").
- Pattern: Regular expression to match band names in epoch filenames.
- Range: Tuple of (min_freq, max_freq) in Hz for calculations.

Notes:

- The "broadband" band is required and used for power/spectral variability calculations. You can also used unfiltered epochs for this though you should make sure they are recognized as broadband.
- You can add custom bands following the same format.
- Patterns should match your epoch filename format.
- Ranges must be within Nyquist frequency (sampling_rate/2).

### Output Options
1. **Excel Results**
   - Whole-brain averages, averaged over all the channels or brain areas, and epochs.
   - Channel-level averages (optional), the features calculated per channel or brain area, averaged over epochs.
   - Analysis information sheet.
   - Channel metadata.

2. **Connectivity Matrices**
   - Save raw connectivity matrices. This option saves the full connectivity matrices, averaged over the epochs.
   - Save MST matrices. This option saves the full connectivity matrices, calculated over the epoch-averaged connectivity matrices.
   - Matrices are saved in subject-specific folders with proper channel labeling.

### Performance Options
- Configurable number of processing threads for parallel processing, increasing speed.
- Batch processing with memory management.
- Progress tracking and detailed logging.

### Usage Tips
1. Select your data folder and specify the folder extension.
2. Configure desired measures and their parameters.
3. Choose an appropriate header option based on your epoch file format.
4. Do not forget to specify sample frequency, even if not calculating power-based features.
5. Select output options (matrices, channel averages).
6. Monitor progress through the GUI and log window.
7. Results will be saved in the input folder with timestamp.

### Memory Considerations
- The tool includes memory monitoring.
- Large datasets are processed in batches.
- For large datasets or for calculating entropy measures on long/high sample frequency epochs, consider reducing the number of parallel threads.

## Contributing

If you want to contribute to the development of eeg_preprocessing_umcu,
have a look at the [contribution guidelines](CONTRIBUTING.md).

## Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [NLeSC/python-template](https://github.com/NLeSC/python-template).

## Badges

(Customize these badges with your own links, and check https://shields.io/ or https://badgen.net/ to see which other badges are available.)

| fair-software.eu recommendations | |
| :-- | :--  |
| (1/5) code repository              | [![github repo badge](https://img.shields.io/badge/github-repo-000.svg?logo=github&labelColor=gray&color=blue)](https://github.com/snorben/eeg_preprocessing_umcu) |
| (2/5) license                      | [![github license badge](https://img.shields.io/github/license/snorben/eeg_preprocessing_umcu)](https://github.com/snorben/eeg_preprocessing_umcu) |
| (3/5) community registry           | [![RSD](https://img.shields.io/badge/rsd-eeg_preprocessing_umcu-00a3e3.svg)](https://www.research-software.nl/software/eeg_preprocessing_umcu) [![workflow pypi badge](https://img.shields.io/pypi/v/eeg_preprocessing_umcu.svg?colorB=blue)](https://pypi.python.org/project/eeg_preprocessing_umcu/) |
| (4/5) citation                     | [![DOI](https://zenodo.org/badge/DOI/<replace-with-created-DOI>.svg)](https://doi.org/<replace-with-created-DOI>) |
| (5/5) checklist                    | [![workflow cii badge](https://bestpractices.coreinfrastructure.org/projects/<replace-with-created-project-identifier>/badge)](https://bestpractices.coreinfrastructure.org/projects/<replace-with-created-project-identifier>) |
| howfairis                          | [![fair-software badge](https://img.shields.io/badge/fair--software.eu-%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8B-yellow)](https://fair-software.eu) |
| **Other best practices**           | &nbsp; |
| Static analysis                    | [![workflow scq badge](https://sonarcloud.io/api/project_badges/measure?project=snorben_eeg_preprocessing_umcu&metric=alert_status)](https://sonarcloud.io/dashboard?id=snorben_eeg_preprocessing_umcu) |
| Coverage                           | [![workflow scc badge](https://sonarcloud.io/api/project_badges/measure?project=snorben_eeg_preprocessing_umcu&metric=coverage)](https://sonarcloud.io/dashboard?id=snorben_eeg_preprocessing_umcu) |
| Documentation                      | [![Documentation Status](https://readthedocs.org/projects/eeg_preprocessing_umcu/badge/?version=latest)](https://eeg_preprocessing_umcu.readthedocs.io/en/latest/?badge=latest) |
| **GitHub Actions**                 | &nbsp; |
| Build                              | [![build](https://github.com/snorben/eeg_preprocessing_umcu/actions/workflows/build.yml/badge.svg)](https://github.com/snorben/eeg_preprocessing_umcu/actions/workflows/build.yml) |
| Citation data consistency          | [![cffconvert](https://github.com/snorben/eeg_preprocessing_umcu/actions/workflows/cffconvert.yml/badge.svg)](https://github.com/snorben/eeg_preprocessing_umcu/actions/workflows/cffconvert.yml) |
| SonarCloud                         | [![sonarcloud](https://github.com/snorben/eeg_preprocessing_umcu/actions/workflows/sonarcloud.yml/badge.svg)](https://github.com/snorben/eeg_preprocessing_umcu/actions/workflows/sonarcloud.yml) |
| MarkDown link checker              | [![markdown-link-check](https://github.com/snorben/eeg_preprocessing_umcu/actions/workflows/markdown-link-check.yml/badge.svg)](https://github.com/snorben/eeg_preprocessing_umcu/actions/workflows/markdown-link-check.yml) |
