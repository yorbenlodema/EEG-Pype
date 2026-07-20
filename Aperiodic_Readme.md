### Overview

The EEG-Pype Aperiodic Analysis Tool is a separate GUI-based application for analysing the shape of the EEG power spectrum using Specparam. The module separates the power spectrum into:

* The **aperiodic component**: the broad background pattern or slope across frequencies.
* The **periodic component**: peaks above this background activity, such as an alpha-frequency peak.

The module calculates these measures both for a spectrum averaged across all valid channels or source regions (the global spectrum) and for each individual channel or source region. It can analyse both sensor-level and source-level EEG-Pype output.

As with the quantitative analysis module, it is advisable to run this script from Anaconda Prompt or Terminal rather than from an IDE such as Spyder. This avoids possible problems between the GUI, plotting and parallel processing.

For the current version, Specparam 2.0.0rc7 is required. If this version is not installed yet, first activate the EEG-Pype environment and run:

```bash
python -m pip install specparam==2.0.0rc7
````

To start the module:

```bash
conda activate EEG-Pype
cd path_to/EEG-Pype/src
python eeg_aperiodic_analysis.py
```

### Data Requirements

* Select the main output folder of a completed EEG-Pype preprocessing batch. You do not need to select the individual EEG or epoch files.
* The module recursively searches this folder for EEG-Pype `.txt` output.
* Both separately saved EEG epochs and continuous preprocessed EEG files can be analysed.
* Sensor-level and source-level data are treated as separate analysis profiles.
* Different source-reconstruction methods, atlases and cortical/full region selections are also kept separate. Select and analyse one profile at a time.
* The module will preferably use the unfiltered EEG output. If this is unavailable, it will use the widest filtered output that contains the complete selected fitting range.
* The sampling frequency and epoch length are normally read from the `.pkl` configuration file saved during preprocessing. Always check that the detected values are correct before starting the analysis.

### Step 1: Input and Data Level

#### Preprocessing batch folder and Scan data

Select the preprocessing output folder and click **Scan data**. The data inventory will show:

* The number of files and subjects found.
* Whether sensor-level or source-level data were detected.
* Whether the input consists of saved epochs or continuous files.
* Which filtered and unfiltered versions are available.
* The sampling frequency and epoch length detected from the preprocessing configuration.

#### Analysis profile

Use the **Analysis profile** dropdown menu to select the data you want to analyse. Examples include:

* Sensor level – epoched
* Sensor level – continuous
* Source: beamformer – Desikan – cortical
* Source: eLORETA – AAL2 – cortical

Sensor and source profiles should not be combined in one analysis. Run the module again for another profile when necessary. Each profile receives its own output folder.

#### Sampling frequency

The sampling frequency is normally filled in automatically from the preprocessing configuration. It can be changed manually if the configuration file is unavailable or if the detected value is incorrect.

A correct sampling frequency is necessary for calculating the frequency axis of the power spectrum.

#### Header handling

The options are:

* **Auto**: EEG-Pype tries to determine whether the first row contains channel or region names.
* **Yes**: Always treat the first row as a header.
* **No**: Treat every row as EEG data and generate channel names automatically.

The default **Auto** option should work for standard EEG-Pype output.

#### Window/epoch length

For saved epoch files, this should correspond to the length of the saved epochs.

For continuous files, the recording is internally divided into windows of this length. The module calculates a separate power spectrum for each epoch or window and averages these spectra before fitting the Specparam model.

#### Continuous overlap

This setting is only relevant for continuous recordings. It determines how much consecutive windows overlap.

For example:

* 0% means that the windows do not overlap.
* 50% means that half of each window is reused in the next window.

Increasing the overlap produces more windows but also increases calculation time.

#### Limit epochs/windows

This option can be used to analyse only a fixed number of epochs or windows per subject.

* **Maximum** sets the maximum number used.
* **Seed** makes the random selection reproducible. Using the same seed will select the same windows again.

This can be useful for testing the settings on a small amount of data or for using an equal maximum amount of data per subject. Leave this option unchecked to use all available epochs or windows.

### Step 2: PSD and Specparam Settings

### Multitaper PSD settings

The module first calculates a power spectral density (PSD) using MNE's multitaper method. For most analyses, the default settings are a suitable starting point.

#### Bandwidth

The multitaper bandwidth controls how strongly the power spectrum is smoothed.

* A larger bandwidth produces a smoother spectrum but can merge nearby peaks.
* A smaller bandwidth retains more frequency detail but may produce a noisier spectrum.
* **Auto** lets MNE select the bandwidth based on the epoch or window length.

The text below the setting shows the expected frequency resolution, smoothing and approximate number of tapers.

#### Adaptive weighting

When enabled, the individual multitaper estimates are given different weights before they are combined. This may improve the spectral estimate in some situations but is slower.

For most users, leaving this option disabled is a reasonable starting point.

#### Low-bias tapers

When enabled, only the tapers that are best concentrated within the selected bandwidth are used. This option is enabled by default and can normally be left on.

#### Remove DC

This removes the average signal value before calculating the power spectrum. This helps prevent a constant signal offset from affecting the lowest frequencies and should normally remain enabled.

#### Normalization

The available options are **full** and **length**. The default is **full**.

Normalization mainly affects the absolute scaling of the power spectrum and therefore the aperiodic offset. It is advisable to use the same normalization setting for all datasets that will be compared.

### Specparam fit settings

#### Fit range

This is the frequency range over which the Specparam model is fitted. The default range is 3–40 Hz.

The fitting range should:

* Fall within the frequency range available in the selected EEG output.
* Avoid frequencies very close to a high-pass or low-pass filter edge.
* Be wide enough to estimate the background spectral shape.
* Not extend above half the sampling frequency.

The module will provide a warning when the requested range is too close to a detected filter edge.

#### Aperiodic mode

The options are:

* **Fixed**: Fits the background spectrum without a bend. This is the default and is suitable when the background spectrum is approximately straight over the selected range.
* **Knee**: Allows the background spectrum to contain a bend or “knee”. This can be useful for wider frequency ranges or clearly curved spectra.

Knee mode adds an additional fitted parameter and generally needs a reasonably broad fitting range. The preview should be used to check whether it describes the data better than fixed mode.

#### Periodic mode

This determines the shape used to describe spectral peaks.

* **Gaussian**: Fits approximately symmetrical peaks. This is the default and is recommended as the normal starting point.
* **Skewed Gaussian**: Allows peaks to be asymmetrical.
* **Cauchy**: Allows peaks with a somewhat different, broader-tailed shape.

The alternative modes are mainly intended for sensitivity analyses or spectra in which the Gaussian model clearly fits the peaks poorly. Changing the periodic mode can also change the estimated peak parameters.

#### Peak width limits

These settings specify the minimum and maximum width of peaks that may be detected.

* A minimum that is too small may allow small irregularities or noise to be fitted as peaks.
* A minimum that is too large may miss narrow peaks.
* The maximum prevents very broad parts of the spectrum from being classified as one peak.

The lower limit should not be smaller than the effective frequency resolution shown in the GUI. The program warns when it is approximately less than twice the frequency-bin spacing.

#### Maximum peaks

This is the maximum number of peaks Specparam may detect in one spectrum.

A lower value makes the model more conservative. A very high value increases the possibility that small irregularities in the spectrum are fitted as peaks. The default of 6 is a reasonable starting point for a 3–40 Hz analysis.

Setting this value to 0 disables peak fitting and fits only the aperiodic component.

#### Minimum peak height

This is an absolute minimum for how far a peak must rise above the fitted aperiodic background.

Increasing this value results in fewer, more prominent peaks being detected. The appropriate value can depend on the scaling and quality of the data, so the preview should be inspected when changing it.

#### Peak threshold

This is a relative threshold based on variation in the spectrum after the background has been removed.

Increasing the threshold makes peak detection more conservative. The default value is 2 standard deviations.

Both the minimum peak height and peak threshold are applied during peak detection. A candidate peak must meet both requirements.

#### Line-noise interpolation

This option can be used to replace a narrow line-noise frequency range before fitting. For example:

```text
49-51
```

Multiple intervals can be entered using a semicolon or comma:

```text
49-51; 99-101
```

Only intervals overlapping the selected fitting range are used. The frequencies inside an interval are replaced using the surrounding spectrum.

This option is useful when a line-noise notch or residual line-noise peak would otherwise distort the fitted background or be detected as a physiological peak. Do not enter a broad frequency interval unnecessarily.

#### Post-hoc peak bands

These bands are used to organize the detected peaks after fitting. They do not cause Specparam to fit each frequency band separately.

The default setting is:

```text
delta:0.5-4; theta:4-8; alpha:8-13; beta:13-30
```

The output reports the detected peaks and identifies the strongest peak within each selected band. Additional bands can be added using the same format:

```text
name:lower-upper
```

### Visual QC warnings only

The quality-control thresholds are used to highlight fits that may require inspection. They do not automatically remove a channel, source region or subject from the results.

#### R² warning

R² describes how closely the fitted model follows the overall shape of the observed spectrum. Higher values generally indicate a closer fit.

The default warning threshold marks fits with an R² below 0.90.

#### MAE warning

Mean absolute error (MAE) describes the average difference between the fitted and observed spectrum. Lower values generally indicate a closer fit.

The default warning threshold marks fits with an MAE above 0.10.

#### Exponent warning range

This marks fitted aperiodic exponents outside the selected lower and upper values.

The default range of 0–4 is intended as a broad visual warning range rather than a scientific inclusion criterion. Adjust this range when necessary for a particular dataset or population.

Failed or warned fits remain included in the detailed output and are recorded in the quality-control table.

### Step 3: Preview, Output and Execution

### Preview

It is strongly advisable to preview the model before running the complete batch.

The available preview selections are:

* **First N subjects**: Uses the first subjects found in the selected profile.
* **Random N subjects**: Randomly selects subjects. The seed makes this selection reproducible.
* **All subjects**: Previews every subject. This can take a long time for a large dataset.

After clicking **Prepare & review preview**, a separate review window opens. This window allows you to:

* View the observed power spectrum, the complete fitted model and the aperiodic component.
* View the residual difference between the data and the model.
* Switch between the global spectrum and individual channels or source regions.
* Move to the previous or next fit.
* Jump directly to the fit with the lowest R².
* Jump directly to the fit with the highest MAE.
* Review fitted parameters and warning messages.

Change the settings and repeat the preview until the fits are satisfactory. When settings are changed after the last preview, the program will warn before starting the full batch.

Prepared power spectra are cached. Therefore, changing only the Specparam fitting settings usually allows a faster repeated preview without recalculating all power spectra.

### Output options

#### Output folder name

The default output folder is:

```text
aperiodic_analysis
```

A separate subfolder is created for each sensor or source analysis profile.

#### Save detailed CSV files

This saves the complete machine-readable results, including:

* A subject-level summary.
* Every global, channel and source-region fit.
* Every detected peak.
* Peaks organized by the selected post-hoc frequency bands.
* Fit warnings and failures.
* Data inventory and PSD settings.
* Run settings.

#### Save Excel workbook

This saves the main results in one Excel workbook with separate sheets. This is often easier for manually inspecting results.

The workbook distinguishes between:

* Parameters fitted to the globally averaged spectrum.
* The average of parameters fitted separately to individual channels or regions.

These are different calculations and should not be interpreted as interchangeable.

#### Save native Specparam JSON for global fits

This saves the original Specparam result object for each global spectrum. This is mainly useful for advanced users who want to reload the fitted models directly in Python.

This option currently applies to global fits only.

#### PNG scope

The available options are:

* **No PNG output**: Does not save fit figures.
* **Global fits only**: Saves one plot for the global spectrum of each subject.
* **Global + warned/failed**: Saves all global plots and any channel or region plots with warnings or failed fits.
* **All fits**: Saves every global, channel and region plot.

Saving all plots can create a very large number of files, especially for high-density EEG or source-level analyses.

#### PNG DPI

This determines the image resolution of saved figures. The default of 300 DPI is suitable for most reports and scientific figures.

#### Processing workers

This determines how many subjects are fitted in parallel.

Using more workers can reduce calculation time but also increases memory and processor use. It is advisable to leave one or two CPU cores free for other computer tasks. Reduce the number of workers if the computer becomes slow or the analysis runs into memory problems.

### Running and Reviewing the Full Batch

After reviewing the preview, click **Run full batch & export**.

The module will:

1. Prepare and cache the epoch-averaged or window-averaged power spectra.
2. Fit the global spectrum for each subject.
3. Fit every individual channel or source region.
4. Retain and report any warned or failed fits.
5. Save the selected tables, plots and reproducibility information.

The progress bar and run log show which stage is currently being performed.

The global spectrum is calculated as the arithmetic mean across channels or source regions that contain a valid power spectrum within the fitting range. A flat or otherwise invalid individual channel is recorded as a failed fit but does not automatically remove the complete subject.

### Output and Reproducibility Files

In addition to the selected CSV, Excel and PNG output, the module saves:

* The exact Specparam settings used.
* Package and software versions.
* A run manifest containing the selected profile and analysis settings.
* PSD metadata.
* A detailed log file.
* Information on fit warnings, failed fits and invalid channels or regions.

These files should be kept with the analysis results to make the analysis reproducible.

### Usage Tips

1. Start by scanning the complete preprocessing batch folder.
2. Check the detected sampling frequency and epoch/window length.
3. Analyse sensor and source profiles separately.
4. Start with the default multitaper and Specparam settings.
5. Use a small preview sample first.
6. Inspect both good and poor-fitting spectra before running the full batch.
7. Do not use the QC warning thresholds as automatic exclusion rules.
8. Use the same PSD normalization and main fitting settings when comparing groups or datasets.
9. Be careful when comparing the aperiodic offset between sensor-level and source-level data, since these data can have different power scaling.
10. Reduce the number of processing workers or saved PNG plots if memory use becomes too high.

```

The descriptions of the Specparam controls follow its official 2.0.0rc7 API: the model combines aperiodic and periodic components, supports fixed/knee aperiodic modes and Gaussian, skewed-Gaussian and Cauchy peak modes, and uses peak-width, peak-count and peak-height thresholds to control peak detection. :contentReference[oaicite:2]{index=2} The descriptions of the multitaper controls follow MNE’s definitions of bandwidth, adaptive weighting, low-bias tapers, normalization and DC removal. :contentReference[oaicite:3]{index=3}

One repository detail: `specparam` is not yet included in the current `pyproject.toml` dependencies, so the one-time installation paragraph is presently necessary. Once `specparam==2.0.0rc7` is added to `pyproject.toml` and `Environment.yml`, that paragraph can be shortened or removed. :contentReference[oaicite:4]{index=4}
```

[1]: https://github.com/yorbenlodema/EEG-Pype "GitHub - yorbenlodema/EEG-Pype · GitHub"
