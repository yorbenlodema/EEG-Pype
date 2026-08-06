# EEG-Pype Aperiodic / Specparam Analysis

This module can be used to analyse the aperiodic, or **1/f-like**, part of the EEG power spectrum and the periodic peaks that occur on top of it.

It works with `.txt` files created by EEG-Pype and can analyse:

* sensor-level EEG;
* source-level EEG;
* saved EEG epochs;
* continuous EEG;
* individual channels or source regions;
* the global average spectrum.

The current module version is **0.1.3** and is designed for output from **EEG-Pype v4.6.0**.

## Installation

The module requires Specparam version `2.0.0rc7`.

It is automatically installed when installing EEG-Pype for the first time. You can install it separately using:

```bash
pip install specparam==2.0.0rc7
```

The script checks the installed Specparam version when it starts. Other Specparam versions are currently not supported because the package interface may differ.

## Running the module

Run the script from the EEG-Pype `src` folder:

```bash
python eeg_aperiodic_analysis.py
```

The graphical interface will open automatically.

## Basic workflow

### 1. Select the preprocessing batch

Select the main EEG-Pype batch folder and click **Scan data**.

The module searches this folder and its subfolders for EEG-Pype `.txt` output.

### 2. Select an analysis profile

The available data are divided into separate analysis profiles. For example:

* sensor-level EEG;
* beamformer data using the Desikan atlas;
* eLORETA data using the Desikan atlas;
* cortical AAL2 beamformer data.

Only one profile is analysed per run. This prevents data from different sensor or source configurations from being combined accidentally.

### 3. Check the settings

The default settings are intended as a reasonable starting point:

| Setting                     |   Default |
| --------------------------- | --------: |
| Sampling-frequency fallback |    256 Hz |
| Epoch or window length      | 8 seconds |
| Continuous-data overlap     |        0% |
| Specparam fit range         |   1–30 Hz |
| Aperiodic mode              |     Fixed |
| Periodic peak shape         |  Gaussian |
| Peak-width limits           |    1–8 Hz |
| Maximum number of peaks     |         6 |
| Minimum peak height         |      0.10 |
| Peak threshold              |    2.0 SD |

The sampling-frequency field is only a fallback. When valid EEG-Pype JSON metadata are available, the sampling frequency stored in the metadata is used instead.

### 4. Prepare a preview

Click **Prepare & review preview** before running the full batch.

The preview allows you to inspect fits from:

* the first selected subjects;
* a random selection of subjects;
* all subjects.

The review window can also take you directly to the fit with the lowest R² or highest error.

Change the settings and repeat the preview when the fits do not adequately follow the spectra.

### 5. Run the full analysis

Click **Run full batch & export**.

The results are saved inside the selected preprocessing batch folder:

```text
<batch folder>/<output folder>/<analysis profile>/
```

A separate output folder is therefore created for each analysis profile.

## Selecting the EEG input

The module prefers unfiltered EEG-Pype output when this is available.

When no unfiltered output is available, it selects the widest saved frequency range that fully covers the requested Specparam fit range.

For example, a 3–40 Hz fit cannot be performed using a file filtered to 8–13 Hz. That file will not be selected for the analysis.

## Saved epochs and continuous EEG

Saved epoch files are treated as separate EEG segments.

Continuous EEG is divided into internal windows using the selected:

* window length;
* overlap;
* optional maximum number of windows;
* random seed.

Limiting the number of epochs or windows can make exploratory analyses faster. The random seed makes the selection reproducible.

## Multitaper power spectra

A multitaper power spectrum is calculated for every epoch or continuous-data window.

The default **Auto** bandwidth is suitable for most users. The interface shows an estimate of:

* frequency resolution;
* spectral smoothing;
* the approximate number of tapers.

The module warns when the selected PSD settings are poorly matched to the window length or requested peak widths.

Power spectra are averaged across epochs or windows in ordinary linear power before Specparam is applied.

## Aperiodic models

Two aperiodic models are available:

### Fixed

The fixed model estimates:

* offset;
* exponent.

This is generally the simplest starting point when the spectrum is approximately straight on a log-log scale.

### Knee

The knee model also allows curvature in the aperiodic spectrum.

Use this when the preview shows clear bending that cannot be described adequately by the fixed model. The knee model contains an additional parameter and may be less stable over a limited frequency range.

## Periodic peaks

The periodic model describes peaks above the aperiodic background.

Available peak shapes are:

* Gaussian;
* skewed Gaussian;
* Cauchy.

Gaussian is the default and is the most straightforward starting point.

Detected peaks are also assigned to the following default bands:

```text
delta: 0.5–4 Hz
theta: 4–8 Hz
alpha: 8–13 Hz
beta: 13–30 Hz
```

These bands can be changed in the interface. Band assignment is performed after the peaks have been fitted.

## Line-noise interpolation

Known line-noise intervals can be entered before fitting, for example:

```text
49-51
```

Use this only when the selected fit range includes a clear line-noise disturbance. Leave the field empty when line-noise interpolation is not needed.

## Global and channel-level results

The module fits:

1. a global spectrum created by averaging the valid channel or region spectra;
2. every valid channel or source region separately.

These provide different results.

The parameters fitted to the global average spectrum are **not equivalent** to averaging parameters from separately fitted channels or regions. Both types of result are therefore saved.

## Quality-control warnings

The default warning thresholds are:

| Measure  |     Warning |
| -------- | ----------: |
| R²       |  below 0.90 |
| MAE      |  above 0.10 |
| Exponent | outside 0–4 |

These are warning thresholds, not automatic exclusion criteria.

Warned fits remain in the output. Failed fits are also retained and marked as failed, rather than being silently removed.

Always inspect the fit plots and QC columns before interpreting the results.

## EEG-Pype metadata

EEG-Pype v4.6.0 writes an `eeg_pype_export_metadata.json` file alongside each recording.

When complete and internally consistent, this metadata can provide:

* sampling frequency;
* epoch length;
* continuous or epoched representation;
* channel or region names;
* reference information;
* filter range;
* export scaling.

Older EEG-Pype output without JSON metadata can still be analysed. The module then uses information from the filenames and the manually entered fallback settings.

Metadata warnings are included in the output when information is missing, incomplete or inconsistent.

## Output

The Excel workbook is usually the easiest place to start. It contains sheets for:

* **Subject Summary** – main results per subject;
* **Fits** – all global, channel and region fits;
* **Peaks** – all detected periodic peaks;
* **Band Peaks** – peaks assigned to the selected frequency bands;
* **Fit QC** – warnings and failed fits;
* **Data Inventory** – input files found and selected;
* **PSD Metadata** – PSD and preprocessing information;
* **Run Settings** – settings used for the analysis.

The module can also save the same information as separate CSV files.

Additional output includes:

* package versions;
* Specparam settings;
* an analysis manifest;
* an analysis log;
* optional PNG fit plots;
* optional native Specparam JSON files for global fits.

By default, the module saves:

* the Excel workbook;
* detailed CSV files;
* PNG plots of global fits.

## Practical recommendations

* Start with the default settings and inspect a preview.
* Use the same fit range and model settings for subjects that will be compared.
* Do not interpret warning thresholds as formal exclusion rules.
* Inspect fits with low R², high error or unusual exponent values.
* Use the fixed model unless the spectra show clear curvature.
* Keep sensor and different source-analysis profiles separate.
* Record any settings changed from the defaults when reporting the analysis.
