# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Version 4.3: 
### Preprocessing module
Changed title in GUI window to EEG-Pype, added clean PSD plot function which automatically ignores the most high-    amplitude epochs to directly get a cleaner PSD plot, updated logging to the batch log file to work intermittently (in between EEGs) to make sure logging is saved in the event of a crash mid-batch.
### Quantitative analysis module
Solved a bug: "mean of empty slice" which was caused by NaNs in the power calculation.

## Version 4.3.2:
Added explicit data rank to spatial filter function to remove problem of data rank deficiency when discarding more than 1 ICA component in combination with LCMV beamforming.

## Version 4.4.0:
Added calculation of Phase Lag Time (PLT) to the quantitative analysis script, similar to the calculation in BrainWave (by C.J. Stam). 

## Version 4.4.1:
Added an option to select a custom (filtering) frequency band in the GUI during setup of preprocessing. If left empty, this option does noting. The quantitative analysis module was expanded to include an option to limit the number of epochs used per participant. 

## Version 4.5.0
This version includes more options for source reconstructing the EEG during preprocessing. These now also include eLORETA, sLORETA, MNE and dSPM. In addition, this version included improved .pkl rerunning, inlcuding the option to easily select only several raw EEG files to rerun from a previously ran batch. At the same time, the raw EEG files used previously do not necessarily need to be in the same exact path as when initially running the batch since the user is now prompted to always reselect the raw EEG files via a file browser window. Finally, integer conversion prior to (J)PE calculation was removed since this was a legacy feature meant for comparison with BrainWave that was not used anymore in practice.

## Version 4.5.1
This version added an implementation of MNE's minimum-norm estimate source reconstruction to the recently expanded source reconstruction options. In addition, in the quantitative analysis module, the option was added to save per epoch (so not epoch-averaged) quantitative measures in the final Excel output, in long format. This allows for more fine-grained follow-up analyses, like looking at variability of measures over epochs. At the same time, we smoothed out a discrepancy by outputting the connectivity and MST matrices per epoch. While this does require more disk space, this also allows the user more control in subsequent analyses. 

## Version 4.5.2
This version added more granular options to save either a combination of or in isolation: unfiltered, broadband and band filtered output. At the same time, the quantitative analysis module now provides the option to export the PSD as a plot in PNG format, or in CSV format per region/channel and/or averaged. This allows for further analyses and visualisation outside of EEG-Pype. Finally, also in the quantitative analysis module, the power bands used for all spectral (PSD) measures no longer overlap. Before, the upper limit of one frequency band and the lower limit of the next band would be the same frequency bin. Depending on frequency resolution, this could lead to more or less distortion of power metrics, resulting in a total relative power larger than 1.0. This has now been changed to making the upper limit of each frequency band the frequency bin before the cutoff frequency.

## Version 4.6.0

### Added

* Added a recording-level `eeg_pype_export_metadata.json` sidecar for sensor and source exports.
* Added metadata for preprocessing settings, filters, references, channels or regions, export scaling, software versions, ICA diagnostics and ICLabel results.
* Added metadata states: `in_progress`, `complete` and `failed`.
* Added a standalone aperiodic and periodic spectral-parameterisation module using `specparam==2.0.0rc7`.
* Added multitaper PSD estimation, global and channel/region-level Specparam fitting, preview tools, QC warnings and reproducibility exports.
* Added automatic preprocessing filtering pass-band discovery to the quantitative-analysis module.

### Changed

* ICA now uses deterministic extended infomax → more robust and matches ICLabel expectations.
* Common-average reference is applied after bad-channel selection and before ICA → better ICLabel matching and less chance of problems during ICA with empty reference channels.
* ICA is fitted before bad-channel interpolation.
* Common-average reference is repeated after interpolation over the complete channel set.
* Selecting ICA or source reconstruction automatically enables common-average reference.
* ICA is fitted on a separate 1–100 Hz or Nyquist-limited working copy → better ICLabel matching.
* The ICA copy may be downsampled to 250 or 256 Hz when sufficient samples remain.
* The fitted ICA component count is limited by the user setting and measured EEG rank.
* Relative-power and spectral-variability calculations now use the detected preprocessing pass-band rather than relying on Nyquist alone.

### Fixed

* Prevented metadata-writing errors from stopping EEG preprocessing.
* Prevented one failed recording from terminating the remaining batch.
* Prevented NumPy and non-finite values from making metadata sidecars unwritable.
* Prevented stale ICLabel results from previous reruns from being exported.
* Improved consistency between the reference used for ICA, epoch selection and final export.
* Prevented relative-power denominators from silently including frequencies outside the preprocessing pass-band.

### Compatibility notes

* The ICA, interpolation and common-average-reference order has changed, so results may differ from earlier EEG-Pype versions.
* A sampling frequency of 250 or 256 Hz or higher is recommended for ICLabel.
* An exact 100 Hz ICA upper cutoff requires at least 204 Hz with the current transition-band settings.
* At lower sampling frequencies, EEG-Pype automatically applies a lower Nyquist-safe ICA cutoff and logs a warning.
* The aperiodic module currently requires `specparam==2.0.0rc7`.
* Legacy EEG-Pype text output without JSON metadata remains supported.

