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
