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
