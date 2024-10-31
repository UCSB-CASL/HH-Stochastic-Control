# Stochastic Optimal Control for Neural Oscillators

[![DOI](https://zenodo.org/badge/881074337.svg)](https://doi.org/10.5281/zenodo.14015304)

This repository contains the MATLAB post-processing code and data for analyzing stochastic Hodgkin-Huxley neural networks under event-based control strategies. The code implements numerical solutions of Hamilton-Jacobi-Bellman equations and provides tools for analyzing both single neuron and population-level dynamics.

## Repository Contents

- MATLAB scripts for post-processing HJB solutions and optimal control analysis
- Visualization tools for control signals, phase space trajectories, and system dynamics
- Complete data files including HJB solutions (`phi_*.dat`) and optimal control signals (`uStar_*.dat`)
- Analysis pipeline for both single neuron and population studies with various noise levels
- Monte Carlo simulation capabilities for stochastic systems

## Requirements

- MATLAB R2020a or newer
- MATLAB Statistics and Signal Processing Toolbox
- MATLAB Optimization Toolbox

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/faranakR/HH-Stochastic-Control.git
2. Add all subfolders to the MATLAB path:
addpath(genpath('code'));
addpath(genpath('__Output'));
3. Run example analysis:
cd code/main_scripts
main_HH2D_stochastic

## Code Structure

code/main_scripts/: Main analysis scripts for running simulations
code/visualization/: Tools for plotting and visualization of results
code/functions/: Core analysis functions including:
HJB solution processing
Stochastic integration
Monte Carlo simulations
Event-based control implementation
__Output/: Complete simulation data for various noise levels
Data Description
The __Output directory contains:

D_*/: Subdirectories for different noise levels
phi_*.dat: Cost-to-go function solutions
uStar_*.dat: Optimal control signals
timeMat.txt, timeMat2.txt: Time evolution data

## Citation
If you use this code in your research, please cite:

[Citation information will be added upon publication]
## License
MIT License. See the LICENSE file for details.

## Contact
For questions about the code, please open an issue on GitHub.

Release Information

Version: v1.0
Author: @faranakR

