# Stochastic Optimal Control for Neural Oscillators

This repository contains the MATLAB post-processing code and sample data for analyzing stochastic Hodgkin-Huxley neural networks under event-based control strategies.

## Repository Contents

- MATLAB scripts for post-processing HJB solutions
- Visualization tools for control signals and system dynamics
- Sample data files for representative noise levels
- Analysis pipeline for both single neuron and population studies

## Requirements

- MATLAB R2020a or newer
- MATLAB Statistics and Signal Processing Toolbox
- MATLAB Optimization Toolbox

## Getting Started

1. Clone the repository
2. Add all subfolders to MATLAB path:
   ```matlab
   addpath(genpath('code'));
   addpath(genpath('data'));
   ```
3. Run example analysis:
   ```matlab
   cd code/main_scripts
   main_HH2D_stochastic
   ```

## Code Structure

- `code/main_scripts/`: Main analysis scripts
- `code/visualization/`: Plotting and visualization tools
- `code/functions/`: Core analysis functions
- `data/sample_data/`: Representative simulation outputs

## Citation

If you use this code in your research, please cite:
```
[Citation information will be added upon publication]
```

## License

MIT License. See [LICENSE](LICENSE) file for details.