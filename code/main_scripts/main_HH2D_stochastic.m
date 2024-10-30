%% main_HH2D_stochastic.m
% This script implements and analyzes event-based control strategies for
% Hodgkin-Huxley neural networks under stochastic conditions.
%
% Directory Structure:
% SDESolver/
% ├── main_HH2D_stochastic.m     % This file
% ├── functions/                 % General functions
% │   ├── sde_hh_model_solver.m
% │   ├── saturation.m
% │   ├── rk_hh.m
% │   ├── Montecarlo_hh.m
% │   ├── interpolate_general.m
% │   ├── gridGenerator.m
% │   ├── F_interpolant_u.m
% │   ├── EuclidianNorm.m
% │   ├── EBC_p_no_plot.m
% │   ├── EBC_p.m
% │   ├── CostandControl_traj.m
% │   └── model_specific/        % Model-specific functions
% │       ├── zdyn.m
% │       ├── terminal_penalty.m
% │       ├── hhapprox_noise.m
% │       ├── hhapprox.m
% │       ├── hh_model.m
% │       ├── hh_f_2d_p.m
% │       ├── func_hhapprox_coupled.m
% │       └── func_hhapprox.m
% └── __Output/                  % Output directory
%
% The code is organized into several main sections:
% 1. Parameter Setup & Initialization
% 2. Single Neuron Control Computation
% 3. Population Level Analysis
% 4. Robustness Analysis
% 5. Monte Carlo Simulations
% 6. Visualization & Analysis
%
% Key Features:
% - Implements both deterministic and stochastic control strategies
% - Analyzes three network types: homogeneous, heterogeneous, and sparse
% - Performs Monte Carlo simulations for robustness analysis
% - Generates comprehensive visualizations including histograms and boxplots
%
% Network Types:
% - Homogeneous: All connections have equal strength (α_ij = α)
% - Heterogeneous: Connection strengths follow normal distribution
% - Sparse Heterogeneous: 20% of connections are zero
%
% Parameters:
% - D_noise_vals: Noise intensity levels [0.5, 1, 5, 10, 15]
% - alpha_vals: Coupling strength values [0.05:0.05:0.25]
% - num_sims: Number of Monte Carlo simulations (default: 100)
% - nNeurons: Number of neurons in network (default: 100)
%
% Outputs:
% - Control signals and their integrals
% - Energy consumption metrics
% - Network performance metrics
% - Statistical distributions of control outcomes
%
% Key Functions Used:
% - gridGenerator.m: Generates a grid structure for numerical computation
% - F_interpolant_u.m: Creates interpolants for deterministic and stochastic control
% - zdyn.m: Defines the controlled Hodgkin-Huxley model dynamics
% - CostandControl_traj.m: Calculates cost and control trajectories
% - generate_alpha_matrix.m: Creates network connectivity matrices
% - EBC_p.m: Event-based control for population
% - EBC_p_no_plot.m: Non-plotting version of EBC
% - Montecarlo_hh.m: Monte Carlo simulation handler
% - plotHistogramWithTrendLine.m: Plots histograms with trend lines and performs goodness-of-fit tests
%
% Other Dependencies:
% - Various utility and plotting functions
%
% Author: Faranak Rajabi
% Date: October 24, 2024
% Version: 2
%
% Usage:
% Run the script sections sequentially. Each section is marked with %% and
% can be run independently if previous results are loaded.

clc; clear; close all;

%%% Initialize Global Variables
global nOfNodes Ks x_targ umax gama sigma D_det vspike nspike Dt Tend D_vals

%%% Path Setup
try
    % Locate base directory
    current_folder = pwd;
    [parent_folder, current_name] = fileparts(current_folder);
    
    % Find SDESolver directory
    if ~strcmp(current_name, 'SDESolver')
        if exist(fullfile(parent_folder, 'SDESolver'), 'dir')
            base_dir = fullfile(parent_folder, 'SDESolver');
        elseif exist('SDESolver', 'dir')
            base_dir = fullfile(current_folder, 'SDESolver');
        else
            error('Cannot find SDESolver directory. Please run from correct location.');
        end
    else
        base_dir = current_folder;
    end
    
    % Add required paths
    addpath(fullfile(base_dir, 'functions'));
    addpath(fullfile(base_dir, 'functions', 'model_specific'));
    addpath(fullfile(base_dir, '__Output'));
    
    % Verify critical functions exist
    required_functions = {'hh_model', 'zdyn', 'EBC_p', 'gridGenerator'};
    missing_functions = required_functions(~cellfun(@(x) exist(x, 'file'), required_functions));
    if ~isempty(missing_functions)
        error('Missing required functions: %s', strjoin(missing_functions, ', '));
    end
    
    fprintf('Paths setup successfully!\n');
catch ME
    fprintf('Error setting up paths:\n%s\n', ME.message);
    fprintf('Current directory: %s\n', pwd);
    error('Please ensure you are running from the correct directory.');
end

%%% Parameter Initialization
% Grid parameters
nOfNodes = 320;
Ks = 100;
xMin = -1; xMax = 1;
yMin = 0; yMax = 1;

% Model parameters
D_det = 0;           % Noise Coefficient
vspike = 44.8;       % mV
nspike = 0.459;
targ = [-59.6; 0.403];
gama = 1000;
sigma = 1e-3;
umax = 10;           % Constraint on u

% Time parameters
Tend = 7;
tInitial = 0;
tPlotEnd = 802;
Tsim = 60 * tPlotEnd;    % number of simulation time steps
Dt = Tend / (tPlotEnd - 1);  % forward integration time step
R = 1;
dt = Dt / R;             % time step for Brownian path
N = round(Tsim / dt);    % number of discrete Brownian values
M = 1;                   % number of paths sampled

% Population parameters
nNeurons = 100;
Vth = -20;              % Threshold mean voltage
Tth = 0;                % Threshold time
Ib = 10;                % Base line current

% Analysis parameters
alpha_vals = 0.05:0.05:0.25;
D_vals = [0.5, 1, 5, 10, 15];

%%% Grid Generation and Solution Setup
% Generate grid
grid_struct = gridGenerator(xMin, xMax, yMin, yMax, nOfNodes, nOfNodes);

% Solve for x_targ
options_fsolve = optimoptions('fsolve', 'FunctionTolerance', 1e-9, ...
    'OptimalityTolerance', 1e-9, 'StepTolerance', 1e-9);
x_targ = fsolve(@(x) hh_model(x), targ, options_fsolve);

% Generate Ibvec
Ibvec = Ib * ones(nNeurons, 1);

% Create simulation data structure
sim_data = struct('umax', umax, 'Ks', Ks, 'targ', targ, 'Tend', Tend, ...
    'sigma', sigma, 'nOfNodes', nOfNodes, 'R', R, 'Dt', Dt, 'N', N, ...
    'D_noise', [], 'M', M, 'tPlotEnd', tPlotEnd, 'nNeurons', nNeurons, ...
    'alpha', [], 'Vth', Vth, 'Tth', Tth, 'Tsim', Tsim, 'Ibvec', Ibvec, ...
    'D_new_values', D_vals, 'grid_struct', grid_struct, 'D_HJB', []);

% Generate F_ustar interpolants
[F_ustar_deterministic, F_ustar_stochastic] = F_interpolant_u(D_vals, grid_struct, Tend);

fprintf('Setting Parameters getting control vectors Done.\n');

%% Single Neuron Analysis
% Set ODE solver options for high accuracy
options = odeset('Reltol', 1e-10, 'AbsTol', 1e-9);

% Initialize voltage and gating variable initial conditions
Vpo = vspike;    % Initial voltage (mV)
npo = nspike;    % Initial gating variable
x_initial_periodic = [Vpo/Ks; npo; 0];  % Initial state vector [V/Ks; n; cost]

% Configure simulation parameters in sim_data
sim_data.control = 'off';
sim_data.F_ustar_deterministic = F_ustar_deterministic;
sim_data.F_ustar_stochastic = F_ustar_stochastic;

% Compute periodic orbit (uncontrolled system)
[t_periodic, x_periodic] = ode45(@(t,x) zdyn(t,x,sim_data), [0 Tend*2], x_initial_periodic, options);

% Compute deterministic control trajectory
sim_data.feedback = 'on';
sim_data.control = 'deterministic';
[t_deterministic, x_deterministic] = ode45(@(t,x) zdyn(t,x,sim_data), [0:Dt:Tend], x_initial_periodic, options);
[J_mat_D_0, u_deterministic_D_0] = CostandControl_traj(t_deterministic, x_deterministic, sim_data);
J_mat(1,:) = J_mat_D_0;

% Compute stochastic control trajectories
sim_data.control = 'Stochastic';

% Initialize results matrices
u_deter_Stoch_mat = zeros(length(t_deterministic), length(D_vals));
x_deter_Stoch_mat = zeros(length(t_deterministic), 3*length(D_vals));
t_deter_Stoch_mat = zeros(length(t_deterministic), length(D_vals));

% Loop through different noise intensities
for i = 1:length(D_vals)
    sim_data.D = D_vals(i);
    sim_data.index = i;
    
    [t_deter_uStoch, x_deter_uStoch] = ode45(@(t,x) zdyn(t,x,sim_data), ...
        [0:Dt:Tend], x_initial_periodic, options);
    
    [J_mat_deter_Stoch, u_deter_Stoch] = CostandControl_traj(t_deter_uStoch, ...
        x_deter_uStoch, sim_data);
    
    % Store results
    J_mat(i+1,:) = J_mat_deter_Stoch;                    % Cost matrix
    u_deter_Stoch_mat(:,i) = u_deter_Stoch;              % Control signals
    x_deter_Stoch_mat(:,3*(i-1)+1:3*i) = x_deter_uStoch; % State trajectories
    t_deter_Stoch_mat(:,i) = t_deter_uStoch;             % Time vectors
end

sim_data.t_periodic = t_periodic; 
sim_data.t_deterministic = t_deterministic; 
sim_data.u_deterministic = u_deterministic_D_0; 
sim_data.u_deter_Stoch_mat = u_deter_Stoch_mat; 
sim_data.t_deter_Stoch_mat = t_deter_Stoch_mat; 
sim_data.x_deter_Stoch_mat = x_deter_Stoch_mat; 

fprintf('Computing Control Signals Done.\n');

%% Monte Carlo Analysis of Control Energy Over Time
% Analyzes control energy (integral of u^2) for both feedback and non-feedback cases
% across different noise intensities using Monte Carlo simulation for a single neuron.

% Initialize simulation parameters
D_noise_vals = [0.5 1 5 10 15];     % Noise intensity values
num_runs = 10000;                   % Number of Monte Carlo iterations

% Update sim_data with required fields for Monte Carlo
sim_data.feedback = 'on';
sim_data.F_ustar_deterministic = F_ustar_deterministic;
sim_data.F_ustar_stochastic = F_ustar_stochastic;
sim_data.tInitial = tInitial;

% Preallocate result matrices
u_control_mat = cell(length(D_noise_vals), num_runs);
u_control_nofeedback_mat = cell(length(D_noise_vals), num_runs);
u_squared_integral_mat = cell(length(D_noise_vals), num_runs);
u_squared_integral_mat_nofeedback = cell(length(D_noise_vals), num_runs);

% Start Monte Carlo simulation
tic;
for n = 1:num_runs
    % Run with feedback control
    [~, result_once] = Montecarlo_hh(x_initial_periodic, sim_data, D_noise_vals, n, true);
    
    % Process feedback results
    for d = 1:length(D_noise_vals)
        u_control = result_once{d,1};
        u_control_mat{d, n} = u_control;
        u_squared_integral_mat{d, n} = cumsum(u_control.^2) * sim_data.Dt;
    end
    
    % Run without feedback
    sim_data.feedback = 'off';
    [~, result_once_nofeedback] = Montecarlo_hh(x_initial_periodic, sim_data, D_noise_vals, n, true);
    
    % Process non-feedback results
    for d = 1:length(D_noise_vals)
        u_control_nofeedback = result_once_nofeedback{d,1};
        u_control_nofeedback_mat{d, n} = u_control_nofeedback;
        u_squared_integral_mat_nofeedback{d, n} = cumsum(u_control_nofeedback.^2) * sim_data.Dt;
    end
    
    % Reset feedback parameter for next iteration
    sim_data.feedback = 'on';
    
    % Display progress
    if mod(n, 10) == 0
        fprintf('Completed %d/%d runs\n', n, num_runs);
    end
end

elapsedTime = toc;
fprintf('Monte Carlo simulation completed in %.2f seconds\n', elapsedTime);

%% Monte Carlo Analysis: Stochastic vs Deterministic Control
% Performs Monte Carlo analysis (10K iterations) for single neuron level,
% comparing stochastic and deterministic solutions on deterministic trajectories.

% Initialize simulation parameters
D_noise_vals = [0.5 1 5 10 15];
% Test run
% num_runs = 10; 
num_runs = 10000;

% Update sim_data for Monte Carlo
% We repeat some parameters set-up here, in case we don't want to compute
% previous sections. 
sim_data.feedback = 'on';
sim_data.F_ustar_deterministic = F_ustar_deterministic;
sim_data.F_ustar_stochastic = F_ustar_stochastic;
sim_data.u_deter_Stoch_mat = u_deter_Stoch_mat;
sim_data.t_deter_Stoch_mat = t_deter_Stoch_mat;
sim_data.u_deterministic = u_deterministic_D_0;
sim_data.t_deterministic = t_deterministic;

% Run Monte Carlo Simulations
% With feedback control
[mat_final, ~] = Montecarlo_hh(x_initial_periodic, sim_data, D_noise_vals, num_runs, false);

% Without feedback control
sim_data.feedback = 'off';
[mat_final_nofeedback, ~] = Montecarlo_hh(x_initial_periodic, sim_data, D_noise_vals, num_runs, false);

% Initialize Result Matrices
num_D_vals = length(D_noise_vals);
metrics = {'u', 'end_euclidean', 'end_exp', 'total_euclid', ...
          'total', 'v_end', 'n_end'};

% Initialize containers for all metrics
for metric = metrics
    % Stochastic trajectories
    eval(sprintf('J_%s_ST_mat = zeros(num_runs, num_D_vals);', metric{1}));
    eval(sprintf('J_%s_DT_mat = zeros(num_runs, num_D_vals);', metric{1}));
    
    % No feedback cases
    eval(sprintf('J_%s_no_feedback_ST_mat = zeros(num_runs, num_D_vals);', metric{1}));
    eval(sprintf('J_%s_no_feedback_DT_mat = zeros(num_runs, num_D_vals);', metric{1}));
end

% Process Results
for i = 1:num_D_vals
    % Index mappings for different metrics
    metric_indices = 1:7;  % Corresponds to columns in mat_final
    
    % Process each metric
    for j = 1:length(metrics)
        metric = metrics{j};
        idx = metric_indices(j);
        
        % Extract data for with-feedback cases
        eval(sprintf('J_%s_ST_mat(:,i) = mat_final{i,1}(:,%d);', metric, idx));
        eval(sprintf('J_%s_DT_mat(:,i) = mat_final{i,2}(:,%d);', metric, idx));
        
        % Extract data for no-feedback cases
        eval(sprintf('J_%s_no_feedback_ST_mat(:,i) = mat_final_nofeedback{i,1}(:,%d);', metric, idx));
        eval(sprintf('J_%s_no_feedback_DT_mat(:,i) = mat_final_nofeedback{i,2}(:,%d);', metric, idx));
    end
end

fprintf('Monte Carlo analysis completed for %d runs across %d noise levels\n', ...
    num_runs, num_D_vals);

% Optional: Save Results
% save('monte_carlo_results.mat', 'J_*_mat', 'D_noise_vals');

%% Event Based Control for Population
% Simulates population dynamics with event-based control for the specific case
% of D = 15 and alpha_ij as a scalar value across all neurons. Outputs match
% the EBC_P function results as shown in the paper.

% Define initial conditions
IC = [vspike nspike];    % Spike point initial condition 
% IC = [-59.6 0.403];    % Alternative: phase-less point

% Setup HJB noise values
D_HJB_vals = [0.0 0.5 1 5 10 15];

% Generate Alpha coupling matrix
index = 5;               % Index corresponding to D = 15
alpha = alpha_vals(index);
Alpha = generate_alpha_matrix(1, alpha, 0, sim_data.nNeurons);

% Clear previous run data and close plots
close all; 
clear uu tt param.D_HJB param.D_noise Vp_wcwn np_wcwn

% Set simulation parameters
sim_data.alpha = Alpha;
sim_data.D_noise = 15;  % Noise intensity for ODE
sim_data.D_HJB = 15;    % Noise intensity for HJB

% Select control signals 
% Option 1: Deterministic control
% uu = u_deterministic_D_0; 
% tt = t_deterministic;

% Option 2: Stochastic control (D = 15)
uu = u_deter_Stoch_mat(:, 5); 
tt = t_deter_Stoch_mat(:, 5);

% Run population simulation
[Vp_wcwn, np_wcwn, u_integral_plot] = EBC_p(sim_data, IC, uu, tt);

% Optional: Save control integral results
% u_integrated_str = sprintf('u_squared_D_HJB_%.2f_D_ODE_%.2f.mat', ...
%     sim_data.D_HJB, sim_data.D_noise);
% save(u_integrated_str, 'u_integral_plot');

%% Event Based Control for Population Histogram Analysis
% Performs Monte Carlo simulations to analyze energy costs between stochastic 
% and deterministic control strategies. Computes and stores the integral of u^2
% for population level control across multiple noise intensities.

% Initialize simulation parameters
D_HJB_vals = [0.0 0.5 1 5 10 15];     % D values for HJB equation
D_noise_vals = [0.5 1 5 10 15];       % D values for noise in ODE
IC = [vspike nspike];                 % Initial condition (spike point)
% test run
% num_sims = 5; 
num_sims = 100;                       % Number of Monte Carlo simulations
num_neurons = sim_data.nNeurons;      % Number of neurons in population

% Preallocate storage for simulation results
u_integral_plot_det_sims = cell(1, length(D_noise_vals));
u_integral_plot_stoch_sims = cell(1, length(D_noise_vals));

% Main simulation loop over noise intensities
for d = 1:length(D_noise_vals)
   % Monte Carlo iterations
   for n = 1:num_sims
       % Reset workspace for new iteration
       close all; 
       clear uu tt param.D_HJB param.D_noise Vp_wcwn np_wcwn param.alpha;
       
       % Generate coupling matrix for current noise level
       alpha_mean = alpha_vals(d);
       Alpha = generate_alpha_matrix(1, alpha_mean, 1, sim_data.nNeurons);
       sim_data.alpha = Alpha;
       sim_data.D_noise = D_noise_vals(d);
       
       % Simulate deterministic control case
       sim_data.D_HJB = D_HJB_vals(1);
       uu_det = u_deterministic_D_0;
       tt_det = t_deterministic;
       [u_integral_plot_det] = EBC_p_no_plot(sim_data, IC, uu_det, tt_det, ...
           n + (d-1) * num_sims);
       u_integral_plot_det_sims{1, d}(n, :) = u_integral_plot_det(2, :);
       
       % Simulate stochastic control case
       sim_data.D_HJB = sim_data.D_noise;
       uu_stoch = u_deter_Stoch_mat(:, d);
       tt_stoch = t_deter_Stoch_mat(:, d);
       [u_integral_plot_stoch] = EBC_p_no_plot(sim_data, IC, uu_stoch, tt_stoch, ...
           n + (d-1) * num_sims);
       u_integral_plot_stoch_sims{1, d}(n, :) = u_integral_plot_stoch(2, :);
   end
   
   fprintf('EBC(Energy Comparison): Completed %d Monte Carlo simulations for D = %.2f\n', ...
       num_sims, D_noise_vals(d));
end

%% Population Control Analysis: Robustness Study Across Network Types
% Analyzes control effectiveness and energy costs across different network structures
% (homogeneous, heterogeneous, sparse) for varying noise intensities.

%%% Initialize Parameters
D_noise_vals = [0.5 1 5 10 15];
alpha_vals = 0.05:0.05:0.25;
% test run
% num_sims = 5; 
num_sims = 100;
IC = [vspike, nspike];
case_names = {'Homogeneous', 'Heterogeneous', 'Sparse Heterogeneous'};

% Preallocate storage arrays
u_integral_results = cell(length(D_noise_vals), 3, num_sims);
control_count_results = zeros(length(D_noise_vals), 3, num_sims);

%%% Run Network Simulations
for d = 1:length(D_noise_vals)
   alpha_mean = alpha_vals(d);
   
   % Test each network type
   for alpha_case = 1:3
       % Monte Carlo iterations
       for n = 1:num_sims
           % Generate network structure
           realization_seed = 200 * (d + alpha_case * 10 + n * 100);
           Alpha = generate_alpha_matrix(alpha_case, alpha_mean, realization_seed, ...
               sim_data.nNeurons);
           
           % Configure simulation parameters
           sim_data.alpha = Alpha;
           sim_data.D_noise = D_noise_vals(d);
           sim_data.D_HJB = D_noise_vals(d);
           
           % Run simulation
           uu_stoch = u_deter_Stoch_mat(:, d);
           tt_stoch = t_deter_Stoch_mat(:, d);
           [u_integral_plot, control_count] = EBC_p_no_plot(sim_data, IC, ...
               uu_stoch, tt_stoch, n + (d-1)*num_sims + ...
               (alpha_case-1)*length(D_noise_vals)*num_sims);
           
           % Store results
           u_integral_results{d, alpha_case, n} = u_integral_plot;
           control_count_results(d, alpha_case, n) = control_count;
       end
       
       fprintf('Completed %d simulations: D=%.1f (α=%.2f), %s network\n', ...
           num_sims, D_noise_vals(d), alpha_mean, case_names{alpha_case});
   end
end

%%% Process and Analyze Results
% Get time step from simulation data
Dt = tt_stoch(2) - tt_stoch(1);

% Calculate average metrics
avg_energies = zeros(length(D_noise_vals), 3);
avg_control_times = zeros(length(D_noise_vals), 3);

for d = 1:length(D_noise_vals)
   alpha_mean = alpha_vals(d);
   for alpha_case = 1:3
       % Calculate energy and control metrics
       final_energies = cellfun(@(x) x(2,end), u_integral_results(d, alpha_case, :));
       avg_control_count = mean(control_count_results(d, alpha_case, :));
       
       % Store averages
       avg_energies(d, alpha_case) = mean(final_energies);
       avg_control_times(d, alpha_case) = avg_control_count * Dt;
   end
end

%%% Display Results
fprintf('\n=== Analysis Results ===\n');
fprintf('Time step (Dt) = %.4f\n\n', Dt);

% Print detailed table
fprintf('%-5s %-8s %-22s %-12s %-12s\n', 'D', 'α', 'Network Type', 'Energy', 'Control Time');
fprintf('%s\n', repmat('-', 1, 65));

for d = 1:length(D_noise_vals)
   for alpha_case = 1:3
       fprintf('%-5.1f %-8.2f %-22s %-12.2f %-12.2f\n', ...
           D_noise_vals(d), alpha_vals(d), case_names{alpha_case}, ...
           avg_energies(d, alpha_case), avg_control_times(d, alpha_case));
   end
   fprintf('%s\n', repmat('-', 1, 65));
end

%%% Save Results
save('network_control_analysis.mat', ...
   'control_count_results', 'u_integral_results', ...
   'avg_energies', 'avg_control_times', ...
   'D_noise_vals', 'alpha_vals', 'Dt', 'case_names');

%%% Visualization Functions
plotControlAnalysis(u_control_mat, u_control_nofeedback_mat, ...
   u_deter_Stoch_mat, params, D_noise_vals);

%% Helper functions
function [mean_val, var_val] = plotHistogramWithTrendLine(data, line_style, marker, color, legend_text)
    [counts, edges] = histcounts(data(:), 'Normalization', 'pdf');
    centers = (edges(1:end-1) + edges(2:end)) / 2;
    plot(centers, counts, 'LineStyle', line_style, 'Marker', marker, 'Color', color, 'LineWidth', 1, 'MarkerSize', 2, 'DisplayName', legend_text);
    
    % Fit normal distribution to the data
    params_normal = fitdist(data(:), 'Normal');
    
    % Perform chi-square goodness-of-fit test for normal distribution
    [~, p_normal] = chi2gof(data(:), 'CDF', params_normal);
    
    % Fit exponential distribution if all data points are non-negative
    if all(data(:) >= 0)
        params_exponential = fitdist(data(:), 'Exponential');
        [~, p_exponential] = chi2gof(data(:), 'CDF', params_exponential);
    else
        p_exponential = 0;
    end
    
    % Fit gamma distribution if all data points are non-negative
    if all(data(:) >= 0)
        params_gamma = fitdist(data(:), 'Gamma');
        [~, p_gamma] = chi2gof(data(:), 'CDF', params_gamma);
    else
        p_gamma = 0;
    end
    
    % Compare p-values to determine which distribution fits the data best
    if p_normal >= p_exponential && p_normal >= p_gamma
        fprintf('%s: Normal distribution fits the data best.\n', legend_text);
        mean_val = params_normal.mu;
        var_val = params_normal.sigma^2;
    elseif p_exponential >= p_normal && p_exponential >= p_gamma
        fprintf('%s: Exponential distribution fits the data best.\n', legend_text);
        mean_val = params_exponential.mu;
        var_val = params_exponential.mu^2;
    elseif p_gamma >= p_normal && p_gamma >= p_exponential
        fprintf('%s: Gamma distribution fits the data best.\n', legend_text);
        mean_val = params_gamma.a * params_gamma.b;
        var_val = params_gamma.a * params_gamma.b^2;
    else
        fprintf('%s: No suitable distribution found.\n', legend_text);
        mean_val = NaN;
        var_val = NaN;
    end
end

% Function to generate Alpha matrix based on the specified case
function Alpha = generate_alpha_matrix(alpha_case, alpha_mean, realization_seed, nNeurons)
    switch alpha_case
        case 1 % Symmetric matrix with all values the same
            Alpha = alpha_mean * ones(nNeurons);
            Alpha = Alpha - diag(diag(Alpha)); % Set diagonal to zero
            
        case 2 % Normal distribution
            alpha_std = 0.2 * alpha_mean;
            randn('state', 150 * (realization_seed));
            temp = alpha_mean + alpha_std * randn(nNeurons*(nNeurons-1)/2, 1);
            if min(temp) < 0
                temp = temp - min(temp); % Avoiding negative alphas
            end
            Alpha = triu(ones(nNeurons), 1);
            Alpha(Alpha == 1) = temp;
            Alpha = Alpha + Alpha'; % Making Alpha symmetric
            
        case 3 % Normal distribution with 20% zeros
            alpha_std = 0.2 * alpha_mean;
            randn('state', 150 * realization_seed);  % Changed to use realization_seed
            temp = alpha_mean + alpha_std * randn(nNeurons*(nNeurons-1)/2, 1);
            if min(temp) < 0
                temp = temp - min(temp); % Avoiding negative alphas
            end
            % Randomly set 20% of values to zero using the same seed
            rand('state', 150 * realization_seed);  % Use same seed for reproducibility
            zero_indices = randperm(length(temp), round(0.2*length(temp)));
            temp(zero_indices) = 0;
            Alpha = triu(ones(nNeurons), 1);
            Alpha(Alpha == 1) = temp;
            Alpha = Alpha + Alpha'; % Making Alpha symmetric
            
        otherwise
            error('Invalid alpha case specified');
    end
end


function plotControlAnalysis(u_control_mat, u_control_nofeedback_mat, ...
   u_deter_Stoch_mat, params, D_noise_vals)
   
   num_D_vals = length(D_noise_vals);
   num_runs = size(u_control_mat, 2);
   
   % Setup figure
   figure('Position', [100, 100, 800, 1200]);
   label_counter = 'a';
   
   for i = 1:num_D_vals
       % Calculate time vectors and means
       [mean_u, mean_u_nofeedback, mean_integral_feedback, ...
           mean_integral_nofeedback, time_vector] = calculateMeans(i, ...
           u_control_mat, u_control_nofeedback_mat, params, num_runs);
       
       % Plot control signals
       subplot(num_D_vals, 2, 2*i-1)
       plotControlSignals(time_vector, mean_u, mean_u_nofeedback, ...
           u_deter_Stoch_mat(:,i), D_noise_vals(i), label_counter);
       label_counter = char(label_counter + 1);
       
       % Plot integral
       subplot(num_D_vals, 2, 2*i)
       plotIntegral(time_vector, mean_integral_feedback, ...
           mean_integral_nofeedback, u_deter_Stoch_mat(:,i), ...
           params.Dt, D_noise_vals(i), label_counter);
       label_counter = char(label_counter + 1);
   end
   
   % Save figure
   saveFigure('u_and_integral_analysis.pdf');
end