%% plots.m
% This script creates visualizations for the Hodgkin-Huxley control analysis
%
% The script generates several types of plots:
% 1. State Space Analysis
%   - Deterministic and stochastic trajectories
%   - Phase space plots with periodic orbit
%   - Time evolution of membrane potential
%
% 2. Control Performance Analysis
%   - Control signals over time
%   - Control energy (integral of u^2)
%   - Probability distributions of control costs
%
% 3. Population Level Analysis
%   - Network performance across different coupling structures
%   - Robustness analysis for different noise levels
%   - Statistical distributions of performance metrics
%
% Figure Types:
% - State space trajectories
% - Time series plots
% - Boxplots
% - Histograms
% - Jitter plots
% - Combined multi-panel figures
%
% Dependencies:
% - hhapprox.m           - Approximates HH trajectories
% - hhapprox_noise.m     - Adds noise to HH trajectories
% - Montecarlo_hh.m      - Performs Monte Carlo simulations
%
% Author: Faranak Rajabi
% Date: October 24, 2024
% Version: 1.0

%% Single Neuron Analysis: State Space for HH Model with Noise Without Control
% This script visualizes trajectories of the Hodgkin-Huxley model with 
% stochastic perturbations for a single neuron:
%
% System dynamics:
%   vdot = fv(v,n) + eta        % Membrane potential dynamics
%   ndot = fn(v,n)              % Gating variable dynamics
%   eta = sqrt(2D)N(0,1)        % Gaussian white noise term
%
% where:
% - v: membrane potential
% - n: potassium activation gating variable  
% - D: noise intensity
% - eta: stochastic perturbation (Gaussian white noise)

%%% Parameters and Initial Setup
% Simulation parameters
tSpike = 11.85;              % Time to reach spike
num_noise = 5;               % Number of stochastic realizations

% Initial conditions at spike point
vSpike = 44.8;              % Initial membrane potential (mV) 
nSpike = 0.459;             % Initial gating variable
x0 = [vSpike; nSpike];      % Initial state vector

% Figure settings
figure_position = [100, 100, 1300, 1100];  % [left, bottom, width, height]
pixel_to_cm = @(pixels) pixels * 2.54 / 96;  % Conversion factor for PDF output
figure_width_cm = pixel_to_cm(figure_position(3)) * 1.25;
figure_height_cm = pixel_to_cm(figure_position(4));

% Visual settings for different realizations
linestyles = {'-', '--', '-.', ':', '-', '--', '-.'};
cmap = [0 0 1;   % Blue
       0 1 0;   % Green
       1 0 0;   % Red
       1 0 1;   % Magenta
       0 1 1;   % Cyan
       1 1 1;   % White
       0 0 0];  % Black

%%% Figure 1: State Space (V-n) Plot
% Generate deterministic trajectory
[x0_outOfPhase, v, n, time] = hhapprox(tSpike);

% Create figure
figure('Position', figure_position);
hDeterministic = plot(v, n, '-k', 'LineWidth', 3.5);
hold on;
grid off;

% Generate and plot stochastic realizations
v_noise_all = cell(1, num_noise);
n_noise_all = cell(1, num_noise);
time_noise_all = cell(1, num_noise);

for i = 1:num_noise
   % Generate stochastic trajectory
   [v_noise, n_noise, time_noise] = hhapprox_noise(tSpike);
   
   % Store trajectories for later use
   v_noise_all{i} = v_noise;
   n_noise_all{i} = n_noise;
   time_noise_all{i} = time_noise;
   
   % Plot trajectory with unique style
   hStochastic(i) = plot(v_noise, n_noise, 'LineWidth', 3.5, ...
                        'LineStyle', linestyles{i}, 'Color', cmap(i,:));
end

% Add phase-less point target
hTarget = plot(-59.6, 0.403, 'k.', 'MarkerSize', 40);

% Add labels with LaTeX formatting
xlabel('$V\ (mV)$', 'Interpreter', 'latex', 'FontSize', 18, 'FontName', 'Times');
ylabel('$n$', 'Interpreter', 'latex', 'FontSize', 18, 'FontName', 'Times');

% Add legend
legend([hDeterministic, hStochastic, hTarget], ...
   ['Deterministic', arrayfun(@(x) ['Realization ', num2str(x)], 1:num_noise, 'UniformOutput', false), 'Phase-less Set'], ...
   'Location', 'northeast', 'FontSize', 22, 'Interpreter', 'latex', 'Box', 'off');

% Apply common styling
applyPlotStyle(gca, gcf);

% Save state-space figure
saveFigureAsPDF(gcf, 'state_space_realization_D15.pdf', figure_width_cm, figure_height_cm);

%%% Figure 2: Time Evolution of Membrane Potential
% Create new figure for voltage time series
figure('Position', figure_position);

% Plot deterministic voltage trajectory
[~, v, ~, time_n] = hhapprox(round(tSpike * 1.4));
hDeterministicTime = plot(time_n, v, '-k', 'LineWidth', 3.5);
hold on;

% Generate and plot stochastic voltage trajectories
for i = 1:num_noise
   [v_noise, n_noise, time_noise] = hhapprox_noise(round(tSpike * 1.5));
   hStochasticTime(i) = plot(time_noise, v_noise, 'LineWidth', 3.5, ...
                            'LineStyle', linestyles{i}, 'Color', cmap(i,:));
end

% Add labels with LaTeX formatting
xlabel('$Time\ (ms)$', 'Interpreter', 'latex', 'FontName', 'Times', 'FontSize', 18);
ylabel('$V\ (mV)$', 'Interpreter', 'latex', 'FontName', 'Times', 'FontSize', 18);

% Add legend
legend([hDeterministicTime, hStochasticTime], ...
    ['Deterministic', arrayfun(@(x) ['Realization ', num2str(x)], 1:num_noise, 'UniformOutput', false)], ...
   'Location', 'northeast', 'FontSize', 22, 'Interpreter', 'latex', 'Box', 'off');

% Apply common styling
applyPlotStyle(gca, gcf);

% Save time evolution figure
saveFigureAsPDF(gcf, 'u_vs_t_realization_D15.pdf', figure_width_cm, figure_height_cm);

%% Single Neuron Analysis: Deterministic Control Applied to Deterministic Trajectory
% This script visualizes the controlled trajectory and control input for a 
% deterministic single neuron system (η_i(t) ≡ 0, α_ij = 0). A regeneration
% of Nabi et al result: https://doi.org/10.1007/s10827-012-0419-3
%
% Generates two subplots:
% 1. Phase space trajectory showing:
%    - Deterministic trajectory with control
%    - Periodic orbit
%    - Phase-less target point (V_pl = -59.6, n_pl = 0.403)
% 2. Control input u*(t) over time (bounded |u| ≤ 10 μA/μF)
%
% Initial condition: Spiking point (V_s = 44.2, n_s = 0.465)
%
% Reference: Fig.~\ref{fig:deterministic_trajectory_control}
% Note: Maybe deleted from final draft

%%% Figure Setup
% Create figure with specific dimensions
figure('Units', 'inches', 'Position', [0 0 6 4.5]);
movegui('center');

% Define colors
color1 = [1 0 0];          % Red for deterministic trajectory
color2 = [0 0 1];          % Blue for periodic orbit
color3 = [0.929, 0.694, 0.125];  % Yellow for target point
color4 = [0 0 0];          % Black for control signal

%%% Subplot 1: Phase Space Trajectory
subplot(2, 1, 1)
hold on

% Plot trajectories and target
hDeterministic = plot(x_deterministic(:,1)*Ks, x_deterministic(:,2), ...
                     'LineWidth', 2.5, 'Color', color1);
hPeriodic = plot(x_periodic(:,1)*Ks, x_periodic(:,2), ...
                'LineWidth', 2.5, 'Color', color2);
hTarget = scatter(x_targ(1), x_targ(2), 50, '*', ...
                 'MarkerEdgeColor', 'k', ...
                 'MarkerFaceColor', color3, ...
                 'LineWidth', 1.5);

% Labels and legend
xlabel('$V$ (mV)', 'Interpreter', 'latex');
ylabel('$n$', 'Rotation', 90, 'Interpreter', 'latex');
legend([hDeterministic, hPeriodic, hTarget], ...
   '$\mathrm{Deterministic\;Trajectory\;with\;Control}$', ...
   '$\mathrm{Periodic\;Orbit}$', ...
   '$\mathrm{Target\;Point}$', ...
   'Location', 'NorthWest', 'Interpreter', 'latex', ...
   'FontSize', 10, 'Box', 'off');

% Adjust axis properties
axis tight
set(gca, 'XAxisLocation', 'bottom', 'YAxisLocation', 'left');
applyPlotStyle(gca, gcf);

%%% Subplot 2: Control Input
subplot(2, 1, 2)

% Plot control signal
hControl = plot(t_deterministic(:,1), u_deterministic_D_0, ...
              'LineWidth', 2.5, 'Color', colors.control);

% Labels
xlabel('$\mathrm{Time\;(ms)}$', 'Interpreter', 'latex');
ylabel('$\tilde{u}^*(t)$', 'Rotation', 90, 'Interpreter', 'latex');

% Adjust axis properties
axis tight
set(gca, 'XAxisLocation', 'bottom', 'YAxisLocation', 'left');
applyPlotStyle(gca, gcf);

%%% Save Figure
% Get figure dimensions in centimeters
fig = gcf;
fig.Units = 'centimeters';
figPosition = fig.Position;

% Save with proper dimensions
saveFigureAsPDF(gcf, 'deterministic_trajectory_control.pdf', ...
               figPosition(3), figPosition(4));

%% Single Neuron Analysis: Stochastic Control Applied to Stochastic Trajectories
% This script analyzes and visualizes the behavior of a single Hodgkin-Huxley 
% neuron under different noise intensities, with a single realization of noise. It generates two complementary plots:
% 1. Control signals u*(t) over time
% 2. State space trajectories in the V-n phase plane
%
% System:
%   vdot = fv(v,n) + eta + u*(t)    (with control)
%   ndot = fn(v,n)
%   eta = sqrt(2D)N(0,1)
%
% Noise Levels: D = [0.5, 1, 5, 10, 15]
%

%%% Simulation Parameters
% Noise parameters
D_noise_vals = [0.5 1 5 10 15];    % Noise intensity values
D_new_values = D_noise_vals;        % For consistency
num_noise = length(D_noise_vals);   % Number of noise cases

% Time parameters
params = struct(...
    'Dt', Dt, ...              % Time step
    'tFinal', 7 * 2, ...       % Final time (doubled for better visualization)
    'tInitial', 0, ...         % Initial time
    'feedback', 'on');         % Enable feedback control

% Visual parameters
linestyles = {'-', '--', '-.', ':', '-', '--', '-.'};
cmap = [1 0 0;   % Red
        0 1 0;   % Green
        0 0 1;   % Blue
        1 0 1;   % Magenta
        0 1 1;   % Cyan
        0 0 0];  % Black

%%% Monte Carlo Simulation
num_runs = 1;  % Single run for each noise level
[mat_final, result_once] = Montecarlo_hh(x_initial_periodic, params, ...
                                        D_noise_vals, num_runs);

%%% Figure 1: Control Signals Over Time
figure_position = [100, 100, 1300, 1100];
figure('Position', figure_position);
pixel_to_cm = @(pixels) pixels * 2.54 / 96;
figure_width_cm = pixel_to_cm(figure_position(3)) * 1.25;
figure_height_cm = pixel_to_cm(figure_position(4));

% Plot deterministic case
hDeterministic = line(t_deterministic, u_deterministic_D_0);
set(hDeterministic, 'LineStyle', '-', 'Color', 'k', 'LineWidth', 6);
hold on;

% Plot stochastic realizations
for d_idx = 1:num_noise
    hStochastic(d_idx) = plot(0:Dt:7, result_once{d_idx, 1});
    set(hStochastic(d_idx), 'LineStyle', linestyles{d_idx}, ...
        'Color', rand(1, 3), 'LineWidth', 3.5);
end

% Format plot
ylabel('$u^*(t)$', 'Interpreter', 'latex', 'FontSize', 18, 'FontName', 'Times');
xlabel('$Time\ (ms)$', 'Interpreter', 'latex', 'FontSize', 18, 'FontName', 'Times');

% Add legend for control signals
legendEntries = cell(1, length(D_new_values)+1);
legendEntries{1} = '$D = 0$';
for d_idx = 1:length(D_new_values)
    legendEntries{d_idx+1} = sprintf('$D = %.1f$', D_new_values(d_idx));
end
legend([hDeterministic, hStochastic], legendEntries, 'Location', 'northwest', ...
       'Interpreter', 'latex', 'FontSize', 22, 'Box', 'off');

% Apply styling and save
applyPlotStyle(gca, gcf);
saveFigureAsPDF(gcf, 'stoch_on_stoch_one_u.pdf', figure_width_cm, figure_height_cm);

%%% Figure 2: State Space Trajectories
figure('Position', figure_position);
figure_width_cm = pixel_to_cm(figure_position(3)) * 1.5;

% Plot periodic orbit
[~, v, n] = hhapprox(params.tFinal);
hPO = plot(v, n, '-k', 'LineWidth', 3.5);
hold on;
grid off;

% Plot stochastic trajectories
for d_idx = 1:num_noise
    hStochastic(d_idx) = plot(Ks * result_once{d_idx, 2}(:, 1), ...
                             result_once{d_idx, 2}(:, 2));
    set(hStochastic(d_idx), 'LineStyle', linestyles{d_idx}, ...
        'Color', rand(1, 3), 'LineWidth', 3.5);
end

% Add deterministic trajectory and target point
hDeterministic = plot(Ks * x_deterministic(:, 1), x_deterministic(:, 2), ...
                     'color', [0.5 0.5 0.5], 'LineWidth', 3.5);
hTarget = plot(-59.6, 0.403, 'k.', 'MarkerSize', 40);

% Format plot
ylabel('$n$', 'Interpreter', 'latex', 'FontSize', 18, 'FontName', 'Times');
xlabel('$V\ (mV)$', 'Interpreter', 'latex', 'FontSize', 18, 'FontName', 'Times');

% Add legend for state space
legendEntries = {
    '$Periodic\ Orbit$', ...
    '$Phase-less\ Set$', ...
    '$D = 0$'
};
for d_idx = 1:length(D_new_values)
    legendEntries{d_idx+3} = sprintf('$D = %.1f$', D_new_values(d_idx));
end
legend([hPO, hTarget, hDeterministic, hStochastic], legendEntries, ...
       'Location', 'northeast', 'Interpreter', 'latex', ...
       'FontSize', 22, 'Box', 'off');

% Apply styling and save
applyPlotStyle(gca, gcf);
saveFigureAsPDF(gcf, 'stoch_on_stoch_one_ss.pdf', figure_width_cm, figure_height_cm);

%% Single Neuron Analysis: Stochastic and Deterministic Control Inputs Applied to Deterministic Trajectories 
% This script analyzes single neuron behavior under different control strategies,
% creating two complementary visualizations:
%
% Figure 1: Control Inputs (ũ*(t) vs time)
% Shows control inputs derived from stochastic value function evaluated on 
% deterministic trajectories.
%
% Figure 2: State Space Trajectories (V-n plane)
% Shows corresponding state space trajectories, periodic orbit, and phase-less set.
%
% Reference: Figs.~\ref{fig::stoch_det} and \ref{fig::stoch_det_ss} in the paper
%
% Author: Faranak Rajabi
% Date: October 24, 2024
% Version: 1.0

%%% Common Parameters
% Visual settings
figure_position = [100, 100, 1300, 1100];
linestyles = {'-', '--', '-.', ':', '-', '--', '-.'};
cmap = [1 0 0;   % Red
       0 1 0;   % Green
       0 0 1;   % Blue
       1 0 1;   % Magenta
       0 1 1;   % Cyan
       0 0 0];  % Black

pixel_to_cm = @(pixels) pixels * 2.54 / 96;

%%% Figure 1: Control Inputs
figure('Position', figure_position);
hold on;

% Plot deterministic case (D=0)
hDeterministic = line(t_deterministic, u_deterministic_D_0, ...
   'LineStyle', '-', 'Color', cmap(1,:), 'LineWidth', 3.5);

% Plot stochastic cases
for d_idx = 1:length(D_new_values)
   hStochastic(d_idx) = line(t_deter_Stoch_mat(:,d_idx), ...
                            u_deter_Stoch_mat(:,d_idx), ...
                            'LineStyle', linestyles{d_idx}, ...
                            'Color', cmap(d_idx+1,:), ...
                            'LineWidth', 3.5);
end

% Labels and legend
xlabel('$Time\ (ms)$', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('$\tilde{u}^*(t)$', 'Interpreter', 'latex', 'FontSize', 14);

legendEntries = cell(1, length(D_new_values)+1);
legendEntries{1} = '$D = 0$';
for d_idx = 1:length(D_new_values)
   legendEntries{d_idx+1} = sprintf('$D = %.1f$', D_new_values(d_idx));
end

legend([hDeterministic, hStochastic], legendEntries, ...
      'Location', 'northwest', 'Interpreter', 'latex', ...
      'FontSize', 22, 'Box', 'off');

% Apply style and save
figure_width_cm = pixel_to_cm(figure_position(3)) * 1.25;
figure_height_cm = pixel_to_cm(figure_position(4));
applyPlotStyle(gca, gcf);
saveFigureAsPDF(gcf, 'deterministic_stochastic_u_vs_time.pdf', ...
               figure_width_cm, figure_height_cm);

%%% Figure 2: State Space Trajectories
% Initialize periodic orbit
tSpike = 11.85;
[~, v_po, n_po, ~] = hhapprox(tSpike);

% Create figure
figure('Position', figure_position);
hold on;

% Plot periodic orbit and deterministic trajectory
hPO = line(v_po, n_po, 'LineStyle', '-', 'Color', 'k', 'LineWidth', 3.5);
hDeterministic = line(Ks * x_deterministic(:, 1), x_deterministic(:, 2), ...
                    'LineStyle', '--', 'Color', 'r', 'LineWidth', 3.5);

% Plot stochastic trajectories
for d_idx = 1:length(D_new_values)
   hStochastic(d_idx) = line(Ks * x_deter_Stoch_mat(:,3*d_idx-2), ...
                            x_deter_Stoch_mat(:,3*d_idx-1), ...
                            'LineStyle', linestyles{d_idx}, ...
                            'Color', rand(1, 3), 'LineWidth', 3.5);
end

% Add phase-less target point
hTarget = plot(-59.6, 0.403, 'k.', 'MarkerSize', 40);

% Labels
xlabel('$V\ (mV)$', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('$n$', 'Interpreter', 'latex', 'FontSize', 14);

% Legend entries
legendEntries = {
   '$Periodic\ Orbit$', ...
   '$Phase-less\ Set$', ...
   '$D = 0$'
};
for d_idx = 1:length(D_new_values)
   legendEntries{end+1} = sprintf('$D = %.1f$', D_new_values(d_idx));
end

% Add legend
legend([hPO, hTarget, hDeterministic, hStochastic], legendEntries, ...
      'Location', 'northeast', 'Interpreter', 'latex', ...
      'FontSize', 22, 'Box', 'off');

% Apply style and save
figure_width_cm = pixel_to_cm(figure_position(3)) * 1.35;
figure_height_cm = pixel_to_cm(figure_position(4));
applyPlotStyle(gca, gcf);
saveFigureAsPDF(gcf, 'stoch_det_ss.pdf', figure_width_cm, figure_height_cm);

%% Population Level Analysis: Energy Expenditure Distributions
% This script creates boxplots comparing the distribution of process energy 
% expenditure (∫[u*(t)]²dt) between deterministic and stochastic control strategies
% across different noise intensities and coupling strengths.
%
% Figure Description:
% Shows distribution of control energy for:
% - Stochastic control (red boxes)
% - Deterministic control (blue boxes)
% 
% Each subplot shows:
% - Median (central line)
% - 25th-75th percentiles (box edges)
% - Whiskers (most extreme non-outlier points)
% - Individual outliers ('+' symbols)
%
% Data organization:
% (a) D = 0.5,  α = 0.05
% (b) D = 1.0,  α = 0.10
% (c) D = 5.0,  α = 0.15
% (d) D = 10.0, α = 0.20
% (e) D = 15.0, α = 0.25
%

%%% Figure Setup
% Figure dimensions
fig_width = 12;  % inches
fig_height = 8;  % inches
figure('Units', 'inches', 'Position', [0, 0, fig_width, fig_height], ...
      'PaperPositionMode', 'auto', 'Renderer', 'painters', 'Color', 'w');

% Visual parameters
stoch_color = [0.9, 0.2, 0.2];  % Red for stochastic control
det_color = [0.2, 0.2, 0.9];    % Blue for deterministic control
subplot_labels = {'(a)', '(b)', '(c)', '(d)', '(e)'};

%%% Generate Subplots
for d = 1:length(D_noise_vals)
   % Create subplot with special positioning for d=4,5
   ax = createSubplot(d);
   hold on;

   % Prepare data
   boxplot_data = [u_integral_plot_stoch_sims{1, d}(:, end), ...
                   u_integral_plot_det_sims{1, d}(:, end)];
   group_labels = [ones(size(u_integral_plot_stoch_sims{1, d}(:, end))); ...
                  2*ones(size(u_integral_plot_det_sims{1, d}(:, end)))];

   % Create and customize boxplot
   bp = boxplot(boxplot_data, group_labels, ...
               'Colors', [stoch_color; det_color], ...
               'Symbol', '.', 'OutlierSize', 6);
   set(bp, 'LineWidth', 1.5);

   % Customize boxes and whiskers
   customizeBoxPlot(bp, stoch_color, det_color);

   % Format subplot
   formatSubplot(d, D_noise_vals, subplot_labels);
   hold off;
end

%%% Save Figure
saveFigureAsPDF(gcf, 'probability_distributions_for_each_D_boxplot_legend.pdf', 20, 15);

%%% Comparing results: Plot u_squared_integral and overall performance
% Define parameters
alpha_values = [0.05, 0.1, 0.15, 0.2, 0.25];
D_ODE_values = [0.5, 1.0, 5.0, 10.0, 15.0];

% Initialize arrays to store final values
final_values_det = zeros(1, 5);
final_values_stoch = zeros(1, 5);

% Create figure
figure('Position', [100, 100, 1200, 1000]);

% Define line styles and colors
linestyles = {'-', '--', '-.', ':', '-'};
cmap = [1 0 0; 0 0 1; 0 1 0; 1 0 1; 0 1 1];

% subplot_labels = {'(a)', '(b)', '(c)', '(d)', '(e)'};

subplot_labels = {'(a)', '(b)', '(c)'};
for i = 3:5
    inx = alpha_values(i);
    D_ODE = D_ODE_values(i);
    
    % Create subplot
    if i <= 4
        ax = subplot(2, 2, i-2);
    else
        % Get the position of the first subplot to use as a reference
        pos1 = get(subplot(2, 2, 1), 'Position');
        width = pos1(3);
        height = pos1(4);
        
        % Calculate the position for the centered subplot
        left = (1 - width) / 2;
        bottom = 0.1;  % Adjust this value to align with other subplots vertically
        
        % Create the centered subplot
        ax = subplot('Position', [left bottom width height]);
    end
    hold on;
    
    % Load and plot stochastic case (D_HJB = D_ODE)
    filename_stoch = sprintf('u_squared_D_HJB_%.2f_D_ODE_%.2f.mat', D_ODE, D_ODE);
    load(filename_stoch, 'u_integral_plot');
    hStochastic = plot(u_integral_plot(1,:), u_integral_plot(2,:));
    set(hStochastic, 'LineStyle', '--', 'Color', cmap(2,:), 'LineWidth', 2.5);
    final_values_stoch(i) = u_integral_plot(2,end);

    % Load and plot deterministic case (D_HJB = 0)
    filename_det = sprintf('u_squared_D_HJB_0.00_D_ODE_%.2f.mat', D_ODE);
    load(filename_det, 'u_integral_plot');
    hDeterministic = plot(u_integral_plot(1,:), u_integral_plot(2,:));
    set(hDeterministic, 'LineStyle', '-', 'Color', cmap(1,:), 'LineWidth', 2.5);
    final_values_det(i) = u_integral_plot(2,end);
    
   % Set labels and title
    hXLabel = xlabel('Time (ms)', 'FontName', 'Times New Roman', 'FontSize', 14, 'Interpreter', 'latex');
    hYLabel = ylabel('$\int u(t)^2 dt$', 'Rotation', 90, 'FontName', 'Times New Roman', 'FontSize', 14, 'Interpreter', 'latex');
    title_str = sprintf('$D = %.1f$, $\\alpha = %.2f$', D_ODE, inx);
    xlim([0 max(u_integral_plot(1,:))]);
    y_max = max(u_integral_plot(2,:));
    ylim([0 y_max]);  % Increase upper limit by 10%

    xRange = xlim; yRange = ylim; 
    xRange_half = round(xRange(2))/5; 
    yRange = (yRange(2) - yRange(1)); yRange = yRange - yRange / 6; 
    title(title_str, 'FontName', 'Times New Roman', 'Interpreter', 'latex', 'FontSize', 18, 'FontWeight', 'bold', 'Position', [xRange_half yRange 0]);

    % Add subplot label
    % text(-0.1, 1.13, subplot_labels{i}, 'Units', 'normalized', 'FontSize', 16, ...
    %      'FontWeight', 'bold', 'FontName', 'Times New Roman');
    text(-0.1, 1.13, subplot_labels{i-2}, 'Units', 'normalized', 'FontSize', 16, ...
     'FontWeight', 'bold', 'FontName', 'Times New Roman');
    
    % Add legend
    legend([hDeterministic, hStochastic], {'Deterministic', 'Stochastic'}, ...
                     'Location', 'southeast', 'Interpreter', 'latex', 'FontSize', 10, 'Box', 'off');
    
    % Adjust axes properties
    set(gca, 'FontName', 'Times New Roman', 'FontWeight', 'bold', 'FontSize', 12);
    set(gca, 'Box', 'on', 'TickDir', 'out', 'TickLength', [.02 .02], ...
    'XMinorTick', 'on', 'YMinorTick', 'on', 'YGrid', 'off', ...
    'XColor', [0 0 0], 'YColor', [0 0 0], ...  % Set axis colors to black
    'LineWidth', 2);  % Increase LineWidth for bold axis lines

    hold off;
end

% Adjust layout and save the figure
set(gcf, 'PaperUnits', 'centimeters');  % Use centimeters for better control over size
set(gcf, 'PaperSize', [30 20]);  % Set the paper size [Width Height] in centimeters
set(gcf, 'PaperPosition', [0 0 30 20]);  % Position the plot on the paper

% Save the figure
% print('-dpdf', 'integral_plots_comparison.pdf');
print('-dpdf', 'integral_plots_comparison_3plots.pdf');

% Create table of final values
figure('Position', [100, 100, 600, 200]);
column_names = {'D', '\alpha', 'Deterministic', 'Stochastic'};
table_data = [D_ODE_values', alpha_values', final_values_det', final_values_stoch'];
uitable('Data', table_data, 'ColumnName', column_names, ...
        'RowName', [], 'Units', 'Normalized', 'Position', [0, 0, 1, 1], ...
        'FontName', 'Times New Roman', 'FontSize', 14);
title('Final Values of $\int u^2 dt$', 'FontName', 'Times New Roman', 'Interpreter', 'latex', 'FontSize', 18, 'FontWeight', 'bold');

% Save table figure
print('-dpdf', 'final_values_table.pdf');

%% Population Level Analysis: Network Robustness Under Different Coupling Schemes
% This script analyzes neural network behavior under varying coupling strength 
% conditions with active event-based control.
%
% System:
%   Population of N=100 coupled neurons with event-based control
%   ηᵢ = sqrt(2D)N(0,1) noise term
%   Three coupling schemes:
%   1. Uniform coupling (α_ij = α)
%   2. Heterogeneous coupling (α_ij ~ N(α, (0.2α)²))
%   3. Sparse heterogeneous (80% from scheme 2, 20% zeros)
%
% Reference: Fig.~\ref{fig::alpha_distribution_alpha_0.20}
%

%%% Parameters
% Data paths
main_folder = 'alpha_robustness';
D_vals = [0.5, 1, 5, 10, 15]; 
alpha_vals = [0.05, 0.1, 0.15, 0.2, 0.25];

%%% Plot Settings
% Subplot layout
subplot_height = 0.2;
subplot_width = 0.8;
left_margin = 0.1;
bottom_margins = [0.10, 0.42, 0.74];

% Line colors and styles
colors = struct(...
   'main_line', 'k', ...
   'mean_voltage', [0.8 0.8 0.8], ...
   'threshold', [0.5 0.5 0.5]);

%%% Process Each Alpha Value
for inx = 1:length(alpha_vals)
   % Load data
   folder_path = fullfile(main_folder, sprintf('alpha_%.2f', alpha_values(inx)));
   
   % Load voltage data for different coupling schemes
   load(fullfile(folder_path, 'Vp_wcwn_cell_alpha_same.mat'));
   load(fullfile(folder_path, 'Vp_wcwn_cell_dist_all.mat'));
   load(fullfile(folder_path, 'Vp_wcwn_cell_dist_80.mat'));
   
   % Create figure
   figure('Position', [100, 100, 600, 500]);
   set(gcf, 'Color', 'w');
   
   % Plot uniform coupling case
   createVoltagePlot(1, Vp_wcwn_cell_alpha_same, bottom_margins(3), ...
                    subplot_width, subplot_height, left_margin);
   
   % Plot heterogeneous coupling case
   createVoltagePlot(2, Vp_wcwn_cell_dist_all, bottom_margins(2), ...
                    subplot_width, subplot_height, left_margin);
   
   % Plot sparse heterogeneous case
   createVoltagePlot(3, Vp_wcwn_cell_dist_80, bottom_margins(1), ...
                    subplot_width, subplot_height, left_margin);
   
   % Save figure
   saveFigureWithSettings(gcf, folder_path, alpha_values(inx));
   
   fprintf('Processed α = %.2f\n', alpha_values(inx));
end

%% alpha vals
% Loop through each alpha and D value
for index = 1:length(alpha_vals)
    % Construct the folder path and filename
    folder_path = fullfile(main_folder, sprintf('alpha_%.2f', alpha_vals(index)));
    file_name = sprintf('alpha_%.2f_D_%.2f_all.pdf', alpha_vals(index), D_vals(index));
    full_path = fullfile(folder_path, file_name);
    
    % Check if the file exists
    if ~exist(full_path, 'file')
        warning('File not found: %s', full_path);
        continue;
    end
    
    % Load the PDF
    try
        pdf_data = importdata(full_path);
    catch
        warning('Unable to import file: %s', full_path);
        continue;
    end
    
    % Create subplot
    if index <= 3
        ax = subplot(2, 3, index);
    else
        ax = subplot(2, 3, index + 1);
    end
    
    % Display the PDF
    copyobj(allchild(pdf_data.Children), ax);
    
    % Set the aspect ratio to match the PDF
    daspect(ax, [1 1 1]);
    
    % Remove axes for cleaner look
    axis(ax, 'off');
    
    % Add title to the subplot
    title(ax, sprintf('$\\alpha = %.2f, D = %.2f$', alpha_vals(index), D_vals(index)), ...
          'Interpreter', 'latex', 'FontSize', 10);
end

% Adjust the layout
sgtitle('Combined Alpha and D Plots', 'FontSize', 16, 'FontWeight', 'Bold');

% Adjust subplot spacing
set(fig, 'Units', 'normalized');
set(fig.Children, 'Units', 'normalized');
tightfig(fig);

% Save the combined figure
print(fig, fullfile(main_folder, 'combined_alpha_D_plots.pdf'), '-dpdf', '-bestfit');
fprintf('Combined figure saved as: %s\n', fullfile(main_folder, 'combined_alpha_D_plots.pdf'));

%% 3: Probability Distribution vs Integral(u^2)
% Create figure
figure_position = [100, 100, 1000, 1100];  % [left, bottom, width, height] in pixels
figure('Position', figure_position);

% Get the width and height in centimeters
pixel_to_cm = @(pixels) pixels * 2.54 / 96;  % Conversion factor
figure_width_cm = pixel_to_cm(figure_position(3)) * 1.25;
figure_height_cm = pixel_to_cm(figure_position(4));
% Define line styles, colors, and markers
line_styles = {'-', '--', '-.', ':', '-', '--', '-.'};
colors = [1 0 0; 0 1 0; 0 0 1; 1 0 1; 0 1 1; 0 0 0]; % Red, Green, Blue, Magenta, Cyan, Black
markers = {'.', 'o', 'x', '+', '*', 's'};

% legends
legends = cell(length(D_noise_vals), 1);
for n = 1:length(D_noise_vals)
    legends{n} = sprintf('$D = %.2f$', D_noise_vals(n));
end

% Initialize variables to store overall limits
overall_xlim = [Inf, -Inf];
overall_ylim = [Inf, -Inf];

% Subplot 1: Stochastic Control Solution
subplot(2, 1, 1);
hold on;
for i = 1:length(D_noise_vals)
    [mean_val, var_val] = plotHistogramWithTrendLine(J_u_ST_mat(:, i), line_styles{i}, markers{i}, colors(i, :), legends{i});
    mean_values(i, 1) = mean_val;
    variance_values(i, 1) = var_val;
    
    % Update overall limits
    current_xlim = xlim;
    current_ylim = ylim;
    overall_xlim = [min(overall_xlim(1), current_xlim(1)), max(overall_xlim(2), current_xlim(2))];
    overall_ylim = [min(overall_ylim(1), current_ylim(1)), max(overall_ylim(2), current_ylim(2))];
end
hold off;

% Set labels and legend
xlabel('$\int[u^*(t)]^2 dt$', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('Probability Density', 'Interpreter', 'latex', 'FontSize', 14);
l1 = legend('show', 'Location', 'NorthEast', 'Interpreter', 'latex', 'FontSize', 16, 'Box', 'off');

set(gca, 'FontName', 'Times New Roman', 'FontWeight', 'bold', 'FontSize', 18);
set(gca, 'Box', 'on', 'TickDir', 'out', 'TickLength', [.02 .02], 'FontWeight', 'bold', 'XMinorTick', 'on', 'YMinorTick', 'on', 'YGrid', 'off', ...
'XColor', [0 0 0], 'YColor', [0 0 0], ... 
'LineWidth', 4);
set(gcf, 'Color', 'w');

% Subplot label (a)
text(-0.15, 1.05, '(a)', 'Units', 'normalized', 'FontSize', 18, 'Interpreter', 'latex', 'FontWeight', 'bold');

% Subplot 2: Deterministic Control Solution
subplot(2, 1, 2);
hold on;
for i = 1:length(D_noise_vals)
    [mean_val, var_val] = plotHistogramWithTrendLine(J_u_DT_mat(:, i), line_styles{i}, markers{i}, colors(i, :), legends{i});
    mean_values(i, 2) = mean_val;
    variance_values(i, 2) = var_val;
    
    % Update overall limits
    current_xlim = xlim;
    current_ylim = ylim;
    overall_xlim = [min(overall_xlim(1), current_xlim(1)), max(overall_xlim(2), current_xlim(2))];
    overall_ylim = [min(overall_ylim(1), current_ylim(1)), max(overall_ylim(2), current_ylim(2))];
end
hold off;

% Set labels and legend
xlabel('$\int[u_0^*(t)]^2 dt$', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('Probability Density', 'Interpreter', 'latex', 'FontSize', 14);
l2 = legend('show', 'Location', 'NorthEast', 'Interpreter', 'latex', 'FontSize', 16, 'Box', 'off');

set(gca, 'FontName', 'Times New Roman', 'FontWeight', 'bold', 'FontSize', 18);
set(gca, 'Box', 'on', 'TickDir', 'out', 'TickLength', [.02 .02], 'FontWeight', 'bold', 'XMinorTick', 'on', 'YMinorTick', 'on', 'YGrid', 'off', ...
'XColor', [0 0 0], 'YColor', [0 0 0], ... 
'LineWidth', 4);
set(gcf, 'Color', 'w');

% Subplot label (b)
text(-0.15, 1.05, '(b)', 'Units', 'normalized', 'FontSize', 18, 'Interpreter', 'latex', 'FontWeight', 'bold');

% Apply the same limits to both subplots
for i = 1:2
    subplot(2, 1, i);
    xlim(overall_xlim);
    ylim(overall_ylim);
end

% Set figure background to white and save the figure as a PDF
set(gcf, 'Color', 'w');
print('-dpdf', '-bestfit', 'prob_distribution_vs_u_squared.pdf');

%% 3-2: single neuron level. Integral of u^2 vs time for monte carlo sim 
% Create figure
figure_position = [100, 100, 1000, 1100];  % [left, bottom, width, height] in pixels
figure('Position', figure_position);

% legends
legends = cell(length(D_noise_vals), 1);
for n = 1:length(D_noise_vals)
    legends{n} = sprintf('$D = %.2f$', D_noise_vals(n));
end

% Initialize variables to store overall limits
overall_xlim = [Inf, -Inf];
overall_ylim = [Inf, -Inf];

% Subplot 1: Stochastic Control Solution
subplot(2, 1, 1);
hold on;
for i = 1:length(D_noise_vals)
    [mean_val, var_val] = plotHistogramWithTrendLine(J_u_ST_mat(:, i), line_styles{i}, markers{i}, colors(i, :), legends{i});
    mean_values(i, 1) = mean_val;
    variance_values(i, 1) = var_val;
    
    % Update overall limits
    current_xlim = xlim;
    current_ylim = ylim;
    overall_xlim = [min(overall_xlim(1), current_xlim(1)), max(overall_xlim(2), current_xlim(2))];
    overall_ylim = [min(overall_ylim(1), current_ylim(1)), max(overall_ylim(2), current_ylim(2))];
end
hold off;

% Set labels and legend
xlabel('$\int[u^*(t)]^2 dt$', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('Probability Density', 'Interpreter', 'latex', 'FontSize', 14);
l1 = legend('show', 'Location', 'NorthEast', 'Interpreter', 'latex', 'FontSize', 16, 'Box', 'off');

% Get the width and height in centimeters
pixel_to_cm = @(pixels) pixels * 2.54 / 96;  % Conversion factor
figure_width_cm = pixel_to_cm(figure_position(3)) * 1.25;
figure_height_cm = pixel_to_cm(figure_position(4));

% Define line styles, colors, and markers
line_styles = {'-', '--', '-.', ':', '-', '--', '-.'};
colors = [1 0 0; 0 1 0; 0 0 1; 1 0 1; 0 1 1; 0 0 0]; % Red, Green, Blue, Magenta, Cyan, Black
markers = {'.', 'o', 'x', '+', '*', 's'};



% Set labels and legend
xlabel('$t$', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('Mean of $u^2$', 'Interpreter', 'latex', 'FontSize', 14);
legend('show', 'Location', 'NorthEast', 'Interpreter', 'latex', 'FontSize', 16, 'Box', 'off');

% Customize axes appearance
set(gca, 'FontName', 'Times New Roman', 'FontWeight', 'bold', 'FontSize', 18);
set(gca, 'Box', 'on', 'TickDir', 'out', 'TickLength', [.02 .02], 'FontWeight', 'bold', 'XMinorTick', 'on', 'YMinorTick', 'on', 'YGrid', 'off', ...
    'XColor', [0 0 0], 'YColor', [0 0 0], ... 
    'LineWidth', 4);
set(gcf, 'Color', 'w');

% Set figure background to white and save the figure as a PDF
set(gcf, 'Color', 'w');
print('-dpdf', '-bestfit', 'mean_u_squared_vs_time.pdf');

%% 4: Total cost (exponential + integral u^2) 
% Distribution of total cost (exponential + u^2 integral = paper formula)
figure('Units', 'inches', 'Position', [0, 0, fig_width, fig_height], 'PaperPositionMode', 'auto', 'Renderer', 'painters');

% Subplot 1: Stochastic Control Solution
subplot(2, 1, 1);
hold on;
for i = 1:length(D_new_values)
 [mean_val, var_val] = plotHistogramWithTrendLine(J_total_ST_mat(:, i), line_styles{i}, markers{mod(i-1, numel(markers))+1}, colors{mod(i-1, numel(colors))+1}, legends{i});
 mean_values(i, 17) = mean_val;
 variance_values(i, 17) = var_val;
end
hold off;
hXLabel = xlabel('$\mathrm{Total\;Cost\;}$', 'Interpreter', 'latex');
hYLabel = ylabel('$\mathrm{Probability\;Density}$', 'Rotation', 90, 'Interpreter', 'latex');
hLegend = legend(legends, 'Location', 'NorthEast', 'Interpreter', 'latex', 'Box', 'off');
set(gca, 'Box', 'on', 'TickDir', 'out', 'TickLength', [.02 .02], ...
    'XMinorTick', 'on', 'YMinorTick', 'on', 'YGrid', 'off', ...
    'XColor', [.3 .3 .3], 'YColor', [.3 .3 .3], ...
    'LineWidth', 1.5)
set([hXLabel, hYLabel], 'FontName', 'Times New Roman', 'FontSize', 14, 'FontWeight', 'normal', 'Interpreter', 'latex')
set(hLegend, 'FontSize', 10, 'Interpreter', 'latex', 'Box', 'off')

% Add subplot label (a)
text(-0.15, 1.05, '(a)', 'Units', 'normalized', 'FontSize', 14, 'FontWeight', 'bold', 'FontName', 'Times New Roman')

% Get the axis limits for the first subplot
xlim_vals = xlim;
ylim_vals = ylim;

% Subplot 2: No Feedback Case (Stochastic)
subplot(2, 1, 2);
hold on;
for i = 1:length(D_new_values)
 [mean_val, var_val] = plotHistogramWithTrendLine(J_total_no_feedback_ST_mat(:, i), line_styles{i}, markers{mod(i-1, numel(markers))+1}, colors{mod(i-1, numel(colors))+1}, legends{i});
 mean_values(i, 19) = mean_val;
 variance_values(i, 19) = var_val;
end
hold off;
hXLabel = xlabel('$\mathrm{Total\;Cost\;}$', 'Interpreter', 'latex');
hYLabel = ylabel('$\mathrm{Probability\;Density}$', 'Rotation', 90, 'Interpreter', 'latex');
hLegend = legend(legends, 'Location', 'NorthEast', 'Interpreter', 'latex', 'Box', 'off');
set(gca, 'Box', 'on', 'TickDir', 'out', 'TickLength', [.02 .02], 'LineWidth', 1.5, 'FontName', 'Times New Roman', 'FontSize', 12, ...
    'XMinorTick', 'on', 'YMinorTick', 'on', 'XColor', [.3 .3 .3], 'YColor', [.3 .3 .3])
set([hXLabel, hYLabel], 'FontName', 'Times New Roman', 'FontSize', 14, 'FontWeight', 'normal', 'Interpreter', 'latex')
set(hLegend, 'FontSize', 10, 'Interpreter', 'latex', 'Box', 'off')

% Add subplot label (b)
text(-0.15, 1.05, '(b)', 'Units', 'normalized', 'FontSize', 14, 'FontWeight', 'bold', 'FontName', 'Times New Roman')

% Set the same axis limits for both subplots
xlim(xlim_vals);
ylim(ylim_vals);

set(gcf, 'Color', 'w') % Set figure background to white
print('-dpdf', 'prob_distribution_vs_total_cost.pdf')

%% Combined Plot: Probability Distribution and Total Cost
figure('Units', 'inches', 'Position', [0, 0, fig_width, fig_height*2], 'PaperPositionMode', 'auto', 'Renderer', 'painters');

% Subplot 1: Stochastic Control Solution (u^2 integral)
subplot(2, 2, 1);
hold on;
for i = 1:length(D_new_values)
    [mean_val, var_val] = plotHistogramWithTrendLine(J_u_ST_mat(:, i), line_styles{i}, markers{mod(i-1, numel(markers))+1}, colors{mod(i-1, numel(colors))+1}, legends{i});
    mean_values(i, 1) = mean_val;
    variance_values(i, 1) = var_val;
end
hold off;
xlabel('$\int [u^*(t)]^2 dt$', 'Interpreter', 'latex');
ylabel('$\mathrm{Probability\;Density}$', 'Interpreter', 'latex');
legend(legends, 'Location', 'NorthWest', 'Interpreter', 'latex', 'Box', 'off');

% Adjust axes properties
set(gca, 'FontName', 'Times New Roman', 'FontWeight', 'bold', 'FontSize', 12);
set(gca, 'Box', 'on', 'TickDir', 'out', 'TickLength', [.02 .02], ...
'XMinorTick', 'on', 'YMinorTick', 'on', 'YGrid', 'off', ...
'XColor', [0 0 0], 'YColor', [0 0 0], ...  % Set axis colors to black
'LineWidth', 2);  % Increase LineWidth for bold axis lines
xlim([0 260]);

text(-0.2, 1.1, '(a)', 'Units', 'normalized', 'FontSize', 14, 'FontWeight', 'bold', 'FontName', 'Times New Roman');

% Subplot 2: Deterministic Control Solution (u^2 integral)
subplot(2, 2, 3);
hold on;
for i = 1:length(D_new_values)
    [mean_val, var_val] = plotHistogramWithTrendLine(J_u_DT_mat(:, i), line_styles{i}, markers{mod(i-1, numel(markers))+1}, colors{mod(i-1, numel(colors))+1}, legends{i});
    mean_values(i, 2) = mean_val;
    variance_values(i, 2) = var_val;
end
hold off;
xlabel('$\int [u_0^*(t)]^2 dt$', 'Interpreter', 'latex');
ylabel('$\mathrm{Probability\;Density}$', 'Interpreter', 'latex');
legend(legends, 'Location', 'NorthWest', 'Interpreter', 'latex', 'Box', 'off');

% Adjust axes properties
set(gca, 'FontName', 'Times New Roman', 'FontWeight', 'bold', 'FontSize', 12);
set(gca, 'Box', 'on', 'TickDir', 'out', 'TickLength', [.02 .02], ...
'XMinorTick', 'on', 'YMinorTick', 'on', 'YGrid', 'off', ...
'XColor', [0 0 0], 'YColor', [0 0 0], ...  % Set axis colors to black
'LineWidth', 2);  % Increase LineWidth for bold axis lines
xlim([0 260]);
text(-0.2, 1.1, '(b)', 'Units', 'normalized', 'FontSize', 14, 'FontWeight', 'bold', 'FontName', 'Times New Roman');

% Subplot 3: Stochastic Control Solution (Total Cost)
subplot(2, 2, 2);
hold on;
for i = 1:length(D_new_values)
    [mean_val, var_val] = plotHistogramWithTrendLine(J_total_ST_mat(:, i), line_styles{i}, markers{mod(i-1, numel(markers))+1}, colors{mod(i-1, numel(colors))+1}, legends{i});
    mean_values(i, 17) = mean_val;
    variance_values(i, 17) = var_val;
end
hold off;
xlabel('$\mathrm{Total\;Cost\;}$', 'Interpreter', 'latex');
ylabel('$\mathrm{Probability\;Density}$', 'Interpreter', 'latex');
legend(legends, 'Location', 'NorthEast', 'Interpreter', 'latex', 'Box', 'off');

% Adjust axes properties
set(gca, 'FontName', 'Times New Roman', 'FontWeight', 'bold', 'FontSize', 12);
set(gca, 'Box', 'on', 'TickDir', 'out', 'TickLength', [.02 .02], ...
'XMinorTick', 'on', 'YMinorTick', 'on', 'YGrid', 'off', ...
'XColor', [0 0 0], 'YColor', [0 0 0], ...  % Set axis colors to black
'LineWidth', 2);  % Increase LineWidth for bold axis lines
xlim(xlim_vals);
ylim(ylim_vals);
text(-0.2, 1.1, '(c)', 'Units', 'normalized', 'FontSize', 14, 'FontWeight', 'bold', 'FontName', 'Times New Roman');

% Subplot 4: No Feedback Case (Stochastic) (Total Cost)
subplot(2, 2, 4);
hold on;
for i = 1:length(D_new_values)
    [mean_val, var_val] = plotHistogramWithTrendLine(J_total_no_feedback_ST_mat(:, i), line_styles{i}, markers{mod(i-1, numel(markers))+1}, colors{mod(i-1, numel(colors))+1}, legends{i});
    mean_values(i, 19) = mean_val;
    variance_values(i, 19) = var_val;
end
hold off;
xlabel('$\mathrm{Total\;Cost\;}$', 'Interpreter', 'latex');
ylabel('$\mathrm{Probability\;Density}$', 'Interpreter', 'latex');
legend(legends, 'Location', 'northeast', 'Interpreter', 'latex', 'Box', 'off');

% Adjust axes properties
set(gca, 'FontName', 'Times New Roman', 'FontWeight', 'bold', 'FontSize', 12);
set(gca, 'Box', 'on', 'TickDir', 'out', 'TickLength', [.02 .02], ...
'XMinorTick', 'on', 'YMinorTick', 'on', 'YGrid', 'off', ...
'XColor', [0 0 0], 'YColor', [0 0 0], ...  % Set axis colors to black
'LineWidth', 2);  % Increase LineWidth for bold axis lines

xlim(xlim_vals);
ylim(ylim_vals);
text(-0.2, 1.1, '(d)', 'Units', 'normalized', 'FontSize', 14, 'FontWeight', 'bold', 'FontName', 'Times New Roman');

% Set figure background to white
set(gcf, 'Color', 'w');
% Adjust layout before saving the figure
set(gcf, 'PaperPositionMode', 'auto');  % Ensure the paper position mode is auto
set(gcf, 'PaperUnits', 'centimeters');  % Use centimeters for better control over size
set(gcf, 'PaperSize', [30 20]);  % Set the paper size [Width Height] in centimeters
set(gcf, 'PaperPosition', [0 0 30 20]);  % Position the plot on the paper

% Save the figure as a PDF
print('-dpdf', 'combined_prob_distribution_total_cost.pdf');


%% integral of u^2 for all the alpha cases and boxplots control counts
% Parameters for figure size
fig_width = 12;
fig_height = 8;

% Create a new figure for u^2 integral plots
figure('Units', 'inches', 'Position', [0, 0, fig_width, fig_height], ...
       'PaperPositionMode', 'auto', 'Renderer', 'painters', 'Color', 'w');

% Define colormap, linestyles, and markers
cmap = [0 0 1; 0 1 0; 1 0 0]; % Colors for alpha cases
linestyles = {'-', '--', '-.'}; % Valid line styles in MATLAB
markers = {'o', 's', '^'}; % Markers for alpha cases

% Alpha values for legend labels
alpha_values = 0.05:0.05:0.25;

for d = 1:length(D_noise_vals)
    % Create subplot
    if d <= 3
        ax = subplot(2, 3, d);
    elseif d == 4
        % Position between subplot 1 and 2
        pos1 = get(subplot(2, 3, 1), 'Position');
        pos2 = get(subplot(2, 3, 2), 'Position');
        width = pos1(3);
        height = pos1(4);
        left = pos1(1) + (pos2(1) - pos1(1)) / 2;
        bottom = 0.1; % Adjust this value to align with other subplots vertically
        ax = subplot('Position', [left bottom width height]);
    else % d == 5
        % Position between subplot 2 and 3
        pos2 = get(subplot(2, 3, 2), 'Position');
        pos3 = get(subplot(2, 3, 3), 'Position');
        width = pos2(3);
        height = pos2(4);
        left = pos2(1) + (pos3(1) - pos2(1)) / 2;
        bottom = 0.1; % Adjust this value to align with other subplots vertically
        ax = subplot('Position', [left bottom width height]);
    end
    hold on;
    
    % Loop over each alpha case and plot the average u^2 integral over time
    for alpha_case = 1:3
        % Calculate the average u^2 integral over time across simulations
        avg_u_integral = mean(cell2mat(u_integral_results(d, alpha_case, :)), 3);
        
        % Use line function to plot with explicit properties
        hLine = line('XData', avg_u_integral(1,:), 'YData', avg_u_integral(2,:), ...
            'LineStyle', linestyles{alpha_case}, 'Color', cmap(alpha_case, :), ...
            'LineWidth', 2, 'Marker', markers{alpha_case}, 'MarkerSize', 0.75, ...
            'MarkerFaceColor', cmap(alpha_case, :));
        
        % Debugging: explicitly setting properties in case of rendering issues
        set(hLine, 'LineWidth', 2);
    end
    
    % Set plot properties
    xlabel('$Time\ (ms)$', 'FontName', 'AvantGarde', 'FontSize', 14, 'FontWeight', 'bold', 'Interpreter', 'latex');
    ylabel('$\int \tilde{u}^2(t) dt$', 'Rotation', 90, 'FontName', 'AvantGarde', 'FontSize', 14, 'FontWeight', 'bold', 'Interpreter', 'latex');
    
    % Add D value as text inside the plot
    text(0.4, 0.1, sprintf('$D = %.1f$', D_noise_vals(d)), 'Units', 'normalized', ...
         'FontSize', 14, 'FontWeight', 'bold', 'FontName', 'AvantGarde', 'Interpreter', 'latex', ...
         'VerticalAlignment', 'top');
    
    % Adjust axes properties
    set(gca, 'FontName', 'Times New Roman', 'FontWeight', 'bold', 'FontSize', 16, ...
        'Box', 'on', 'TickDir', 'out', 'TickLength', [.02 .02], ...
        'XMinorTick', 'on', 'YMinorTick', 'on', 'XColor', [0 0 0], 'YColor', [0 0 0], ...
        'LineWidth', 2, 'XAxisLocation', 'bottom', 'YAxisLocation', 'left');
    
    % Set x-axis limit
    xlim([0, u_integral_plot_stoch(1, end)]);
    
    % Add subplot label (a), (b), etc.
    text(-0.2, 1.1, sprintf('\\textbf{(%c)}', 'a' + d - 1), 'Units', 'normalized', ...
         'FontSize', 14, 'FontWeight', 'bold', 'FontName', 'Times New Roman', 'Interpreter', 'latex');
    
    % Create legend for this subplot
    alpha = alpha_values(d);
   legendText = {
    'Homogeneous network',
    'Heterogeneous network',
    'Sparse heterogeneous network'
    };
    legend(legendText, 'Interpreter', 'latex', 'FontSize', 11.5, 'Location', 'northwest', 'Box', 'off');
end

% Set figure background to white
set(gcf, 'Color', 'w');

% Adjust subplot spacing
set(gcf, 'Units', 'normalized');
set(gcf, 'Position', [0.1, 0.1, 0.8, 0.8]);

% Adjust layout and save the figure
set(gcf, 'PaperUnits', 'centimeters');  % Use centimeters for better control over size
set(gcf, 'PaperSize', [42 30]);  % Set the paper size [Width Height] in centimeters
set(gcf, 'PaperPosition', [0 0 42 30]);  % Position the plot on the paper
print('-dpdf', 'robustness_all_usquared.pdf');

%% Jitter plot for control counts 
% Parameters for figure size
fig_width = 12;
fig_height = 8;

% Create a new figure for control count jitter plots
figure('Units', 'inches', 'Position', [0, 0, fig_width, fig_height], ...
       'PaperPositionMode', 'auto', 'Renderer', 'painters', 'Color', 'w');

% Define colormap for D values
cmap = lines(length(D_noise_vals)); % Use a color map with different colors for D values
alpha_cases = 1:3; % Alpha case indices
alpha_values = 0.05:0.05:0.25; % Alpha values for legends

% Create centered subplots for jitter plots
for d = 1:length(D_noise_vals)
    if d <= 3
        ax = subplot(2, 3, d);
    elseif d == 4
        % Position between subplot 1 and 2
        pos1 = get(subplot(2, 3, 1), 'Position');
        pos2 = get(subplot(2, 3, 2), 'Position');
        width = pos1(3);
        height = pos1(4);
        left = pos1(1) + (pos2(1) - pos1(1)) / 2;
        bottom = 0.1; % Adjust this value to align with other subplots vertically
        ax = subplot('Position', [left bottom width height]);
    else % d == 5
        % Position between subplot 2 and 3
        pos2 = get(subplot(2, 3, 2), 'Position');
        pos3 = get(subplot(2, 3, 3), 'Position');
        width = pos2(3);
        height = pos2(4);
        left = pos2(1) + (pos3(1) - pos2(1)) / 2;
        bottom = 0.1; % Adjust this value to align with other subplots vertically
        ax = subplot('Position', [left bottom width height]);
    end
    hold on;
    
    % Loop over each alpha case and plot jittered control counts
    for alpha_case = alpha_cases
        % Collect control count data for current D and alpha case
        control_counts = squeeze(control_count_results(d, alpha_case, :));
        
        % Jitter the x-values for better visualization (decrease jitter range to avoid overlap)
        x_jitter = alpha_case + (rand(size(control_counts)) - 0.5) * 0.15; % Smaller jitter around alpha_case
        
        % Introduce jitter in y-values as well (small jitter)
        y_jitter = control_counts + (rand(size(control_counts)) - 0.5) * 0.05 * range(control_counts);
        
        % Plot the jittered scatter points
        scatter(x_jitter, y_jitter, 50, cmap(d, :), 'filled', 'MarkerEdgeColor', [0 0 0], ...
            'DisplayName', sprintf('D=%.1f, Alpha Case %d', D_noise_vals(d), alpha_case));
    end
    
    % Set plot properties
    ylabel('Control Count', 'FontName', 'AvantGarde', 'FontSize', 14, 'FontWeight', 'bold', 'Interpreter', 'latex');

    % Set x-axis limits to exactly fit 1, 2, and 3
    xlim([0 4]);
    
    % Set x-axis ticks and labels
    xticks(3:4); % Set ticks to discrete values [1, 2, 3]
    xticklabels({
        ' ', ...
        ' ', ...
        ' '
    });
    
    % Ensure x-tick labels are horizontal and use latex interpreter
    % set(gca, 'TickLabelInterpreter', 'latex', 'XTickLabelRotation', 0);
    
    % Add D value as a title for each subplot
    title(sprintf('$D = %.1f$', D_noise_vals(d)), 'FontName', 'AvantGarde', ...
        'FontSize', 14, 'FontWeight', 'bold', 'Interpreter', 'latex');

    % Adjust axes properties
    set(gca, 'FontName', 'Times New Roman', 'FontWeight', 'bold', 'FontSize', 14.5, ...
        'Box', 'on', 'TickDir', 'out', 'TickLength', [.02 .02], ...
        'XMinorTick', 'off', 'YMinorTick', 'on', 'XColor', [0 0 0], 'YColor', [0 0 0], ...
        'LineWidth', 2, 'XAxisLocation', 'bottom', 'YAxisLocation', 'left');
    
    % Add subplot label (a), (b), etc.
    text(-0.2, 1.1, sprintf('\\textbf{(%c)}', 'a' + d - 1), 'Units', 'normalized', ...
         'FontSize', 14, 'FontWeight', 'bold', 'FontName', 'Times New Roman', 'Interpreter', 'latex');
end

% Set figure background to white
set(gcf, 'Color', 'w');

% Adjust subplot spacing
set(gcf, 'Units', 'normalized');
set(gcf, 'Position', [0.1, 0.1, 0.8, 0.8]);

% Adjust layout and save the figure
set(gcf, 'PaperUnits', 'centimeters');  % Use centimeters for better control over size
set(gcf, 'PaperSize', [45 30]);  % Set the paper size [Width Height] in centimeters
set(gcf, 'PaperPosition', [0 0 45 30]);  % Position the plot on the paper
print('-dpdf', 'robustness_control_count_jitter.pdf');

%% Helper functions
function [mean_val, var_val] = plotHistogramWithTrendLine(data, line_style, marker, color, legend_text)
   % Calculate histogram counts and bin centers
   [counts, edges] = histcounts(data(:), 'Normalization', 'pdf');
   centers = (edges(1:end-1) + edges(2:end)) / 2;
   
   % Plot line connecting the tops of the histogram bins (trend line)
   plot(centers, counts, 'LineStyle', line_style, 'Color', color, 'LineWidth', 3.5, 'DisplayName', legend_text);
   
   % Plot markers at the bin centers
   % plot(centers, counts, 'Marker', marker, 'MarkerSize', 6, 'MarkerFaceColor', color, 'MarkerEdgeColor', color, 'LineStyle', 'none');

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


% Function to apply common settings
function applyCommonSettings(ax, hXLabel, hYLabel)
    set(ax, 'FontName', 'Times New Roman', 'FontWeight', 'bold', 'FontSize', 12);
    set(ax, 'Box', 'on', 'TickDir', 'out', 'TickLength', [.02 .02], ...
        'XMinorTick', 'on', 'YMinorTick', 'on', 'YGrid', 'off', ...
        'XColor', [0 0 0], 'YColor', [0 0 0], ... % Set axis colors to black
        'LineWidth', 2); % Increase LineWidth for bold axis lines
    set([hXLabel, hYLabel], 'FontName', 'Times New Roman', 'FontSize', 14, 'FontWeight', 'bold', 'Interpreter', 'latex')
    axis tight
    set(ax, 'XAxisLocation', 'bottom', 'YAxisLocation', 'left')
end

function tightfig(fig)
    % Custom function to tighten figure layout
    tightInset = get(gca, 'TightInset');
    position(1) = tightInset(1);
    position(2) = tightInset(2);
    position(3) = 1 - tightInset(1) - tightInset(3);
    position(4) = 1 - tightInset(2) - tightInset(4);
    set(gca, 'Position', position);
    set(gcf, 'PaperPositionMode', 'auto');
end


function applyPlotStyle(ax, fig)
    % Applies common styling to plot axes and figure
    set(ax, 'FontName', 'Times New Roman', 'FontWeight', 'bold', 'FontSize', 30);
    set(ax, 'Box', 'on', 'TickDir', 'out', 'TickLength', [.02 .02], ...
        'XMinorTick', 'on', 'YMinorTick', 'on', 'YGrid', 'off', ...
        'XColor', [0 0 0], 'YColor', [0 0 0], 'LineWidth', 4);
    set(fig, 'Color', 'w');
end

function saveFigureAsPDF(fig, filename, width_cm, height_cm)
    % Saves figure as PDF with specified dimensions
    set(fig, 'PaperUnits', 'centimeters');
    set(fig, 'PaperSize', [width_cm height_cm]);
    set(fig, 'PaperPosition', [0 0 width_cm height_cm]);
    print('-dpdf', filename, '-bestfit');
end

function ax = createSubplot(d)
   % Creates subplot with special positioning for subplots 4 and 5
   if d <= 3
       ax = subplot(2, 3, d);
   else
       % Get positions for alignment
       pos1 = get(subplot(2, 3, 1), 'Position');
       pos2 = get(subplot(2, 3, 2), 'Position');
       pos3 = get(subplot(2, 3, 3), 'Position');
       
       % Calculate position
       width = pos1(3);
       height = pos1(4);
       if d == 4
           left = (pos1(1) + pos2(1)) / 2;  % Between 1st and 2nd
       else
           left = (pos2(1) + pos3(1)) / 2;  % Between 2nd and 3rd
       end
       ax = subplot('Position', [left 0.1 width height]);
   end
end

function customizeBoxPlot(bp, stoch_color, det_color)
   % Customizes boxplot colors and appearance
   h_boxes = findobj(bp, 'Tag', 'Box');
   h_whiskers = findobj(bp, 'Tag', 'Whisker');
   
   % Color boxes if found
   if ~isempty(h_boxes) && length(h_boxes) >= 2
       patch(get(h_boxes(1), 'XData'), get(h_boxes(1), 'YData'), ...
             stoch_color, 'FaceAlpha', 0.3, 'EdgeColor', stoch_color);
       patch(get(h_boxes(2), 'XData'), get(h_boxes(2), 'YData'), ...
             det_color, 'FaceAlpha', 0.3, 'EdgeColor', det_color);
   end
   
   % Color whiskers if found
   if ~isempty(h_whiskers) && length(h_whiskers) >= 4
       set(h_whiskers(1:2), 'Color', det_color);     % Deterministic
       set(h_whiskers(3:4), 'Color', stoch_color);   % Stochastic
   end
end

function formatSubplot(d, D_noise_vals, subplot_labels)
   % Formats subplot appearance
   xlabel('', 'FontName', 'Times New Roman', 'FontSize', 14);
   ylabel('$\int [u^*(t)]^2 dt$', 'FontName', 'Times New Roman', ...
          'FontSize', 14, 'Interpreter', 'latex');
          
   title(sprintf('$D = %.2f$', D_noise_vals(d)), ...
         'FontName', 'Times New Roman', 'Interpreter', 'latex', ...
         'FontSize', 18, 'FontWeight', 'bold');
         
   xticklabels({'\textbf{$\tilde{u}^*(t)$}', '$u_0^*(t)$'});
   set(gca, 'TickLabelInterpreter', 'latex', 'XTickLabelRotation', 0);
   
   % Set axes properties
   set(gca, 'FontName', 'Times New Roman', 'FontWeight', 'bold', ...
       'FontSize', 12, 'Box', 'on', 'TickDir', 'out', ...
       'TickLength', [.02 .02], 'XMinorTick', 'off', ...
       'YMinorTick', 'on', 'LineWidth', 2);
       
   % Add subplot label
   text(-0.15, 1.15, subplot_labels{d}, 'Units', 'normalized', ...
        'FontSize', 14, 'FontWeight', 'bold', ...
        'FontName', 'Times New Roman');
end

function createVoltagePlot(subplot_num, data, bottom_margin, width, height, left)
   % Creates a voltage plot subplot with consistent formatting
   ax = subplot('Position', [left, bottom_margin, width, height]);
   hold on;
   
   % Plot voltage traces
   plot(data{1}, data{2}, 'k', 'LineWidth', 1.5);
   plot(data{1}, data{3}, ':', 'color', [0.8 0.8 0.8], 'LineWidth', 1.5);
   plot([data{1}(1) data{1}(end)], [-20 -20], ':', ...
        'color', [0.5 0.5 0.5], 'LineWidth', 1.5);
   
   % Labels and formatting
   hXLabel = xlabel('Time (ms)');
   hYLabel = ylabel('$V_{i,wc}$');
   applyPlotStyle(ax, gcf);
   
   % Set axis limits
   set(ax, 'XLim', [0 round(data{1}(end))]);
   set(ax, 'YLim', [-100 80]);
   hold off;
end

function saveFigureWithSettings(fig, folder_path, alpha_value)
   % Saves figure with standardized settings
   set(fig, 'PaperUnits', 'points');
   set(fig, 'PaperPosition', [0 0 600 500]);
   set(fig, 'PaperSize', [600 500]);
   
   % Save figure
   figFileName = fullfile(folder_path, sprintf('alpha_%.2f_vs_time.pdf', alpha_value));
   print(fig, figFileName, '-dpdf', '-r300');
end
