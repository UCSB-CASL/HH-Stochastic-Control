function visualizeHJB(makeMovie)
% VISUALIZEHJB Creates visualizations of HJB solutions for optimal control
%   - Static figures of uStar and phi at initial and final times
%   - Optional movie showing evolution over time
%
% Usage:
%   visualizeHJB()        - Creates static figures only
%   visualizeHJB(true)    - Creates both static figures and movies
%
% Author: Faranak Rajabi
% Date: October 28, 2024

clc; close all; 
% If makeMovie parameter is not provided, default to false
if nargin < 1
   makeMovie = false;
end

%%% Parameters and Setup
% Directory setup
chosen_D = 0;
outputDir = "__Output";
DResultsDir = "D_" + num2str(chosen_D);
basePath = fullfile(outputDir, DResultsDir);

% Load initial data to get dimensions
final_time_data = readmatrix(fullfile(basePath, "uStar_0.dat"));
initial_time_data = readmatrix(fullfile(basePath, "uStar_736.dat"));
[nX, nY] = size(final_time_data);

% Grid parameters
K = 100;
x1_ = linspace(-100, 100, nX);
y1_ = linspace(0, 1, nY);
[X, Y] = meshgrid((1/K) * x1_, y1_);

% Target points and periodic orbit
vPL = -59.6; nPL = 0.403;  % Phase-less point
vSpike = 44.8; nSpike = 0.459;  % Spike point
x1Targ = (1/K) * vPL; y1Targ = nPL;

% Generate periodic orbit
tSpike = 11.85;
[~, v_po, n_po, ~] = hhapprox(tSpike);

% Visual settings
fig_width = 1600;
fig_height = 600;

%%% Generate Visualizations
if makeMovie
   % Setup video writers
   writerObj1 = VideoWriter('uStar_evolution.avi');
   writerObj2 = VideoWriter('phi_evolution.avi');
   writerObj1.FrameRate = 5;
   writerObj2.FrameRate = 5;
   open(writerObj1);
   open(writerObj2);
   
   % Create movie frames
   fprintf('Generating movie frames...\n');
   for idx = 736:-1:0
       if mod(idx, 50) == 0
           fprintf('Processing frame %d/736\n', 736-idx);
       end
       createFrames(idx, basePath, X, Y, K, x1_, y1_, v_po, n_po, x1Targ, y1Targ, writerObj1, writerObj2);
   end
   
   % Close video writers
   close(writerObj1);
   close(writerObj2);
   fprintf('Movies saved as uStar_evolution.avi and phi_evolution.avi\n');
end

% Create static figures
createStaticFigures(basePath, X, Y, K, x1_, y1_, v_po, n_po, x1Targ, y1Targ, fig_width, fig_height);
fprintf('Static figures saved as uStar_comparison.pdf and phi_comparison.pdf\n');
end

function createStaticFigures(basePath, X, Y, K, x1_, y1_, v_po, n_po, x1Targ, y1Targ, fig_width, fig_height)
%%% Create Figure 1: uStar Visualization
h1 = figure('Name', 'uStar Visualization', 'Position', [100, 100, fig_width, fig_height]);

% First subplot: uStar at t=7ms
ax1 = subplot(1, 2, 1);
final_time_data = readmatrix(fullfile(basePath, "uStar_0.dat"));
surf(X, Y, final_time_data', 'EdgeColor', 'none');
hold on;
plot3(v_po/K, n_po, zeros(size(v_po)) + min(final_time_data(:)), 'k-', 'LineWidth', 1.5);
plot3(x1Targ, y1Targ, min(final_time_data(:)), 'r*', 'LineWidth', 1.5, 'MarkerSize', 12);
title('$u^*(t=7\,ms)$', 'Interpreter', 'latex', 'FontSize', 12);
format_subplot(ax1, h1, X, Y, final_time_data, K, x1_, y1_);
legend('$u^*$', 'Periodic Orbit', 'Phase-less Set', 'Interpreter', 'latex', ...
   'Location', 'eastoutside', 'FontSize', 18, 'Box', 'off');

% Second subplot: uStar at t=0ms
ax2 = subplot(1, 2, 2);
initial_time_data = readmatrix(fullfile(basePath, "uStar_736.dat"));
surf(X, Y, initial_time_data', 'EdgeColor', 'none');
hold on;
plot3(v_po/K, n_po, zeros(size(v_po)) + min(initial_time_data(:)), 'k-', 'LineWidth', 1.5);
plot3(x1Targ, y1Targ, min(initial_time_data(:)), 'r*', 'LineWidth', 1.5, 'MarkerSize', 12);
title('$u^*(t=0\,ms)$', 'Interpreter', 'latex', 'FontSize', 12);
format_subplot(ax2, h1, X, Y, initial_time_data, K, x1_, y1_);
legend('$u^*$', 'Periodic Orbit', 'Phase-less Set', 'Interpreter', 'latex', ...
   'Location', 'eastoutside', 'FontSize', 18, 'Box', 'off');

% sgtitle('Optimal Control Function at Initial and Final Times', 'FontSize', 14, 'Interpreter', 'latex');
saveFigureAsPDF(h1, 'uStar_comparison.pdf', fig_width, fig_height);

%%% Create Figure 2: Phi Visualization
h2 = figure('Name', 'Phi', 'Position', [100, 100, fig_width, fig_height]);

% First subplot: phi at t=7ms
ax3 = subplot(1, 2, 1);
phi_final = readmatrix(fullfile(basePath, "phi_0.dat"));
surf(X, Y, phi_final', 'EdgeColor', 'none');
hold on;
plot3(v_po/K, n_po, zeros(size(v_po)) + min(phi_final(:)), 'k-', 'LineWidth', 1.5);
plot3(x1Targ, y1Targ, min(phi_final(:)), 'r*', 'LineWidth', 1.5, 'MarkerSize', 12);
title('$\mathcal{V}(t=7\,ms)$', 'Interpreter', 'latex', 'FontSize', 12);
format_subplot(ax3, h2, X, Y, phi_final, K, x1_, y1_);
legend('$\mathcal{V}$', 'Periodic Orbit', 'Phase-less Set', 'Interpreter', 'latex', ...
   'Location', 'eastoutside', 'FontSize', 18, 'Box', 'off');
% sgtitle('Value Function at Initial and Final Times', 'FontSize', 14, 'Interpreter', 'latex');

% Second subplot: phi at t=0ms
ax4 = subplot(1, 2, 2);
phi_initial = readmatrix(fullfile(basePath, "phi_736.dat"));
surf(X, Y, phi_initial', 'EdgeColor', 'none');
hold on;
plot3(v_po/K, n_po, zeros(size(v_po)) + min(phi_initial(:)), 'k-', 'LineWidth', 1.5);
plot3(x1Targ, y1Targ, min(phi_initial(:)), 'r*', 'LineWidth', 1.5, 'MarkerSize', 12);
title('$\mathcal{V}(t=0\,ms)$', 'Interpreter', 'latex', 'FontSize', 12);
format_subplot(ax4, h2, X, Y, phi_initial, K, x1_, y1_);
legend('$\mathcal{V}$', 'Periodic Orbit', 'Phase-less Set', 'Interpreter', 'latex', ...
   'Location', 'eastoutside', 'FontSize', 18, 'Box', 'off');
% sgtitle('Value Function at Initial and Final Times', 'FontSize', 14, 'Interpreter', 'latex');
saveFigureAsPDF(h2, 'phi_comparison.pdf', fig_width, fig_height);
end

function createFrames(idx, basePath, X, Y, K, x1_, y1_, v_po, n_po, x1Targ, y1Targ, writerObj1, writerObj2)
% Create frame for uStar
figure(1);
clf;
data = readmatrix(fullfile(basePath, sprintf("uStar_%d.dat", idx)));
surf(X, Y, data', 'EdgeColor', 'none');
hold on;
plot3(v_po/K, n_po, zeros(size(v_po)) + min(data(:)), 'k-', 'LineWidth', 1.5);
plot3(x1Targ, y1Targ, min(data(:)), 'r*', 'LineWidth', 1.5, 'MarkerSize', 8);
title(sprintf('$u^*(t=%.2fms)$', (736-idx)*0.0095), 'Interpreter', 'latex', 'FontSize', 12);
format_subplot(gca, gcf, X, Y, data, K, x1_, y1_);
legend('$u^*$', 'Periodic Orbit', 'Phase-less Set', 'Interpreter', 'latex', ...
   'Location', 'eastoutside', 'FontSize', 18, 'Box', 'off');
writeVideo(writerObj1, getframe(gcf));

% Create frame for phi
figure(2);
clf;
data = readmatrix(fullfile(basePath, sprintf("phi_%d.dat", idx)));
surf(X, Y, data', 'EdgeColor', 'none');
hold on;
plot3(v_po/K, n_po, zeros(size(v_po)) + min(data(:)), 'k-', 'LineWidth', 1.5);
plot3(x1Targ, y1Targ, min(data(:)), 'r*', 'LineWidth', 1.5, 'MarkerSize', 8);
title(sprintf('$\mathcal{V}(t=%.2fms)$', (736-idx)*0.0095), 'Interpreter', 'latex', 'FontSize', 12);
format_subplot(gca, gcf, X, Y, data, K, x1_, y1_);
legend('$\mathcal{V}(t)$', 'Periodic Orbit', 'Phase-less Set', 'Interpreter', 'latex', ...
   'Location', 'eastoutside', 'FontSize', 18, 'Box', 'off');
writeVideo(writerObj2, getframe(gcf));
end

function format_subplot(ax, fig, X, Y, data, K, x1_, y1_)
    shading interp;
    colormap(ax, 'parula');
    c = colorbar(ax);
    c.TickLabelInterpreter = 'latex';
    c.FontSize = 16;

    fontSize = 16;
    xlabel('$V (mV)$', 'Interpreter', 'latex', 'FontSize', fontSize);
    ylabel('$n$', 'Interpreter', 'latex', 'FontSize', fontSize);
    
    parentName = get(fig, 'Name'); % Get the name of the figure directly
    disp(['Parent Name: ', parentName]); % Debugging line
    
    if isempty(parentName)
        disp('Parent Name is empty. Defaulting to phi limits.');
        zlabel('$\mathcal{V}(t)$', 'Interpreter', 'latex', 'FontSize', fontSize);
        zlim([0 1000]);
    elseif contains(lower(parentName), 'ustar') % Use lowercase for case-insensitivity
        disp('Entering if statement for uStar');
        zlabel('$u^*(t)$', 'Interpreter', 'latex', 'FontSize', fontSize);
        zlim([-10 10]);
    else
        disp('Entering else statement for phi');
        zlabel('$\mathcal{V}(t)$', 'Interpreter', 'latex', 'FontSize', fontSize);
        zlim([0 1000]);
    end

    grid off;
    view(-45, 30);
    xlim([(1/K) * floor(x1_(1)), (1/K) * round(x1_(end))]);
    ylim([floor(y1_(1)), round(y1_(end))]);

    applyPlotStyle(ax, fig);
end

function applyPlotStyle(ax, fig)
set(ax, 'FontName', 'Times New Roman', 'FontWeight', 'bold', 'FontSize', 20);
set(ax, 'Box', 'on', 'TickDir', 'out', 'TickLength', [.02 .02], ...
   'XMinorTick', 'on', 'YMinorTick', 'on', 'YGrid', 'off', ...
   'XColor', [0 0 0], 'YColor', [0 0 0], 'LineWidth', 2.5);
set(fig, 'Color', 'w');
end

function saveFigureAsPDF(figHandle, filename, fig_width, fig_height)
    % Adjust paper properties to match figure dimensions
    set(figHandle, 'PaperUnits', 'points');
    set(figHandle, 'PaperPosition', [0, 0, fig_width, fig_height]);
    set(figHandle, 'PaperSize', [fig_width, fig_height]);
    % Save as PDF
    print(figHandle, filename, '-dpdf', '-r0');
end
