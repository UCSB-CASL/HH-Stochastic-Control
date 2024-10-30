% function visualizeHJB_final()
    % VISUALIZEHJB Creates static 3D visualizations of HJB solutions for optimal control
    % Generates two figures with six subplots each for `uStar` and `phi` values
    %
    % Author: Faranak Rajabi
    % Date: October 28, 2024
    clc; close all;

    %%% Parameters and Setup
    outputDir = "__Output";
    D_values = [0, 0.5, 1, 5, 10, 15]; % Different D values
    finalFilePattern = ["phi_368.dat", "uStar_368.dat"]; % Default final files
    finalFilePattern_D0 = ["phi_736.dat", "uStar_736.dat"]; % Exception for D=0

    % Load data dimensions dynamically from D_0 directory
    DDir_0 = fullfile(outputDir, "D_0");
    sampleData = readmatrix(fullfile(DDir_0, finalFilePattern_D0(2))); % Use uStar_736.dat for dimensions
    [nX, nY] = size(sampleData);

    % Grid parameters
    K = 100;
    x1_ = linspace(-100, 100, nX);
    y1_ = linspace(0, 1, nY);
    [X, Y] = meshgrid((1/K) * x1_, y1_);

    % Figure setup
    fig_width = 1600;
    fig_height = 800;

    %%% Generate uStar and phi Figures with 3D Visualization
    % phi Visualization
    h2 = figure('Name', 'phi 3D Comparison', 'Position', [100, 100, fig_width, fig_height]);
    for i = 1:length(D_values)
        D = D_values(i);
        DDir = fullfile(outputDir, "D_" + num2str(D_values(i)));

        % Load the appropriate final data files
        if D == 0
            phiData = readmatrix(fullfile(DDir, finalFilePattern_D0(1)));
        else
            phiData = readmatrix(fullfile(DDir, finalFilePattern(1)));
        end

        % Plot the phi data for the current D in 3D
        ax_i = subplot(2, 3, i);
        surf(X, Y, phiData, 'EdgeColor', 'none', 'FaceColor', 'interp');
        view(45, 30); % Set 3D viewing angle
        grid off;
        title(sprintf("$D = %.1f$", D), 'Interpreter', 'latex', 'FontSize', 12);
        format_subplot(ax_i, h2, X, Y, phiData, K, x1_, y1_);
    end
    % saveFigureAsPDF(h2, 'phi_comparison_all_Ds.pdf', fig_width, fig_height);

    % uStar Visualization
    h1 = figure('Name', 'uStar 3D Comparison', 'Position', [100, 100, fig_width, fig_height]);
    for i = 1:length(D_values)
        D = D_values(i);
        DDir = fullfile(outputDir, "D_" + num2str(D_values(i)));

        % Load the appropriate final data files
        if D == 0
            uStarData = readmatrix(fullfile(DDir, finalFilePattern_D0(2)));
        else
            uStarData = readmatrix(fullfile(DDir, finalFilePattern(2)));
        end

        % Plot the uStar data for the current D in 3D
        ax_i = subplot(2, 3, i);
        surf(X, Y, uStarData, 'EdgeColor', 'none', 'FaceColor', 'interp');
        view(45, 30); % Set 3D viewing angle
        title(sprintf("$D = %.1f$", D), 'Interpreter', 'latex', 'FontSize', 12);
        format_subplot(ax_i, h1, X, Y, uStarData, K, x1_, y1_);
    end
    % saveFigureAsPDF(h1, 'uStar_comparison_all_Ds.pdf', fig_width, fig_height);

    % fprintf('3D static figures saved as uStar_3D_comparison.pdf and phi_3D_comparison.pdf\n');
% end

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
    % disp(['Parent Name: ', parentName]); % Debugging line
    
    if isempty(parentName)
        % disp('Parent Name is empty. Defaulting to phi limits.');
        zlabel('$\mathcal{V}(t)$', 'Interpreter', 'latex', 'FontSize', fontSize);
        zlim([0 1000]);
    elseif contains(lower(parentName), 'ustar') % Use lowercase for case-insensitivity
        % disp('Entering if statement for uStar');
        zlabel('$u^*(t)$', 'Interpreter', 'latex', 'FontSize', fontSize);
        zlim([-10 10]);
    else
        % disp('Entering else statement for phi');
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