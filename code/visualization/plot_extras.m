%% Plotting Extras
% This script generates plots related to the Monte Carlo simulations for 
% single-neuron activity. Ensure the data has been precomputed in the 
% Monte Carlo simulation for the single-neuron level within 
% `main_HH2D_stochastic.m` before running this script to visualize results.

%% Boxplots
% Distribution of integral of u^2
figure('Units', 'inches', 'Position', [0, 0, 10, 9], 'PaperPositionMode', 'auto', 'Renderer', 'painters');
subplot(2, 2, 1);
boxplot(J_u_ST_mat, D_new_values, 'Widths', 0.6);
xlabel('Noise Level', 'Interpreter', 'latex', 'FontSize', 10);
ylabel('$\int u^2$', 'Interpreter', 'latex', 'FontSize', 10);
title('Stochastic Control Solution', 'Interpreter', 'latex', 'FontSize', 12, 'FontWeight', 'bold');
subplot(2, 2, 2); 
boxplot(J_u_DT_mat, D_new_values, 'Widths', 0.6);
xlabel('Noise Level', 'Interpreter', 'latex', 'FontSize', 10);
ylabel('$\int u^2$', 'Interpreter', 'latex', 'FontSize', 10);
title('Deterministic Control Solution', 'Interpreter', 'latex', 'FontSize', 12, 'FontWeight', 'bold');
subplot(2, 2, 3);
boxplot(J_u_no_feedback_ST_mat, D_new_values, 'Widths', 0.6);
xlabel('Noise Level', 'Interpreter', 'latex', 'FontSize', 10);
ylabel('$\int u^2$', 'Interpreter', 'latex', 'FontSize', 10);
title('No Feedback Case (Stochastic)', 'Interpreter', 'latex', 'FontSize', 12, 'FontWeight', 'bold');
subplot(2, 2, 4);
boxplot(J_u_no_feedback_DT_mat, D_new_values, 'Widths', 0.6);
xlabel('Noise Level', 'Interpreter', 'latex', 'FontSize', 10);
ylabel('$\int u^2$', 'Interpreter', 'latex', 'FontSize', 10);
title('No Feedback Case (Deterministic)', 'Interpreter', 'latex', 'FontSize', 12, 'FontWeight', 'bold');
saveas(gcf, 'integral_of_u_squared', 'png');

% Distribution of euclidean norm of final point error
figure('Units', 'inches', 'Position', [0, 0, 10, 9]);
subplot(2, 2, 1);
boxplot(J_end_euclidean_ST_mat, D_new_values, 'Widths', 0.6);
title('Stochastic Control Solution', 'FontWeight', 'bold');
xlabel('Noise Level');
ylabel('Euclidean Norm');
subplot(2, 2, 2);
boxplot(J_end_euclidean_DT_mat, D_new_values, 'Widths', 0.6);
title('Deterministic Control Solution', 'FontWeight', 'bold');
xlabel('Noise Level');
ylabel('Euclidean Norm');
subplot(2, 2, 3);
boxplot(J_end_euclidean_no_feedback_ST_mat, D_new_values, 'Widths', 0.6);
title('No Feedback Case (Stochastic)', 'FontWeight', 'bold');
xlabel('Noise Level');
ylabel('Euclidean Norm');
subplot(2, 2, 4);
boxplot(J_end_euclidean_no_feedback_DT_mat, D_new_values, 'Widths', 0.6);
title('No Feedback Case (Deterministic)', 'FontWeight', 'bold');
xlabel('Noise Level');
ylabel('Euclidean Norm');
saveas(gcf, 'euclidean_norm_final_point_error', 'png');

% Distribution of exponential norm of final point error
figure('Units', 'inches', 'Position', [0, 0, 10, 9]);
subplot(2, 2, 1);
boxplot(J_end_exp_ST_mat, D_new_values, 'Widths', 0.6);
title('Stochastic Control Solution', 'FontWeight', 'bold');
xlabel('Noise Level');
ylabel('Exponential Norm');
subplot(2, 2, 2);
boxplot(J_end_exp_DT_mat, D_new_values, 'Widths', 0.6);
title('Deterministic Control Solution', 'FontWeight', 'bold');
xlabel('Noise Level');
ylabel('Exponential Norm');
subplot(2, 2, 3);
boxplot(J_end_exp_no_feedback_ST_mat, D_new_values, 'Widths', 0.6);
title('No Feedback Case (Stochastic)', 'FontWeight', 'bold');
xlabel('Noise Level');
ylabel('Exponential Norm');
subplot(2, 2, 4);
boxplot(J_end_exp_no_feedback_DT_mat, D_new_values, 'Widths', 0.6);
title('No Feedback Case (Deterministic)', 'FontWeight', 'bold');
xlabel('Noise Level');
ylabel('Exponential Norm');
saveas(gcf, 'exponential_norm_final_point_error', 'png');

% Distribution of total cost (euclidean + u^2 integral)
figure('Units', 'inches', 'Position', [0, 0, 10, 9]);
subplot(2, 2, 1);
boxplot(J_total_euclid_ST_mat, D_new_values, 'Widths', 0.6);
title('Stochastic Control Solution', 'Interpreter', 'latex', 'FontWeight', 'bold');
xlabel('Noise Level');
ylabel('Total Cost');
subplot(2, 2, 2);
boxplot(J_total_euclid_DT_mat, D_new_values, 'Widths', 0.6);
title('Deterministic Control Solution', 'Interpreter', 'latex', 'FontWeight', 'bold');
xlabel('Noise Level');
ylabel('Total Cost');
subplot(2, 2, 3);
boxplot(J_total_euclid_no_feedback_ST_mat, D_new_values, 'Widths', 0.6);
title('No Feedback Case (Stochastic)', 'Interpreter', 'latex', 'FontWeight', 'bold');
xlabel('Noise Level');
ylabel('Total Cost');
subplot(2, 2, 4);
boxplot(J_total_euclid_no_feedback_DT_mat, D_new_values, 'Widths', 0.6);
title('No Feedback Case (Deterministic)', 'Interpreter', 'latex', 'FontWeight', 'bold');
xlabel('Noise Level');
ylabel('Total Cost');
saveas(gcf, 'total_cost_euclidean_integral_u_squared', 'png');

% Distribution of total cost (exponential + u^2 integral = paper formula)
figure('Units', 'inches', 'Position', [0, 0, 10, 9], 'PaperPositionMode', 'auto', 'Renderer', 'painters');
subplot(2, 2, 1);
boxplot(J_total_ST_mat, D_new_values, 'Widths', 0.6);
title('Stochastic Control Solution', 'Interpreter', 'latex', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Noise Level', 'Interpreter', 'latex', 'FontSize', 10);
ylabel('Total Cost', 'Interpreter', 'latex', 'FontSize', 10);
ylim([0 1200]);

subplot(2, 2, 2);
boxplot(J_total_DT_mat, D_new_values, 'Widths', 0.6);
title('Deterministic Control Solution', 'Interpreter', 'latex', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Noise Level', 'Interpreter', 'latex', 'FontSize', 10);
ylabel('Total Cost', 'Interpreter', 'latex', 'FontSize', 10);
ylim([0 1200]);

subplot(2, 2, 3);
boxplot(J_total_no_feedback_ST_mat, D_new_values, 'Widths', 0.6);
title('No Feedback Case (Stochastic)', 'Interpreter', 'latex', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Noise Level', 'Interpreter', 'latex', 'FontSize', 10);
ylabel('Total Cost', 'Interpreter', 'latex', 'FontSize', 10);
ylim([0 1200]);

subplot(2, 2, 4);
boxplot(J_total_no_feedback_DT_mat, D_new_values, 'Widths', 0.6);
title('No Feedback Case (Deterministic)', 'Interpreter', 'latex', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Noise Level', 'Interpreter', 'latex', 'FontSize', 10);
ylabel('Total Cost', 'Interpreter', 'latex', 'FontSize', 10);
ylim([0 1200]);

saveas(gcf, 'total_cost_exponential_integral_u_squared', 'png');

% Distribution of v_end
figure('Units', 'inches', 'Position', [0, 0, 10, 9]);
subplot(2, 2, 1);
boxplot(J_v_end_ST_mat, D_new_values, 'Widths', 0.6);
title('Stochastic Control Solution', 'Interpreter', 'latex', 'FontWeight', 'bold');
xlabel('Noise Level');
ylabel('$v_{\mathrm{end}}$', 'Interpreter', 'latex');
subplot(2, 2, 2);
boxplot(J_v_end_DT_mat, D_new_values, 'Widths', 0.6);
title('Deterministic Control Solution', 'Interpreter', 'latex', 'FontWeight', 'bold');
xlabel('Noise Level');
ylabel('$v_{\mathrm{end}}$', 'Interpreter', 'latex');
subplot(2, 2, 3);
boxplot(J_v_end_no_feedback_ST_mat, D_new_values, 'Widths', 0.6);
title('No Feedback Case (Stochastic)', 'Interpreter', 'latex', 'FontWeight', 'bold');
xlabel('Noise Level');
ylabel('$v_{\mathrm{end}}$', 'Interpreter', 'latex');
subplot(2, 2, 4);
boxplot(J_v_end_no_feedback_DT_mat, D_new_values, 'Widths', 0.6);
title('No Feedback Case (deterministic)', 'Interpreter', 'latex', 'FontWeight', 'bold');
xlabel('Noise Level');
ylabel('$v_{\mathrm{end}}$', 'Interpreter', 'latex');
saveas(gcf, 'finalv', 'png');

% Distribution of n_end
figure('Units', 'inches', 'Position', [0, 0, 10, 9]);
subplot(2, 2, 1);
boxplot(J_n_end_ST_mat, D_new_values, 'Widths', 0.6);
title('Stochastic Control Solution', 'Interpreter', 'latex', 'FontWeight', 'bold');
xlabel('Noise Level');
ylabel('$n_{\mathrm{end}}$', 'Interpreter', 'latex');
subplot(2, 2, 2);
boxplot(J_n_end_ST_mat, D_new_values, 'Widths', 0.6);
title('Deterministic Control Solution', 'Interpreter', 'latex', 'FontWeight', 'bold');
xlabel('Noise Level');
ylabel('$n_{\mathrm{end}}$', 'Interpreter', 'latex');
subplot(2, 2, 3);
boxplot(J_n_end_no_feedback_ST_mat, D_new_values, 'Widths', 0.6);
title('No Feedback Case (Stochastic)', 'Interpreter', 'latex', 'FontWeight', 'bold');
xlabel('Noise Level');
ylabel('$n_{\mathrm{end}}$', 'Interpreter', 'latex');
subplot(2, 2, 4);
boxplot(J_n_end_no_feedback_DT_mat, D_new_values, 'Widths', 0.6);
title('No Feedback Case (deterministic)', 'Interpreter', 'latex', 'FontWeight', 'bold');
xlabel('Noise Level');
ylabel('$n_{\mathrm{end}}$', 'Interpreter', 'latex');
saveas(gcf, 'finalv', 'png');

%% Histogram Plots with Trend Lines
% Define line styles, markers, and colors based on the number of D_new_values
num_cases = length(D_new_values);
line_styles = cellstr(repmat('-', num_cases, 1));
markers = {'o', 's', 'd', '^', 'v', '>', '<', 'p', 'h'};
colors = {'r', 'b', 'g', 'm', 'c', 'k', 'y', [0.5, 0.5, 0.5], [0.75, 0, 0.75]};
legends = cell(1, num_cases);
for i = 1:num_cases
    legends{i} = sprintf('D = %.1f', D_new_values(i));
end

% Initialize matrices to store mean and variance values
mean_values = zeros(length(D_new_values), 6);
variance_values = zeros(length(D_new_values), 6);

% Set figure properties
fig_width = 10;
fig_height = 9;
font_size = 12;
line_width = 1.5;

% Distribution of integral of u^2
figure('Units', 'inches', 'Position', [0, 0, fig_width, fig_height], 'PaperPositionMode', 'auto', 'Renderer', 'painters');
subplot(2, 2, 1);
hold on;
for i = 1:length(D_new_values)
    [mean_val, var_val] = plotHistogramWithTrendLine(J_u_ST_mat(:, i), line_styles{i}, markers{mod(i-1, numel(markers))+1}, colors{mod(i-1, numel(colors))+1}, legends{i});
    mean_values(i, 1) = mean_val;
    variance_values(i, 1) = var_val;
end
hold off;
xlabel('$\int u^2$', 'Interpreter', 'latex', 'FontSize', font_size);
ylabel('Probability Density', 'Interpreter', 'latex', 'FontSize', font_size);
title('Stochastic Control Solution', 'Interpreter', 'latex', 'FontSize', font_size+2, 'FontWeight', 'bold');
legend('Location', 'northeast', 'FontSize', font_size-2, 'Box', 'off', 'Interpreter', 'latex');
set(gca, 'FontSize', font_size-2, 'LineWidth', line_width, 'TickLabelInterpreter', 'latex');

subplot(2, 2, 2);
hold on;
for i = 1:length(D_new_values)
    [mean_val, var_val] = plotHistogramWithTrendLine(J_u_DT_mat(:, i), line_styles{i}, markers{mod(i-1, numel(markers))+1}, colors{mod(i-1, numel(colors))+1}, legends{i});
    mean_values(i, 2) = mean_val;
    variance_values(i, 2) = var_val;
end
hold off;
xlabel('$\int u^2$', 'Interpreter', 'latex', 'FontSize', font_size);
ylabel('Probability Density', 'Interpreter', 'latex', 'FontSize', font_size);
title('Deterministic Control Solution', 'Interpreter', 'latex', 'FontSize', font_size+2, 'FontWeight', 'bold');
legend('Location', 'northeast', 'FontSize', font_size-2, 'Box', 'off', 'Interpreter', 'latex');
set(gca, 'FontSize', font_size-2, 'LineWidth', line_width, 'TickLabelInterpreter', 'latex');

subplot(2, 2, 3);
hold on;
for i = 1:length(D_new_values)
    [mean_val, var_val] = plotHistogramWithTrendLine(J_u_no_feedback_ST_mat(:, i), line_styles{i}, markers{mod(i-1, numel(markers))+1}, colors{mod(i-1, numel(colors))+1}, legends{i});
    mean_values(i, 3) = mean_val;
    variance_values(i, 3) = var_val;
end
hold off;
xlabel('$\int u^2$', 'Interpreter', 'latex', 'FontSize', font_size);
ylabel('Probability Density', 'Interpreter', 'latex', 'FontSize', font_size);
title('No Feedback Case (Stochastic)', 'Interpreter', 'latex', 'FontSize', font_size+2, 'FontWeight', 'bold');
legend('Location', 'northeast', 'FontSize', font_size-2, 'Box', 'off', 'Interpreter', 'latex');
set(gca, 'FontSize', font_size-2, 'LineWidth', line_width, 'TickLabelInterpreter', 'latex');

subplot(2, 2, 4);
hold on;
for i = 1:length(D_new_values)
    [mean_val, var_val] = plotHistogramWithTrendLine(J_u_no_feedback_DT_mat(:, i), line_styles{i}, markers{mod(i-1, numel(markers))+1}, colors{mod(i-1, numel(colors))+1}, legends{i});
    mean_values(i, 4) = mean_val;
    variance_values(i, 4) = var_val;
end
hold off;
xlabel('$\int u^2$', 'Interpreter', 'latex', 'FontSize', font_size);
ylabel('Probability Density', 'Interpreter', 'latex', 'FontSize', font_size);
title('No Feedback Case (Deterministic)', 'Interpreter', 'latex', 'FontSize', font_size+2, 'FontWeight', 'bold');
legend('Location', 'northeast', 'FontSize', font_size-2, 'Box', 'off', 'Interpreter', 'latex');
set(gca, 'FontSize', font_size-2, 'LineWidth', line_width, 'TickLabelInterpreter', 'latex');

saveas(gcf, 'integral_of_u_squared_distribution.png');

% Distribution of euclidean norm of final point error
figure('Units', 'inches', 'Position', [0, 0, fig_width, fig_height], 'PaperPositionMode', 'auto', 'Renderer', 'painters');
subplot(2, 2, 1);
hold on;
for i = 1:length(D_new_values)
    [mean_val, var_val] = plotHistogramWithTrendLine(J_end_euclidean_ST_mat(:, i), line_styles{i}, markers{mod(i-1, numel(markers))+1}, colors{mod(i-1, numel(colors))+1}, legends{i});
    mean_values(i, 5) = mean_val;
    variance_values(i, 5) = var_val;
end
hold off;
xlabel('Euclidean Norm', 'Interpreter', 'latex', 'FontSize', font_size);
ylabel('Probability Density', 'Interpreter', 'latex', 'FontSize', font_size);
title('Stochastic Control Solution', 'Interpreter', 'latex', 'FontSize', font_size+2, 'FontWeight', 'bold');
legend('Location', 'northeast', 'FontSize', font_size-2, 'Box', 'off', 'Interpreter', 'latex');
set(gca, 'FontSize', font_size-2, 'LineWidth', line_width, 'TickLabelInterpreter', 'latex');

subplot(2, 2, 2);
hold on;
for i = 1:length(D_new_values)
    [mean_val, var_val] = plotHistogramWithTrendLine(J_end_euclidean_DT_mat(:, i), line_styles{i}, markers{mod(i-1, numel(markers))+1}, colors{mod(i-1, numel(colors))+1}, legends{i});
    mean_values(i, 6) = mean_val;
    variance_values(i, 6) = var_val;
end
hold off;
xlabel('Euclidean Norm', 'Interpreter', 'latex', 'FontSize', font_size);
ylabel('Probability Density', 'Interpreter', 'latex', 'FontSize', font_size);
title('Deterministic Control Solution', 'Interpreter', 'latex', 'FontSize', font_size+2, 'FontWeight', 'bold');
legend('Location', 'northeast', 'FontSize', font_size-2, 'Box', 'off', 'Interpreter', 'latex');
set(gca, 'FontSize', font_size-2, 'LineWidth', line_width, 'TickLabelInterpreter', 'latex');

subplot(2, 2, 3);
hold on;
for i = 1:length(D_new_values)
    [mean_val, var_val] = plotHistogramWithTrendLine(J_end_euclidean_no_feedback_ST_mat(:, i), line_styles{i}, markers{mod(i-1, numel(markers))+1}, colors{mod(i-1, numel(colors))+1}, legends{i});
    mean_values(i, 7) = mean_val;
    variance_values(i, 7) = var_val;
end
hold off;
xlabel('Euclidean Norm', 'Interpreter', 'latex', 'FontSize', font_size);
ylabel('Probability Density', 'Interpreter', 'latex', 'FontSize', font_size);
title('No Feedback Case (Stochastic)', 'Interpreter', 'latex', 'FontSize', font_size+2, 'FontWeight', 'bold');
legend('Location', 'northeast', 'FontSize', font_size-2, 'Box', 'off', 'Interpreter', 'latex');
set(gca, 'FontSize', font_size-2, 'LineWidth', line_width, 'TickLabelInterpreter', 'latex');

subplot(2, 2, 4);
hold on;
for i = 1:length(D_new_values)
    [mean_val, var_val] = plotHistogramWithTrendLine(J_end_euclidean_no_feedback_DT_mat(:, i), line_styles{i}, markers{mod(i-1, numel(markers))+1}, colors{mod(i-1, numel(colors))+1}, legends{i});
    mean_values(i, 8) = mean_val;
    variance_values(i, 8) = var_val;
end
hold off;
xlabel('Euclidean Norm', 'Interpreter', 'latex', 'FontSize', font_size);
ylabel('Probability Density', 'Interpreter', 'latex', 'FontSize', font_size);
title('No Feedback Case (Deterministic)', 'Interpreter', 'latex', 'FontSize', font_size+2, 'FontWeight', 'bold');
legend('Location', 'northeast', 'FontSize', font_size-2, 'Box', 'off', 'Interpreter', 'latex');
set(gca, 'FontSize', font_size-2, 'LineWidth', line_width, 'TickLabelInterpreter', 'latex');

saveas(gcf, 'euclidean_norm_final_point_error_distribution.png');

% Distribution of exponential norm of final point error
figure('Units', 'inches', 'Position', [0, 0, fig_width, fig_height], 'PaperPositionMode', 'auto', 'Renderer', 'painters');
subplot(2, 2, 1);
hold on;
for i = 1:length(D_new_values)
    [mean_val, var_val] = plotHistogramWithTrendLine(J_end_exp_ST_mat(:, i), line_styles{i}, markers{mod(i-1, numel(markers))+1}, colors{mod(i-1, numel(colors))+1}, legends{i});
    mean_values(i, 9) = mean_val;
    variance_values(i, 9) = var_val;
end
hold off;
xlabel('Exponential Norm', 'Interpreter', 'latex', 'FontSize', font_size);
ylabel('Probability Density', 'Interpreter', 'latex', 'FontSize', font_size);
title('Stochastic Control Solution', 'Interpreter', 'latex', 'FontSize', font_size+2, 'FontWeight', 'bold');
legend('Location', 'northeast', 'FontSize', font_size-2, 'Box', 'off', 'Interpreter', 'latex');
set(gca, 'FontSize', font_size-2, 'LineWidth', line_width, 'TickLabelInterpreter', 'latex');

subplot(2, 2, 2);
hold on;
for i = 1:length(D_new_values)
    [mean_val, var_val] = plotHistogramWithTrendLine(J_end_exp_DT_mat(:, i), line_styles{i}, markers{mod(i-1, numel(markers))+1}, colors{mod(i-1, numel(colors))+1}, legends{i});
    mean_values(i, 10) = mean_val;
    variance_values(i, 10) = var_val;
end
hold off;
xlabel('Exponential Norm', 'Interpreter', 'latex', 'FontSize', font_size);
ylabel('Probability Density', 'Interpreter', 'latex', 'FontSize', font_size);
title('Deterministic Control Solution', 'Interpreter', 'latex', 'FontSize', font_size+2, 'FontWeight', 'bold');
legend('Location', 'northeast', 'FontSize', font_size-2, 'Box', 'off', 'Interpreter', 'latex');
set(gca, 'FontSize', font_size-2, 'LineWidth', line_width, 'TickLabelInterpreter', 'latex');

subplot(2, 2, 3);
hold on;
for i = 1:length(D_new_values)
    [mean_val, var_val] = plotHistogramWithTrendLine(J_end_exp_no_feedback_ST_mat(:, i), line_styles{i}, markers{mod(i-1, numel(markers))+1}, colors{mod(i-1, numel(colors))+1}, legends{i});
    mean_values(i, 11) = mean_val;
    variance_values(i, 11) = var_val;
end
hold off;
xlabel('Exponential Norm', 'Interpreter', 'latex', 'FontSize', font_size);
ylabel('Probability Density', 'Interpreter', 'latex', 'FontSize', font_size);
title('No Feedback Case (Stochastic)', 'Interpreter', 'latex', 'FontSize', font_size+2, 'FontWeight', 'bold');
legend('Location', 'northeast', 'FontSize', font_size-2, 'Box', 'off', 'Interpreter', 'latex');
set(gca, 'FontSize', font_size-2, 'LineWidth', line_width, 'TickLabelInterpreter', 'latex');

subplot(2, 2, 4);
hold on;
for i = 1:length(D_new_values)
    [mean_val, var_val] = plotHistogramWithTrendLine(J_end_exp_no_feedback_DT_mat(:, i), line_styles{i}, markers{mod(i-1, numel(markers))+1}, colors{mod(i-1, numel(colors))+1}, legends{i});
    mean_values(i, 11) = mean_val;
    variance_values(i, 11) = var_val;
end
hold off;
xlabel('Exponential Norm', 'Interpreter', 'latex', 'FontSize', font_size);
ylabel('Probability Density', 'Interpreter', 'latex', 'FontSize', font_size);
title('No Feedback Case (Deterministic)', 'Interpreter', 'latex', 'FontSize', font_size+2, 'FontWeight', 'bold');
legend('Location', 'northeast', 'FontSize', font_size-2, 'Box', 'off', 'Interpreter', 'latex');
set(gca, 'FontSize', font_size-2, 'LineWidth', line_width, 'TickLabelInterpreter', 'latex');

% Distribution of total cost (euclidean + u^2 integral)
figure('Units', 'inches', 'Position', [0, 0, fig_width, fig_height], 'PaperPositionMode', 'auto', 'Renderer', 'painters');
subplot(2, 2, 1);
hold on;
for i = 1:length(D_new_values)
    [mean_val, var_val] = plotHistogramWithTrendLine(J_total_euclid_ST_mat(:, i), line_styles{i}, markers{mod(i-1, numel(markers))+1}, colors{mod(i-1, numel(colors))+1}, legends{i});
    mean_values(i, 13) = mean_val;
    variance_values(i, 13) = var_val;
end
hold off;
xlabel('Total Cost (Euclidean)', 'Interpreter', 'latex', 'FontSize', font_size);
ylabel('Probability Density', 'Interpreter', 'latex', 'FontSize', font_size);
title('Stochastic Control Solution', 'Interpreter', 'latex', 'FontSize', font_size+2, 'FontWeight', 'bold');
legend('Location', 'northeast', 'FontSize', font_size-2, 'Box', 'off', 'Interpreter', 'latex');
set(gca, 'FontSize', font_size-2, 'LineWidth', line_width, 'TickLabelInterpreter', 'latex');

subplot(2, 2, 2);
hold on;
for i = 1:length(D_new_values)
    [mean_val, var_val] = plotHistogramWithTrendLine(J_total_euclid_DT_mat(:, i), line_styles{i}, markers{mod(i-1, numel(markers))+1}, colors{mod(i-1, numel(colors))+1}, legends{i});
    mean_values(i, 14) = mean_val;
    variance_values(i, 14) = var_val;
end
hold off;
xlabel('Total Cost (Euclidean)', 'Interpreter', 'latex', 'FontSize', font_size);
ylabel('Probability Density', 'Interpreter', 'latex', 'FontSize', font_size);
title('Deterministic Control Solution', 'Interpreter', 'latex', 'FontSize', font_size+2, 'FontWeight', 'bold');
legend('Location', 'northeast', 'FontSize', font_size-2, 'Box', 'off', 'Interpreter', 'latex');
set(gca, 'FontSize', font_size-2, 'LineWidth', line_width, 'TickLabelInterpreter', 'latex');

subplot(2, 2, 3);
hold on;
for i = 1:length(D_new_values)
    [mean_val, var_val] = plotHistogramWithTrendLine(J_total_euclid_no_feedback_ST_mat(:, i), line_styles{i}, markers{mod(i-1, numel(markers))+1}, colors{mod(i-1, numel(colors))+1}, legends{i});
    mean_values(i, 15) = mean_val;
    variance_values(i, 15) = var_val;
end
hold off;
xlabel('Total Cost (Euclidean)', 'Interpreter', 'latex', 'FontSize', font_size);
ylabel('Probability Density', 'Interpreter', 'latex', 'FontSize', font_size);
title('No Feedback Case (Stochastic)', 'Interpreter', 'latex', 'FontSize', font_size+2, 'FontWeight', 'bold');
legend('Location', 'northeast', 'FontSize', font_size-2, 'Box', 'off', 'Interpreter', 'latex');
set(gca, 'FontSize', font_size-2, 'LineWidth', line_width, 'TickLabelInterpreter', 'latex');

subplot(2, 2, 4);
hold on;
for i = 1:length(D_new_values)
    [mean_val, var_val] = plotHistogramWithTrendLine(J_total_euclid_no_feedback_DT_mat(:, i), line_styles{i}, markers{mod(i-1, numel(markers))+1}, colors{mod(i-1, numel(colors))+1}, legends{i});
    mean_values(i, 16) = mean_val;
    variance_values(i, 16) = var_val;
end
hold off;
xlabel('Total Cost (Euclidean)', 'Interpreter', 'latex', 'FontSize', font_size);
ylabel('Probability Density', 'Interpreter', 'latex', 'FontSize', font_size);
title('No Feedback Case (Deterministic)', 'Interpreter', 'latex', 'FontSize', font_size+2, 'FontWeight', 'bold');
legend('Location', 'northeast', 'FontSize', font_size-2, 'Box', 'off', 'Interpreter', 'latex');
set(gca, 'FontSize', font_size-2, 'LineWidth', line_width, 'TickLabelInterpreter', 'latex');

saveas(gcf, 'total_cost_euclidean_distribution.png');

% Distribution of total cost (exponential + u^2 integral = paper formula)
figure('Units', 'inches', 'Position', [0, 0, fig_width, fig_height], 'PaperPositionMode', 'auto', 'Renderer', 'painters');
subplot(2, 2, 1);
hold on;
for i = 1:length(D_new_values)
    [mean_val, var_val] = plotHistogramWithTrendLine(J_total_ST_mat(:, i), line_styles{i}, markers{mod(i-1, numel(markers))+1}, colors{mod(i-1, numel(colors))+1}, legends{i});
    mean_values(i, 17) = mean_val;
    variance_values(i, 17) = var_val;
end
hold off;
xlabel('Total Cost (Exponential)', 'Interpreter', 'latex', 'FontSize', font_size);
ylabel('Probability Density', 'Interpreter', 'latex', 'FontSize', font_size);
title('Stochastic Control Solution', 'Interpreter', 'latex', 'FontSize', font_size+2, 'FontWeight', 'bold');
legend('Location', 'northeast', 'FontSize', font_size-2, 'Box', 'off', 'Interpreter', 'latex');
set(gca, 'FontSize', font_size-2, 'LineWidth', line_width, 'TickLabelInterpreter', 'latex');

subplot(2, 2, 2);
hold on;
for i = 1:length(D_new_values)
    [mean_val, var_val] = plotHistogramWithTrendLine(J_total_DT_mat(:, i), line_styles{i}, markers{mod(i-1, numel(markers))+1}, colors{mod(i-1, numel(colors))+1}, legends{i});
    mean_values(i, 18) = mean_val;
    variance_values(i, 18) = var_val;
end
hold off;
xlabel('Total Cost (Exponential)', 'Interpreter', 'latex', 'FontSize', font_size);
ylabel('Probability Density', 'Interpreter', 'latex', 'FontSize', font_size);
title('Deterministic Control Solution', 'Interpreter', 'latex', 'FontSize', font_size+2, 'FontWeight', 'bold');
legend('Location', 'northeast', 'FontSize', font_size-2, 'Box', 'off', 'Interpreter', 'latex');
set(gca, 'FontSize', font_size-2, 'LineWidth', line_width, 'TickLabelInterpreter', 'latex');

subplot(2, 2, 3);
hold on;
for i = 1:length(D_new_values)
    [mean_val, var_val] = plotHistogramWithTrendLine(J_total_no_feedback_ST_mat(:, i), line_styles{i}, markers{mod(i-1, numel(markers))+1}, colors{mod(i-1, numel(colors))+1}, legends{i});
    mean_values(i, 19) = mean_val;
    variance_values(i, 19) = var_val;
end
hold off;
xlabel('Total Cost (Exponential)', 'Interpreter', 'latex', 'FontSize', font_size);
ylabel('Probability Density', 'Interpreter', 'latex', 'FontSize', font_size);
title('No Feedback Case (Stochastic)', 'Interpreter', 'latex', 'FontSize', font_size+2, 'FontWeight', 'bold');
legend('Location', 'northeast', 'FontSize', font_size-2, 'Box', 'off', 'Interpreter', 'latex');
set(gca, 'FontSize', font_size-2, 'LineWidth', line_width, 'TickLabelInterpreter', 'latex');

subplot(2, 2, 4);
hold on;
for i = 1:length(D_new_values)
    [mean_val, var_val] = plotHistogramWithTrendLine(J_total_no_feedback_DT_mat(:, i), line_styles{i}, markers{mod(i-1, numel(markers))+1}, colors{mod(i-1, numel(colors))+1}, legends{i});
    mean_values(i, 20) = mean_val;
    variance_values(i, 20) = var_val;
end
hold off;
xlabel('Total Cost (Exponential)', 'Interpreter', 'latex', 'FontSize', font_size);
ylabel('Probability Density', 'Interpreter', 'latex', 'FontSize', font_size);
title('No Feedback Case (Deterministic)', 'Interpreter', 'latex', 'FontSize', font_size+2, 'FontWeight', 'bold');
legend('Location', 'northeast', 'FontSize', font_size-2, 'Box', 'off', 'Interpreter', 'latex');
set(gca, 'FontSize', font_size-2, 'LineWidth', line_width, 'TickLabelInterpreter', 'latex');

saveas(gcf, 'total_cost_exponential_distribution.png');

% Distribution of n_end
figure('Units', 'inches', 'Position', [0, 0, fig_width, fig_height], 'PaperPositionMode', 'auto', 'Renderer', 'painters');
subplot(2, 2, 1);
hold on;
for i = 1:length(D_new_values)
    [mean_val, var_val] = plotHistogramWithTrendLine(J_n_end_ST_mat(:, i), line_styles{i}, markers{mod(i-1, numel(markers))+1}, colors{mod(i-1, numel(colors))+1}, legends{i});
    mean_values(i, 21) = mean_val;
    variance_values(i, 21) = var_val;
end
hold off;
xlabel('$n_{end}$', 'Interpreter', 'latex', 'FontSize', font_size);
ylabel('Probability Density', 'Interpreter', 'latex', 'FontSize', font_size);
title('Stochastic Control Solution', 'Interpreter', 'latex', 'FontSize', font_size+2, 'FontWeight', 'bold');
legend('Location', 'northeast', 'FontSize', font_size-2, 'Box', 'off', 'Interpreter', 'latex');
set(gca, 'FontSize', font_size-2, 'LineWidth', line_width, 'TickLabelInterpreter', 'latex');

subplot(2, 2, 2);
hold on;
for i = 1:length(D_new_values)
    [mean_val, var_val] = plotHistogramWithTrendLine(J_n_end_DT_mat(:, i), line_styles{i}, markers{mod(i-1, numel(markers))+1}, colors{mod(i-1, numel(colors))+1}, legends{i});
    mean_values(i, 22) = mean_val;
    variance_values(i, 22) = var_val;
end
hold off;
xlabel('$n_{end}$', 'Interpreter', 'latex', 'FontSize', font_size);
ylabel('Probability Density', 'Interpreter', 'latex', 'FontSize', font_size);
title('Deterministic Control Solution', 'Interpreter', 'latex', 'FontSize', font_size+2, 'FontWeight', 'bold');
legend('Location', 'northeast', 'FontSize', font_size-2, 'Box', 'off', 'Interpreter', 'latex');
set(gca, 'FontSize', font_size-2, 'LineWidth', line_width, 'TickLabelInterpreter', 'latex');

subplot(2, 2, 3);
hold on;
for i = 1:length(D_new_values)
    [mean_val, var_val] = plotHistogramWithTrendLine(J_n_end_no_feedback_ST_mat(:, i), line_styles{i}, markers{mod(i-1, numel(markers))+1}, colors{mod(i-1, numel(colors))+1}, legends{i});
    mean_values(i, 23) = mean_val;
    variance_values(i, 23) = var_val;
end
hold off;
xlabel('$n_{end}$', 'Interpreter', 'latex', 'FontSize', font_size);
ylabel('Probability Density', 'Interpreter', 'latex', 'FontSize', font_size);
title('No Feedback Case (Stochastic)', 'Interpreter', 'latex', 'FontSize', font_size+2, 'FontWeight', 'bold');
legend('Location', 'northeast', 'FontSize', font_size-2, 'Box', 'off', 'Interpreter', 'latex');
set(gca, 'FontSize', font_size-2, 'LineWidth', line_width, 'TickLabelInterpreter', 'latex');

subplot(2, 2, 4);
hold on;
for i = 1:length(D_new_values)
    [mean_val, var_val] = plotHistogramWithTrendLine(J_n_end_no_feedback_DT_mat(:, i), line_styles{i}, markers{mod(i-1, numel(markers))+1}, colors{mod(i-1, numel(colors))+1}, legends{i});
    mean_values(i, 24) = mean_val;
    variance_values(i, 24) = var_val;
end
hold off;
xlabel('$n_{end}$', 'Interpreter', 'latex', 'FontSize', font_size);
ylabel('Probability Density', 'Interpreter', 'latex', 'FontSize', font_size);
title('No Feedback Case (Deterministic)', 'Interpreter', 'latex', 'FontSize', font_size+2, 'FontWeight', 'bold');
legend('Location', 'northeast', 'FontSize', font_size-2, 'Box', 'off', 'Interpreter', 'latex');
set(gca, 'FontSize', font_size-2, 'LineWidth', line_width, 'TickLabelInterpreter', 'latex');

saveas(gcf, 'n_end_distribution.png');

% Distribution of v_end
figure('Units', 'inches', 'Position', [0, 0, fig_width, fig_height], 'PaperPositionMode', 'auto', 'Renderer', 'painters');
subplot(2, 2, 1);
hold on;
for i = 1:length(D_new_values)
    [mean_val, var_val] = plotHistogramWithTrendLine(J_v_end_ST_mat(:, i), line_styles{i}, markers{mod(i-1, numel(markers))+1}, colors{mod(i-1, numel(colors))+1}, legends{i});
    mean_values(i, 21) = mean_val;
    variance_values(i, 21) = var_val;
end
hold off;
xlabel('$v_{end}$', 'Interpreter', 'latex', 'FontSize', font_size);
ylabel('Probability Density', 'Interpreter', 'latex', 'FontSize', font_size);
title('Stochastic Control Solution', 'Interpreter', 'latex', 'FontSize', font_size+2, 'FontWeight', 'bold');
legend('Location', 'northeast', 'FontSize', font_size-2, 'Box', 'off', 'Interpreter', 'latex');
set(gca, 'FontSize', font_size-2, 'LineWidth', line_width, 'TickLabelInterpreter', 'latex');

subplot(2, 2, 2);
hold on;
for i = 1:length(D_new_values)
    [mean_val, var_val] = plotHistogramWithTrendLine(J_v_end_DT_mat(:, i), line_styles{i}, markers{mod(i-1, numel(markers))+1}, colors{mod(i-1, numel(colors))+1}, legends{i});
    mean_values(i, 22) = mean_val;
    variance_values(i, 22) = var_val;
end
hold off;
xlabel('$v_{end}$', 'Interpreter', 'latex', 'FontSize', font_size);
ylabel('Probability Density', 'Interpreter', 'latex', 'FontSize', font_size);
title('Deterministic Control Solution', 'Interpreter', 'latex', 'FontSize', font_size+2, 'FontWeight', 'bold');
legend('Location', 'northeast', 'FontSize', font_size-2, 'Box', 'off', 'Interpreter', 'latex');
set(gca, 'FontSize', font_size-2, 'LineWidth', line_width, 'TickLabelInterpreter', 'latex');

subplot(2, 2, 3);
hold on;
for i = 1:length(D_new_values)
    [mean_val, var_val] = plotHistogramWithTrendLine(J_v_end_no_feedback_ST_mat(:, i), line_styles{i}, markers{mod(i-1, numel(markers))+1}, colors{mod(i-1, numel(colors))+1}, legends{i});
    mean_values(i, 23) = mean_val;
    variance_values(i, 23) = var_val;
end
hold off;
xlabel('$v_{end}$', 'Interpreter', 'latex', 'FontSize', font_size);
ylabel('Probability Density', 'Interpreter', 'latex', 'FontSize', font_size);
title('No Feedback Case (Stochastic)', 'Interpreter', 'latex', 'FontSize', font_size+2, 'FontWeight', 'bold');
legend('Location', 'northeast', 'FontSize', font_size-2, 'Box', 'off', 'Interpreter', 'latex');
set(gca, 'FontSize', font_size-2, 'LineWidth', line_width, 'TickLabelInterpreter', 'latex');

subplot(2, 2, 4);
hold on;
for i = 1:length(D_new_values)
    [mean_val, var_val] = plotHistogramWithTrendLine(J_v_end_no_feedback_DT_mat(:, i), line_styles{i}, markers{mod(i-1, numel(markers))+1}, colors{mod(i-1, numel(colors))+1}, legends{i});
    mean_values(i, 24) = mean_val;
    variance_values(i, 24) = var_val;
end
hold off;
xlabel('$v_{end}$', 'Interpreter', 'latex', 'FontSize', font_size);
ylabel('Probability Density', 'Interpreter', 'latex', 'FontSize', font_size);
title('No Feedback Case (Deterministic)', 'Interpreter', 'latex', 'FontSize', font_size+2, 'FontWeight', 'bold');
legend('Location', 'northeast', 'FontSize', font_size-2, 'Box', 'off', 'Interpreter', 'latex');
set(gca, 'FontSize', font_size-2, 'LineWidth', line_width, 'TickLabelInterpreter', 'latex');

saveas(gcf, 'v_end_distribution.png');

%%
% Create a table with the mean and variance values
table_data = [mean_values, variance_values];
row_names = legends;
col_names = {};

% Construct column names dynamically
case_titles = {'StoControl', 'DetControl', 'NoFeedStoc', 'NoFeedDet'};

for i = 1:length(case_titles)
    col_names{end+1} = sprintf('%s_IntU2_Mean', case_titles{i});
    col_names{end+1} = sprintf('%s_IntU2_Var', case_titles{i});
    col_names{end+1} = sprintf('%s_EucNorm_Mean', case_titles{i});
    col_names{end+1} = sprintf('%s_EucNorm_Var', case_titles{i});
    col_names{end+1} = sprintf('%s_ExpNorm_Mean', case_titles{i});
    col_names{end+1} = sprintf('%s_ExpNorm_Var', case_titles{i});
    col_names{end+1} = sprintf('%s_TotCostEuc_Mean', case_titles{i});
    col_names{end+1} = sprintf('%s_TotCostEuc_Var', case_titles{i});
    col_names{end+1} = sprintf('%s_TotCostExp_Mean', case_titles{i});
    col_names{end+1} = sprintf('%s_TotCostExp_Var', case_titles{i});
    col_names{end+1} = sprintf('%s_VEnd_Mean', case_titles{i});
    col_names{end+1} = sprintf('%s_VEnd_Var', case_titles{i});
end

table_data = array2table(table_data, 'RowNames', row_names, 'VariableNames', col_names);

% Display the table
disp(table_data)

% Convert the table to a cell array
cell_data = table2cell(table_data);

