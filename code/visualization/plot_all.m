% Define the folder path and file names
folder_name = 'pics/';

% Figure 1 - Stochastic on Stochastic
f1 = figure('Name', 'Stochastic on Stochastic', 'Position', [100, 100, 600, 800]);
% First image
subplot(2,1,1);
img1 = imread(fullfile(folder_name, 'stoch_on_stoch_one_ss.jpg'));
imshow(img1);

% Second image
subplot(2,1,2);
img2 = imread(fullfile(folder_name, 'stoch_on_stoch_one_u.jpg'));
imshow(img2);

% Adjust Figure 1 layout
set(f1, 'Color', 'w');

% Figure 2 - Stochastic-Deterministic
f2 = figure('Name', 'Stochastic-Deterministic', 'Position', [700, 100, 600, 800]);
% First image
subplot(2,1,1);
img3 = imread(fullfile(folder_name, 'stoch_det_ss.jpg'));
imshow(img3);

% Second image
subplot(2,1,2);
img4 = imread(fullfile(folder_name, 'deterministic_stochastic_u_vs_time.jpg'));
imshow(img4);

% Adjust Figure 2 layout
set(f2, 'Color', 'w');

% Adjust subplot spacing for both figures
figures = [f1, f2];
for fig = figures
    figure(fig);
    
    % Get figure dimensions
    set(fig, 'Units', 'normalized');
    
    % Adjust first subplot
    subplot(2,1,1);
    pos1 = get(gca, 'Position');
    set(gca, 'Position', [pos1(1) 0.52 pos1(3) 0.45]);  % Reduced from 0.5+spacing to 0.45
    
    % Adjust second subplot
    subplot(2,1,2);
    pos2 = get(gca, 'Position');
    set(gca, 'Position', [pos2(1) 0.02 pos2(3) 0.45]);  % Reduced spacing and height
end