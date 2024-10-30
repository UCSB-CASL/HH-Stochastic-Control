function [F_ustar_deterministic, F_ustarstochastic] = F_interpolant_u(D_new_values, grid_struct, tFinal)
%%% F_INTERPOLANT_U Creates interpolants for deterministic and stochastic control solutions
%
% Usage:
%   [F_ustar_deterministic, F_ustarstochastic] = F_interpolant_u(D_new_values, grid_struct, tFinal)
%
% Inputs:
%   D_new_values - Vector of noise intensity values
%   grid_struct  - Grid structure from gridGenerator
%   tFinal      - Final time for simulation
%
% Outputs:
%   F_ustar_deterministic - Interpolant for deterministic control solution
%   F_ustarstochastic    - Cell array of interpolants for stochastic solutions
%                         (one for each noise intensity)
%
% Dependencies:
%   Requires global variables: nOfNodes, Ks, D_det, x_targ, uMax, gama, sigma
%   Requires data files in "__Output" directory structure
%
% File Structure:
%   "__Output/D_[value]/uStar_[n].dat" - Control solutions at different times
%   "__Output/timeMat2.txt"            - Time information
%
% Notes:
%   1. Uses linear interpolation for both deterministic and stochastic cases
%   2. Reads pre-computed control solutions from data files
%   3. Creates 3D interpolants (x, y, time) for control signals
%
% Author: Faranak Rajabi
% Version: 1.0 (October 24, 2024)

global nOfNodes Ks D_det x_targ uMax gama sigma

% Setup time grid for deterministic case
timeMat_2(:, 1) = 1:length(0:9.52138e-05*100:7);
timeMat_2(:, 2) = 0:9.52138e-05*100:7;
timeMat_2(:, 3) = 7:-9.52138e-05*100:0;
i = length(timeMat_2)+1;

%% Deterministic u* Interpolant
uStar_mat = zeros(grid_struct.nX, grid_struct.nX, length(timeMat_2));
uStar = cell(i - 1, 2);
uStar(:, 1) = num2cell(timeMat_2(:, 2));

% Read deterministic control data
uStarFolderName = "__Output/D_" + num2str(D_det);
for n = 1:i - 1
    uStarFileName = uStarFolderName + "/uStar_" + num2str(n - 1) + ".dat";
    uStar{i - 1 - n + 1, 2} = readmatrix(uStarFileName);
    uStar_mat(:,:,i - 1 - n + 1) = uStar{i - 1 - n + 1, 2};
end

% Create deterministic interpolant
gv = {grid_struct.x, grid_struct.y, timeMat_2(:, 2)};
F_ustar_deterministic = griddedInterpolant(gv, uStar_mat, 'linear');

%% Stochastic u* Interpolant
% Read time information
dataFolder = "__Output/";
filePath = fullfile(dataFolder, "timeMat2.txt");
timeMat = read_time_matrix(filePath, tFinal);
nOutput = length(timeMat(:, 3));

% Create stochastic interpolants for each noise intensity
for d_idx = 1:length(D_new_values)
    D_new = D_new_values(d_idx);
    uStarStochasticFolderName = "__Output/D_" + num2str(D_new);
    
    % Initialize storage
    uStarStochastic = cell(nOutput, 2);
    uStarStochastic(:, 1) = num2cell(timeMat(:, 2));
    
    % Read stochastic control data
    for n = 1:nOutput
        uStarStochasticFileName = uStarStochasticFolderName + "/uStar_" + num2str(n - 1) + ".dat";
        uStarStochastic{nOutput - n + 1, 2} = readmatrix(uStarStochasticFileName);
        uStarStochastic_mat(:,:,nOutput - n + 1) = uStarStochastic{nOutput - n + 1, 2};
    end
    
    % Create stochastic interpolant
    gv_2 = {grid_struct.x, grid_struct.y, timeMat(:, 2)};
    F_ustarstochastic{d_idx} = griddedInterpolant(gv_2, uStarStochastic_mat, 'linear');
end
end

% Helper function to read time matrix
function timeMat = read_time_matrix(filePath, tFinal)
    % Open file
    fid = fopen(filePath, 'r');
    if fid == -1
        error('Cannot open file: %s', filePath);
    end

    % Initialize variables
    i = 1;
    timeMat = [];

    % Read file line by line
    while ~feof(fid)
        line = fgetl(fid);
        splittedLine = strsplit(line);
        desiredLine = find(strcmp(splittedLine, "currentTime"));
        
        if ~isempty(desiredLine)
            currentTimeStr = desiredLine + 2;
            outputNum = str2double(splittedLine{currentTimeStr});
            
            timeMat(i, 1) = i;
            timeMat(i, 2) = outputNum;
            timeMat(i, 3) = tFinal - outputNum;
            i = i + 1;
        end
    end

    % Close file
    fclose(fid);
end