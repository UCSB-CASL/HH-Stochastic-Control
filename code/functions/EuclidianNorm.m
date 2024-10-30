function norm_out = EuclidianNorm(xEndVec, yEndVec, runNum)
%%% EUCLIDIANNORM Calculates the normalized Euclidean distance to target state
%
% Usage:
%   norm_out = EuclidianNorm(xEndVec, yEndVec, runNum)
%
% Inputs:
%   xEndVec - Vector of x-coordinates (membrane potential values)
%   yEndVec - Vector of y-coordinates (gating variable values)
%   runNum  - Number of runs for normalization
%
% Outputs:
%   norm_out - Normalized Euclidean distance to target state
%
% Dependencies:
%   Requires global variable x_targ to be defined, containing target state [v_target; n_target]
%
% Notes:
%   Distance is normalized by dividing by runNum to get average distance
%
% Author: Faranak Rajabi
% Version: 1.0 (October 24, 2024)

global x_targ

% Calculate Euclidean distance to target state
norm_out = sqrt(((xEndVec) - x_targ(1)).^2 + ((yEndVec) - x_targ(2)).^2);

% Normalize by number of runs
norm_out = norm_out / runNum;
end