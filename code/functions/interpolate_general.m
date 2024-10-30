function [t_out, fOft_out] = interpolate_general(t, fOft, num_nodes, method)
%%% INTERPOLATE_GENERAL Performs interpolation on a given function
%
% Usage:
%   [t_out, fOft_out] = interpolate_general(t, fOft, num_nodes, method)
%
% Inputs:
%   t         - Input time vector
%   fOft      - Input function values corresponding to time vector t
%   num_nodes - Number of nodes for the output interpolated function
%   method    - Interpolation method ('linear' or 'quadratic')
%
% Outputs:
%   t_out     - Output time vector with num_nodes equally spaced points
%   fOft_out  - Output interpolated function values corresponding to t_out
%
% Description:
%   This function performs interpolation on a given function defined by the input 
%   time vector (t) and corresponding function values (fOft). The interpolation is 
%   performed using the specified method ('linear' or 'quadratic') to generate an 
%   output function with num_nodes equally spaced points.
%
% Input Validation:
%   - Checks if the input vectors t and fOft have the same number of elements.
%   - Converts t and fOft to column vectors if necessary.
%
% Interpolation Methods:
%   - 'linear': Performs linear interpolation using MATLAB's interp1 function.
%   - 'quadratic': Performs shape-preserving piecewise cubic interpolation using 
%                  MATLAB's interp1 function with the 'pchip' option.
%
% Error Handling:
%   - Throws an error if the input vectors have incompatible sizes.
%   - Throws an error if an unsupported interpolation method is specified.
%
% Example Usage:
%   t = 0:0.1:1;
%   fOft = sin(2*pi*t);
%   num_nodes = 50;
%   method = 'quadratic';
%   [t_out, fOft_out] = interpolate_general(t, fOft, num_nodes, method);
%
% Author: Faranak Rajabi
% Version: 1.0 (October 28, 2024)
    % Input validation
    if numel(t) ~= numel(fOft)
        error('Error in interpolate_general. Incompatible vector sizes.')
    end
    
    % Ensure column vectors
    t = t(:);
    fOft = fOft(:);
    
    % Create output time vector
    t_out = linspace(t(1), t(end), num_nodes)';
    
    % Initialize output function values
    % fOft_out = zeros(num_nodes, 1);
    
    % Perform interpolation based on specified method
    switch method
        case 'linear'
            fOft_out = interp1(t, fOft, t_out, 'linear');
        case 'quadratic'
            fOft_out = interp1(t, fOft, t_out, 'pchip');  % 'pchip' for shape-preserving piecewise cubic interpolation
        otherwise
            error('Unsupported interpolation method. Use "linear" or "quadratic".');
    end
end