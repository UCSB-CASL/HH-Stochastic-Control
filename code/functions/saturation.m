function [x_new]  = saturation(max,min,x)
%%% SATURATION Applies saturation limits to a given value
%
% Usage:
%   [x_new] = saturation(max, min, x)
%
% Inputs:
%   max - Upper saturation limit
%   min - Lower saturation limit
%   x   - Input value
%
% Output:
%   x_new - Saturated value
%
% Description:
%   This function applies saturation limits to a given input value x. If x is less than
%   the lower saturation limit (min), the function returns min. If x is greater than the
%   upper saturation limit (max), the function returns max. If x is within the saturation
%   limits, the function returns x unchanged.
%
% Example:
%   max = 10;
%   min = -5;
%   x = 15;
%   x_new = saturation(max, min, x);
%   % x_new will be 10 (upper saturation limit)
%
% Author: Faranak Rajabi
% Version: 1.0 (October 28, 2024)

if x < min 
    x_new = min;
elseif x > max 
    x_new = max;
else
    x_new = x;
end
end