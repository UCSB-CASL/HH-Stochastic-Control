function grid_struct = gridGenerator(xMin, xMax, yMin, yMax, nX, nY)
%%% GRIDGENERATOR Creates a 2D grid structure for numerical computations
%
% Usage:
%   grid_struct = gridGenerator(xMin, xMax, yMin, yMax, nX, nY)
%
% Inputs:
%   xMin - Minimum x-coordinate value
%   xMax - Maximum x-coordinate value
%   yMin - Minimum y-coordinate value
%   yMax - Maximum y-coordinate value
%   nX   - Number of points in x-direction
%   nY   - Number of points in y-direction
%
% Outputs:
%   grid_struct - Structure containing grid information:
%       .x    - Vector of x-coordinates
%       .y    - Vector of y-coordinates
%       .X    - Matrix of x-coordinates (from meshgrid)
%       .Y    - Matrix of y-coordinates (from meshgrid)
%       .min  - Vector [xMin, yMin]
%       .max  - Vector [xMax, yMax]
%       .nX   - Number of points in x-direction
%       .nY   - Number of points in y-direction
%
% Author: Faranak Rajabi
% Version: 1.0 (October 24, 2024)

    % Calculate grid spacing
    dx = (xMax - xMin) / (nX - 1);
    dy = (yMax - yMin) / (nY - 1);

    % Generate coordinate vectors
    x = xMin:dx:xMax;
    y = yMin:dy:yMax;

    % Create meshgrid for 2D computations
    [X, Y] = meshgrid(x, y);

    % Populate output structure
    grid_struct.x = x;
    grid_struct.y = y;
    grid_struct.X = X;
    grid_struct.Y = Y;
    grid_struct.min = [xMin, yMin];
    grid_struct.max = [xMax, yMax];
    grid_struct.nX = nX;
    grid_struct.nY = nY;
end