function [state_stochastic, uu_stochastic_D_n] = sde_hh_model_solver(x_initial, params, D_new)
%%% SDE_HH_MODEL_SOLVER Solves stochastic HH model using Honeycutt's RK2 method
%
% Usage:
%   [state_stochastic, uu_stochastic_D_n] = sde_hh_model_solver(x_initial, params, D_new)
%
% Inputs:
%   x_initial - Initial state vector [v0; n0]
%   params    - Structure containing:
%       .tFinal   - Final simulation time
%       .Dt       - Time step
%       .tInitial - Initial time
%       .index    - Index for stochastic control
%   D_new     - Noise intensity
%
% Outputs:
%   state_stochastic    - Matrix of state trajectories [v, n]
%   uu_stochastic_D_n  - Vector of control values
%
% Notes:
%   Uses Honeycutt's RK2 method for SDE integration with additive noise
%
% Dependencies:
%   - zdyn.m
%   - Requires global variable Ks
%
% Author: Faranak Rajabi
% Version: 1.0 (October 24, 2024)

    global Ks
    
    % Extract parameters
    tFinal = params.Tend; 
    dt = params.Dt;
    
    % Initialize state vectors
    xStochastic = zeros(length(0:dt:tFinal), 1);
    yStochastic = xStochastic;
    t = zeros(length(0:dt:tFinal), 1);
    
    % Set initial conditions
    xStochastic(1) = x_initial(1);
    yStochastic(1) = x_initial(2);
    t(1) = params.tInitial;
    
    % Initialize control vector
    uu_stochastic_D_n = zeros(length(0:dt:tFinal), 1); 
    
    % Get initial control
    [~,uu_stochastic_D_n(1,1)] = zdyn(t(1),[xStochastic(1); yStochastic(1)], params);
    
    % Main integration loop
    for i = 2:size(xStochastic, 1)
        t(i) = (i - 1) * dt;
        rx = randn; % Random normal variable
        xold = xStochastic(i - 1);
        yold = yStochastic(i - 1);

        % First RK2 step
        [z1,~] = zdyn(t(i-1),[xold; yold], params);
        f1x = z1(1); f1y = z1(2);
        
        % Midpoint with noise
        x_mid = xold + dt * f1x + (sqrt(2 * D_new * dt) * rx) / Ks;
        y_mid = yold + dt * f1y;
        
        % Second RK2 step
        z2 = zdyn(t(i),[x_mid; y_mid], params);
        f2x = z2(1); f2y = z2(2);

        % Update states
        xStochastic(i) = xold + 0.5 * dt * (f1x + f2x) + (sqrt(2 * D_new * dt) * rx) / Ks;
        yStochastic(i) = yold + 0.5 * dt * (f1y + f2y);
        
        % Get control
        [~,uu_stochastic_D_n(i,1)] = zdyn(t(i),[xStochastic(i); yStochastic(i)], params);
    end
    
    % Combine states
    state_stochastic = [xStochastic,yStochastic];
end