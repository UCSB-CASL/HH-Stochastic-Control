function [x0_outOfPhase, v, n, time] = hhapprox(t)
%%% HHAPPROX Simulates the Hodgkin-Huxley model using the built-in ODE solver ode45
%
% Usage:
%   [x0_outOfPhase, v, n, time] = hhapprox(t)
%
% Input:
%   t - End time for the simulation
%
% Outputs:
%   x0_outOfPhase - Final state vector [v; n] at the end of the simulation
%   v             - Membrane potential values
%   n             - Gating variable values
%   time          - Time vector corresponding to the solutions
%
% Global Variables:
%   gna - Sodium conductance (default: 120)
%   gk  - Potassium conductance (default: 36)
%   gl  - Leak conductance (default: 0.3)
%   vna - Sodium reversal potential (default: 50)
%   vk  - Potassium reversal potential (default: -77)
%   vl  - Leak reversal potential (default: -54.4)
%   II  - External current (default: 10)
%   c   - Membrane capacitance (default: 1)
%
% Description:
%   This function simulates the Hodgkin-Huxley model using the built-in ODE solver 
%   ode45. The function sets default values for the global variables representing the 
%   model parameters and initial conditions. It then calls the ode45 function to 
%   compute the solution and extracts the membrane potential (v) and gating variable 
%   (n) values from the solution matrix.
%
%   The function also computes the final state vector (x0_outOfPhase) at the end of 
%   the simulation by calling ode45 again with the same initial conditions and end 
%   time.
%
% Dependencies:
%   - func_hhapprox function
%
% Note:
%   The function uses default values for the model parameters and initial conditions. 
%   These values can be modified by changing the corresponding global variables before 
%   calling the function.
%
% Example Usage:
%   t = 20; % End time for the simulation
%   [x0_outOfPhase, v, n, time] = hhapprox(t);
%   plot(time, v); % Plot the membrane potential
%
% Author: Faranak Rajabi
% Version: 1.0 (October 28, 2024)

global gna gk gl vna vk vl II c

gna = 120;
gk = 36;
gl = 0.3;
vna = 50;
vk = -77;
vl = -54.4;
II = 10;
c = 1;

vSpike = 44.8;  % mV 44.2
nSpike = 0.459; % 0.465
% tSpike = 12;
tSpike = 11.85;

x0 = [vSpike; nSpike];    % initial conditions
t0 = 0;
% tf = tSpike;      % start and end time for solution

% % only for comptiblity of time and v vectors for plotting 
% if strcmp(isNoisy, 'off')
%     t_ode = tf; 
% elseif strcmp(isNoisy, 'on')
%     t_ode = tSpike; 
% else
%     error('wrong noise status.')
% end


% compute the solution using the built-in ode solver ode45
t_ode = t;
[time, x_out] = ode45(@func_hhapprox, [t0, t_ode], x0);

% get solutions for each variable from solution matrix x_out
v = x_out(:, 1);
n = x_out(:, 2);

allOut = zeros(length(v), 3);
allOut(:, 1) = time; allOut(:, 2) = v; allOut(:, 3) = n;

[time1, x_out1] = ode45(@func_hhapprox, [t0, t_ode], x0);
v1 = x_out1(:, 1);
n1 = x_out1(:, 2);
x0_outOfPhase = [v1(end), n1(end)];


