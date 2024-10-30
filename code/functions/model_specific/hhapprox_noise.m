function [v_noise, n_noise, t_noise] = hhapprox_noise(t)
%%% HHAPPROX_NOISE Simulates the Hodgkin-Huxley model with noise using the RK2 algorithm
%
% Usage:
%   [v_noise, n_noise, t_noise] = hhapprox_noise(t)
%
% Input:
%   t - End time for the simulation
%
% Outputs:
%   v_noise  - Membrane potential values with noise
%   n_noise  - Gating variable values with noise
%   t_noise  - Time vector corresponding to the noise-added solutions
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
%   D   - Noise intensity (default: 15)
%
% Description:
%   This function simulates the Hodgkin-Huxley model with added noise using the 
%   second-order Runge-Kutta (RK2) algorithm implemented in the rk_hh function. The 
%   function sets default values for the global variables representing the model 
%   parameters and initial conditions. It then calls the rk_hh function to compute 
%   the solution with noise and extracts the membrane potential (v_noise) and gating 
%   variable (n_noise) values from the solution matrix.
%
% Dependencies:
%   - rk_hh function
%
% Note:
%   The function uses default values for the model parameters and initial conditions. 
%   These values can be modified by changing the corresponding global variables before 
%   calling the function.
%
% Example Usage:
%   t = 100; % End time for the simulation
%   [v_noise, n_noise, t_noise] = hhapprox_noise(t);
%   plot(t_noise, v_noise); % Plot the membrane potential with noise
%
% Author: Faranak Rajabi
% Version: 1.0 (October 28, 2024)

global gna gk gl vna vk vl II c D

gna = 120;
gk = 36;
gl = 0.3;
vna = 50;
vk = -77;
vl = -54.4;
II = 10;
c = 1;

D = 15;

x0=[44.7173;0.4590];    % initial conditions
t0=0; tf=t;     % start and end time for solution

% compute the solution using the built-in ode solver ode23
%[time, x_out] = ode45(@func_hhapprox, [t0, tf], x0);

[t_noise,x_out] = rk_hh(x0,tf);

% get solutions for each variable from solution matrix x_out
v_noise = x_out(:,1);     
n_noise = x_out(:,2);     


  

