function [t, x_out] = rk_hh(x0, t_total)
%%% RK_HH Implements Honeycutt's stochastic RK2 algorithm for a 2D Hodgkin-Huxley model
%
% Usage:
%   [t, x_out] = rk_hh(x0, t_total)
%
% Inputs:
%   x0      - Initial condition vector [x0; y0]
%   t_total - Total time of integration
%
% Outputs:
%   t      - Time vector
%   x_out  - Solution matrix [x, y] where:
%            x - Membrane potential (V)
%            y - Gating variable (n)
%
% Global Variables:
%   gna - Sodium conductance
%   gk  - Potassium conductance
%   gl  - Leak conductance
%   vna - Sodium reversal potential
%   vk  - Potassium reversal potential
%   vl  - Leak reversal potential
%   II  - External current
%   c   - Membrane capacitance
%   D   - Noise intensity
%
% Dependencies:
%   - func_hhapprox function
%
% Algorithm:
%   This function implements Honeycutt's stochastic second-order Runge-Kutta (RK2) 
%   algorithm for solving the Hodgkin-Huxley model. The algorithm uses a fixed time 
%   step (dt) and calculates the solution at a specified number of points (numpts) 
%   over the total integration time (t_total).
%
%   The algorithm follows these steps:
%   1. Initialize time (t) and state variables (x, y) with the initial condition (x0).
%   2. For each time step:
%      a. Generate a random number (rx) from a standard normal distribution.
%      b. Calculate the first stage (f1) of the RK2 method using the current state.
%      c. Calculate the second stage (f2) using the updated state based on f1.
%      d. Update the state variables (x, y) using a weighted average of f1 and f2,
%         and add the stochastic term.
%   3. Store the time vector (t) and the solution matrix (x_out) containing the 
%      state variables (x, y) at each time step.
%
% Notes:
%   - The time step (dt) is fixed at 0.01.
%   - The function assumes the existence of a global variable D representing the 
%     noise intensity.
%   - The function calls the func_hhapprox function to evaluate the right-hand side 
%     of the Hodgkin-Huxley equations.
%
% Author: Faranak Rajabi
% Version: 1.0 (October 28, 2024)
  
global gna gk gl vna vk vl II c D

dt = 0.01;  % timestep (fixed)
numpts = t_total / dt;  % number of points to calculate solution at

t(1) = 0;
x(1) = x0(1);
y(1) = x0(2);

for i=2:numpts
   t(i) = i*dt;
   rx = randn;
   xold = x(i-1);
   yold = y(i-1);
   f1 = func_hhapprox(t(i),[xold,yold]);
   f1x = f1(1);
   f1y = f1(2);
   f2 = func_hhapprox(t(i),[xold+dt*f1x+sqrt(2*D*dt)*rx,yold+dt*f1y]);
   f2x = f2(1);
   f2y = f2(2);

   x(i) = xold + 0.5*dt*(f1x+f2x) + sqrt(2*D*dt)*rx;
   y(i) = yold + 0.5*dt*(f1y+f2y);

end  

x_out = [x',y'];
end
  
