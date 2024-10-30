function u = func_hhapprox(t,x)
%%% FUNC_HHAPPROX Defines the Hodgkin-Huxley approximation system
%
% Usage:
%   u = func_hhapprox(t, x)
%
% Inputs:
%   t - Time variable (not used in the function)
%   x - State vector [v; n] where:
%       v - Membrane potential
%       n - Gating variable
%
% Output:
%   u - Column vector of the Hodgkin-Huxley system equations [fv; fn]
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
%
% Description:
%   This function defines the Hodgkin-Huxley approximation system. It takes the state 
%   vector x and returns the column vector u containing the derivatives of the state 
%   variables (fv, fn) according to the Hodgkin-Huxley equations.
%
%   The function uses the global variables representing the model parameters (gna, 
%   gk, gl, vna, vk, vl, II, c). The equations for the membrane potential (fv) and 
%   the gating variable (fn) are defined based on the Hodgkin-Huxley formulation.
%
% Note:
%   The function assumes the existence of the global variables mentioned above. Make 
%   sure to define these variables before calling the function.
%
% Example Usage:
%   global gna gk gl vna vk vl II c
%   % Set the global variable values
%   x = [v; n]; % State vector
%   t = 0; % Time variable (not used in this function)
%   u = func_hhapprox(t, x);
%
% Author: Faranak Rajabi
% Version: 1.0 (October 28, 2024)

global gna gk gl vna vk vl II c

v = x(1);
n = x(2);

fv = (II - gna*(0.1*(v+40.) / (1. - exp(-(v+40.)/10.)) / (0.1*(v+40.) / (1. - exp(-(v+40.)/10.)) + 4.*exp(-(v+65.)/18.)))^3 * (0.8-n)*(v - vna) - gk*n^4.*(v - vk) - gl*(v - vl))/c;

fn = 0.01*(v+55.) / (1.- exp(-(v+55.)/10.))*(1.-n) - 0.125*exp(-(v+65.)/80.)*n;
  
u = [fv,fn]';

end
