function u = func_hhapprox_coupled(t,x)
%%% FUNC_HHAPPROX_COUPLED Defines the coupled Hodgkin-Huxley approximation system
%
% Usage:
%   u = func_hhapprox_coupled(t, x)
%
% Inputs:
%   t - Time variable (not used in the function)
%   x - State vector [v1; n1; v2; n2] where:
%       v1 - Membrane potential of neuron 1
%       n1 - Gating variable of neuron 1
%       v2 - Membrane potential of neuron 2
%       n2 - Gating variable of neuron 2
%
% Output:
%   u - Column vector of the coupled Hodgkin-Huxley system equations [fv1; fn1; fv2; fn2]
%
% Global Variables:
%   gna   - Sodium conductance
%   gk    - Potassium conductance
%   gl    - Leak conductance
%   vna   - Sodium reversal potential
%   vk    - Potassium reversal potential
%   vl    - Leak reversal potential
%   II    - External current
%   c     - Membrane capacitance
%   alpha - Coupling strength between the two neurons
%
% Description:
%   This function defines the coupled Hodgkin-Huxley approximation system for two 
%   neurons. It takes the state vector x and returns the column vector u containing 
%   the derivatives of the state variables (fv1, fn1, fv2, fn2) according to the 
%   coupled Hodgkin-Huxley equations.
%
%   The function uses the global variables representing the model parameters (gna, 
%   gk, gl, vna, vk, vl, II, c) and the coupling strength (alpha) between the two 
%   neurons. The coupling terms (alpha*(v2-v1) and alpha*(v1-v2)) are added to the 
%   membrane potential equations of each neuron to model the interaction between them.
%
% Note:
%   The function assumes the existence of the global variables mentioned above. Make 
%   sure to define these variables before calling the function.
%
% Example Usage:
%   global gna gk gl vna vk vl II c alpha
%   % Set the global variable values
%   x = [v1; n1; v2; n2]; % State vector
%   t = 0; % Time variable (not used in this function)
%   u = func_hhapprox_coupled(t, x);
%
% Author: Faranak Rajabi
% Version: 1.0 (October 28, 2024)
global gna gk gl vna vk vl II c alpha

v1 = x(1);
n1 = x(2);
v2 = x(3);
n2 = x(4);


fv1 = (II - gna*(0.1*(v1+40.) / (1. - exp(-(v1+40.)/10.)) / (0.1*(v1+40.) / (1. - exp(-(v1+40.)/10.)) + 4.*exp(-(v1+65.)/18.)))^3 * (0.8-n1)*(v1 - vna) - gk*n1^4.*(v1 - vk) - gl*(v1 - vl))/c + alpha*(v2-v1);

fn1 = 0.01*(v1+55.) / (1.- exp(-(v1+55.)/10.))*(1.-n1) - 0.125*exp(-(v1+65.)/80.)*n1;

fv2 = (II - gna*(0.1*(v2+40.) / (1. - exp(-(v2+40.)/10.)) / (0.1*(v2+40.) / (1. - exp(-(v2+40.)/10.)) + 4.*exp(-(v2+65.)/18.)))^3 * (0.8-n2)*(v2 - vna) - gk*n2^4.*(v2 - vk) - gl*(v2 - vl))/c + alpha*(v1-v2);

fn2 = 0.01*(v2+55.) / (1.- exp(-(v2+55.)/10.))*(1.-n2) - 0.125*exp(-(v2+65.)/80.)*n2;

u = [fv1,fn1,fv2,fn2]';

end
