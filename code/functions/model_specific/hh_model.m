function [f] = hh_model(x)
%%% HH_MODEL Implements the Hodgkin-Huxley neuron model equations
%
% Usage:
%   f = hh_model(x)
%
% Inputs:
%   x - State vector [v; n] where:
%       v: Membrane potential (mV)
%       n: Potassium activation gating variable
%
% Outputs:
%   f - Rate of change vector [dv/dt; dn/dt]
%
% Model Parameters:
%   Ib  = 10     % Baseline current (mA)
%   vNa = 50     % Sodium reversal potential (mV)
%   vK  = -77    % Potassium reversal potential (mV)
%   vL  = -54.4  % Leak reversal potential (mV)
%   C   = 1      % Membrane capacitance (F/cm^2)
%   gNa = 120    % Maximum sodium conductance (mS/cm^2)
%   gK  = 36     % Maximum potassium conductance (mS/cm^2)
%   gL  = 0.3    % Leak conductance (mS/cm^2)
%
% Author: Faranak Rajabi
% Version: 1.0 (October 24, 2024)

    % Extract state variables
    v = x(1);  % Membrane potential
    n = x(2);  % Potassium activation

    % Model parameters
    Ib  = 10;    % mA
    vNa = 50;    % mV
    vK  = -77;   % mV
    vL  = -54.4; % mV
    C   = 1;     % F/cm^2
    gNa = 120;   % mS/cm^2
    gK  = 36;    % mS/cm^2
    gL  = 0.3;   % mS/cm^2
    
    % Gating variable kinetics
    an = 0.01 * (v+55)./( 1-exp( -(v+55)/10 ) );
    bn = 0.125*exp( -(v+65)/80 );
    am = 0.1  * (v+40)./( 1-exp( -(v+40)/10 ) );
    bm = 4 * exp( -(v+65)/18 );
    minf = am ./ (am+bm);
    
    % Compute dynamics
    fv = ( Ib - gNa*minf.^3 .* (0.8-n).*(v-vNa) - gK*n.^4 .* (v-vK) - gL*(v-vL) )/C;
    fn = an.*(1-n) - bn.*n;

    f = [fv;fn];
end