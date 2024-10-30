function [Fv Fn] = hh_f_2d_p(v,n,Alpha,Ibvec)
%%% HH_F_2D_P Defines the Hodgkin-Huxley model equations for a population of coupled neurons
%
% Usage:
%   [Fv, Fn] = hh_f_2d_p(v, n, Alpha, Ibvec)
%
% Inputs:
%   v     - Membrane potential vector (mV)
%   n     - Gating variable vector
%   Alpha - Coupling strength matrix
%   Ibvec - Bias current vector (mA)
%
% Outputs:
%   Fv - Derivative of membrane potential (dv/dt)
%   Fn - Derivative of gating variable (dn/dt)
%
% Description:
%   This function defines the Hodgkin-Huxley model equations for a population of coupled
%   neurons. It calculates the derivatives of the membrane potential (Fv) and the gating
%   variable (Fn) based on the current state of the system.
%
%   The model parameters are defined as follows:
%   - vNa: Sodium reversal potential (mV)
%   - vK: Potassium reversal potential (mV)
%   - vL: Leak reversal potential (mV)
%   - C: Membrane capacitance (F/cm^2)
%   - gNa: Sodium conductance (mS/cm^2)
%   - gK: Potassium conductance (mS/cm^2)
%   - gL: Leak conductance (mS/cm^2)
%
%   The function first calculates the activation and inactivation variables (an, bn, am, bm)
%   based on the current membrane potential. It then computes the steady-state activation of
%   the sodium current (minf).
%
%   The coupling term (fcp) is calculated based on the coupling strength matrix (Alpha) and
%   the difference between the membrane potential of each neuron and the average membrane
%   potential of the population.
%
%   Finally, the derivatives of the membrane potential (Fv) and the gating variable (Fn)
%   are computed using the Hodgkin-Huxley model equations, taking into account the coupling
%   term and the bias current (Ibvec).
%
% Note:
%   - The function assumes that the input vectors v, n, and Ibvec have the same length,
%     corresponding to the number of neurons in the population.
%   - The coupling strength matrix (Alpha) should be a square matrix with dimensions equal
%     to the number of neurons in the population.
%
% Author: Faranak Rajabi
% Version: 1.0 (October 28, 2024)

% function dY = hh_f_2d(t,Y)
% v= Y(1); 
% n= Y(2);
N = length(v); % number of neurons in the population
% Ib  = 10;    % mA
vNa = 50;    % mV
vK  = -77;   % mV
vL  = -54.4; % mV
C   = 1;     % F/cm^2
gNa = 120;   % mS/cm^2
gK  = 36;    % mS/cm^2
gL  = 0.3;   % mS/cm^2

% randn('state',100);
% Ib = Ib+2*randn(N,1); % to induce heterogeniety in individual neurons

% if length(alpha)==2 % alpha is in the form alpha = [mean,std]
%     randn('state',200);
%     temp = alpha(1)+alpha(2)*randn(N*(N-1)/2,1); % randomly assigning alpha values for the entire system
%     fprintf('mean of alphas is %g\n minimum alpha is %g\n',mean(temp),min(temp));
%     if min(temp)<0; temp = temp - min(temp); end % avoiding negative alphas
%     fprintf('mean of shifted alphas is %g\n',mean(temp));
%     Alpha = triu(ones(N),1);  % making the ALpha matrix first as an upper triangular matrix with zero diag
%     Alpha(Alpha==1) = temp;   % placing the temp into matrix Alpha
%     Alpha = Alpha +Alpha';    % Making Alpha symmetric aij = aji
% else %alpha is just a scalar: e.g., alpha = 0.1
%     Alpha = alpha*ones(N);
%     Alpha = Alpha - diag(diag(Alpha));
% end
an = 0.01 * (v+55)./( 1-exp( -(v+55)/10 ) );
bn = 0.125*exp( -(v+65)/80 );
am = 0.1  * (v+40)./( 1-exp( -(v+40)/10 ) );
bm = 4 * exp( -(v+65)/18 );
minf = am ./ (am+bm);


%coupling term:
fcp = zeros(N,1);
for i = 1 : N
%     fcp(i,1) = alpha/N * sum(v-v(i)); % f of coupling
    fcp(i,1) = 1/N * sum(Alpha(:,i).*(v-v(i))); % f of coupling
end
Fv = ( Ibvec - gNa*minf.^3 .* (0.8-n).*(v-vNa) - gK*n.^4 .* (v-vK) - gL*(v-vL) )/C +fcp;
Fn = an.*(1-n) - bn.*n;
% dY = [fv ;fn];

%  0.1*(v+40)./(0.1*(v+40)+(1-exp(-(v+40)/10)).*4*exp(-(v+65)/18))