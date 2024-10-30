function [u_integral_plot, control_count] = EBC_p_no_plot(param, IC, uu, tt, realization_seed)
%%% EBC_P_NO_PLOT Calculates the control integral for event-based control
%%% for population level
%
% Usage:
%   [u_integral_plot, control_count] = EBC_p_no_plot(param, IC, uu, tt, realization_seed)
%
% Inputs:
%   param           - Structure containing simulation parameters
%       .Dt            - Time step size
%       .R             - Ratio of simulation time step to control time step
%       .M             - Number of simulation paths
%       .nNeurons      - Number of neurons
%       .Tsim          - Total simulation time steps
%       .D_noise       - Noise intensity
%       .Vth           - Voltage threshold for control application
%       .Tth           - Time threshold for control application
%       .alpha         - Parameter for hh_f_2d_p function
%       .Ibvec         - Parameter for hh_f_2d_p function
%   IC              - Initial conditions [V0; n0]
%   uu              - Control input sequence
%   tt              - Time vector
%   realization_seed - Seed for random number generation
%
% Outputs:
%   u_integral_plot - Array containing time and control integral values
%   control_count   - Number of times control is applied
%
% Description:
%   This function implements the event-based control algorithm for a network of neurons 
%   described by the Hodgkin-Huxley model. The algorithm applies control input when the 
%   average voltage of the neurons exceeds a specified threshold and a certain time has 
%   elapsed since the last control application.
%
%   The function simulates the system for multiple paths and calculates the control 
%   integral and the number of times control is applied. It uses the second-order 
%   Runge-Kutta method for numerical integration and incorporates noise in the form of 
%   Brownian increments.
%
% Dependencies:
%   - hh_f_2d_p function
%
% Note:
%   The function assumes the existence of the hh_f_2d_p function, which defines the 
%   Hodgkin-Huxley model equations.
%
% Author: Faranak Rajabi
% Version: 1.0 (October 28, 2024)

% Initialize parameters
param.Dt = tt(2) - tt(1); 
dt = param.Dt / param.R; 

% Initialize variables to store results
Vp_wcwn = cell(param.M,1); 
np_wcwn = cell(param.M,1); 
Vavg_wcwn = cell(param.M,1);

% Loop over each path
for m = 1 : param.M
    % Initial conditions for control and noise
    Vp_wcwn{m}(:,1) = ones(param.nNeurons,1) * IC(1); 
    np_wcwn{m}(:,1) = ones(param.nNeurons,1) * IC(2); 
    Vavg_wcwn{m}(1) = 1 / param.nNeurons * sum(Vp_wcwn{m}(:,1));
    
    % Initialize control signal and control application flags
    u = zeros(1, param.Tsim);
    flag = 0; 
    count = 0;
    control_count = 0;

    randn('state',150*realization_seed);
    dW = sqrt(2*param.D_noise*dt)*randn(param.nNeurons,param.N); % Brownian increments
    for i = 1:param.Tsim
        % Calculate noise increments
        Winc = sum(dW(:,param.R*(i-1)+1:param.R*i),2); 
        
        % Check if control should be applied based on average voltage
        if ~flag
            flag = Vavg_wcwn{m}(i) > param.Vth && param.Dt * i > param.Tth; 
            if param.nNeurons == 1 && count == length(uu)
                flag = 0;
            else
                count = 0;
            end
        end    
        if flag
            count = count + 1;
            u(i) = uu(count);
            control_count = control_count + 1;
        end
        if count == length(uu)
            flag = 0;
        end
        
        % Update average voltage with the provided `dW`
        [Fv1, Fn1] = hh_f_2d_p(Vp_wcwn{m}(:,i), np_wcwn{m}(:,i), param.alpha, param.Ibvec);
        Fv1 = Fv1 + ones(param.nNeurons,1) * u(i);
        
        % Apply the second-order RK integration
        [Fv2, Fn2] = hh_f_2d_p(Vp_wcwn{m}(:,i) + Fv1 * param.Dt + Winc, np_wcwn{m}(:,i) + Fn1 * param.Dt, param.alpha, param.Ibvec);
        Fv2 = Fv2 + ones(param.nNeurons,1) * u(i);
        Vp_wcwn{m}(:,i+1) = Vp_wcwn{m}(:,i) + 0.5 * param.Dt * (Fv1 + Fv2) + Winc;
        np_wcwn{m}(:,i+1) = np_wcwn{m}(:,i) + 0.5 * param.Dt * (Fn1 + Fn2);

        Vavg_wcwn{m}(i+1) = 1 / param.nNeurons * sum(Vp_wcwn{m}(:,i+1));
    end
    
    % Calculate u_integral
    u_integral = cumsum(u.^2) * param.Dt;
    u_integral_plot = zeros(2, length(0:param.Dt:param.Dt*i - param.Dt)); 
    u_integral_plot(1, :) = 0:param.Dt:param.Dt*i - param.Dt; 
    u_integral_plot(2, :) = u_integral; 
end
end
