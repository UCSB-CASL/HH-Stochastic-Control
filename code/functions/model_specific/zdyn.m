function [fz, u_interpolated] = zdyn(t, x, params)
%%% ZDYN Computes dynamics and control for HH system in transformed coordinates
%
% Usage:
%   [fz, u_interpolated] = zdyn(t, x, params)
%
% Inputs:
%   t      - Current time
%   x      - State vector [v/Ks; n]
%   params - Structure containing:
%       .control    - Control type ('off'/'deterministic'/'Stochastic')
%       .feedback   - Feedback flag ('on'/'off')
%       .umax      - Maximum control magnitude
%       .F_ustar_deterministic - Interpolant for deterministic control
%       .F_ustar_stochastic   - Interpolant for stochastic control
%       .index     - Index for stochastic control
%       .u_deter_Stoch_mat    - Pre-computed stochastic controls
%       .t_deter_Stoch_mat    - Time points for stochastic controls
%       .u_deterministic      - Pre-computed deterministic control
%       .t_deterministic      - Time points for deterministic control
%
% Outputs:
%   fz             - State derivatives [dx/dt; dy/dt; du^2/dt]
%   u_interpolated - Interpolated control value
%
% Dependencies:
%   - hh_model.m
%   - saturation.m
%   - Requires global variable Ks
%
% Author: Faranak Rajabi
% Version: 1.0 (October 24, 2024)

    global Ks
    % Transform coordinates
    v = Ks * x(1);
    n = x(2);

    % Get HH dynamics
    [f] = hh_model([v;n]);
    fv = f(1);
    fn = f(2);

    % Determine control based on mode
    if strcmp(params.control,'off')
        u_interpolated = 0;

    elseif strcmp(params.control,'deterministic') && strcmp(params.feedback,'on')
        % Deterministic feedback control
        u_interp = params.F_ustar_deterministic({x(1), x(2), t});
        u_interpolated = saturation(params.umax,-params.umax, u_interp);

    elseif strcmp(params.control,'Stochastic') && strcmp(params.feedback,'on')
        % Stochastic feedback control
        index = params.index;
        u_interp = params.F_ustar_stochastic{index}({x(1), x(2), t});
        u_interpolated = saturation(params.umax,-params.umax, u_interp);

    elseif strcmp(params.control,'Stochastic') && strcmp(params.feedback,'off')
        % Pre-computed stochastic control
        index = params.index;
        u_deter_Stoch_vec = params.u_deter_Stoch_mat(:,index);
        t_deter_Stoch_vec = params.t_deter_Stoch_mat(:,index);
        u_interp = interp1(t_deter_Stoch_vec,u_deter_Stoch_vec,t,"linear");
        u_interpolated = saturation(params.umax,-params.umax, u_interp);

    elseif strcmp(params.control,'deterministic') && strcmp(params.feedback,'off')
        % Pre-computed deterministic control
        u_deterministic = params.u_deterministic;
        t_deterministic = params.t_deterministic;
        u_interp = interp1(t_deterministic,u_deterministic,t,"linear");
        u_interpolated = saturation(params.umax,-params.umax, u_interp);
    end

    % Compute derivatives
    fx = 1 / Ks * fv + 1 / Ks * u_interpolated;
    fy = fn;
    fz = [fx; fy; u_interpolated^2];
end