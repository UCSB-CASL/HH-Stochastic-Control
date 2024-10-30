function [J_mat,u] = CostandControl_traj(t,x,params)
%%% COSTANDCONTROL_TRAJ Calculates the cost and control trajectory
%
% Usage:
%   [J_mat, u] = CostandControl_traj(t, x, params)
%
% Inputs:
%   t      - Time vector
%   x      - State trajectory matrix [v, n, J_u]
%   params - Structure containing simulation parameters
%
% Outputs:
%   J_mat - Matrix containing cost values [J_u, J_end_euclidean, J_end, J_total_euclid, J_total]
%   u     - Control input trajectory
%
% Global Variables:
%   Ks - Scaling factor for voltage
%
% Description:
%   This function calculates the cost and control trajectory for a given state trajectory.
%   It uses the zdyn function to compute the control input at each time step and stores
%   it in the u vector. The function then calculates various cost terms, including the
%   control cost (J_u), the Euclidean distance from the desired state (J_end_euclidean),
%   the terminal penalty (J_end), and the total cost (J_total_euclid and J_total).
%
%   The cost terms are computed as follows:
%   - J_u: The control cost, which is the integral of the squared control input over time.
%          It is computed as the last element of the state trajectory matrix x.
%   - J_end_euclidean: The Euclidean distance between the scaled final voltage (x(end,1)*Ks)
%                      and the final gating variable (x(end,2)).
%   - J_end: The terminal penalty, calculated using the terminal_penalty function.
%   - J_total_euclid: The total cost, which is the sum of the control cost (J_u) and the
%                     weighted Euclidean distance (1000*J_end_euclidean).
%   - J_total: An alternative total cost, which is the sum of the control cost (J_u) and
%              the terminal penalty (J_end).
%
% Dependencies:
%   - zdyn function
%   - EuclidianNorm function
%   - terminal_penalty function
%
% Note:
%   The function assumes the existence of the zdyn, EuclidianNorm, and terminal_penalty
%   functions. Make sure these functions are defined and available in the same directory
%   or in the MATLAB path.
%
% Author: Faranak Rajabi
% Version: 1.0 (October 28, 2024)

global Ks

u = zeros(size(t));
for  i = 1:length(t)
   [fz,u_interpolated] = zdyn(t(i),x(i,1:2),params);
   u(i) = u_interpolated;
end

J_u = x(end,3);
J_end_euclidean = EuclidianNorm(x(end,1)*Ks, x(end,2), 1);
J_end = terminal_penalty(x(end,1), x(end,2));
J_total_euclid = 1000*J_end_euclidean + J_u;
J_total = J_end + J_u;
J_mat = [J_u J_end_euclidean J_end J_total_euclid J_total];

end
