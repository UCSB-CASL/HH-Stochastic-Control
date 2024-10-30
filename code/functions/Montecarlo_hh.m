function [mat_final, result_once] = Montecarlo_hh(x_initial, params, D_new_values, n_simulation, only_once)
%%% MONTECARLO_HH Performs Monte Carlo simulation of Hodgkin-Huxley control system
%
% Usage:
%   [mat_final, result_once] = Montecarlo_hh(x_initial, params, D_new_values, n_simulation, only_once)
%
% Inputs:
%   x_initial    - Initial state vector [v; n]
%   params       - Structure containing simulation parameters:
%       .Dt         - Time step
%       .tFinal     - Final simulation time
%       .tInitial   - Initial simulation time
%       .feedback   - Control feedback flag ('on'/'off')
%       .control    - Control type ('Stochastic'/'deterministic')
%   D_new_values - Vector of noise intensity values
%   n_simulation - Number of Monte Carlo runs
%   only_once    - Flag for single simulation (true) or full analysis (false)
%
% Outputs:
%   mat_final   - Cell array containing full analysis results:
%       {j,1} - Statistics for stochastic control
%       {j,2} - Statistics for deterministic control
%       {j,3} - State trajectories for stochastic control
%       {j,4} - State trajectories for deterministic control
%   result_once - Cell array containing single simulation results:
%       {j,1} - Control signal
%       {j,2} - State trajectory [v, n]
%
% In main implementation:
%   For single simulation (original behavior):
%   [~, result_once] = Montecarlo_hh(x_initial, params, D_new_values, 1, true);
% 
%   For full Monte Carlo analysis:
%   [mat_final, ~] = Montecarlo_hh(x_initial, params, D_new_values, n_simulation, false);
%
% Author: Faranak Rajabi
% Version: 1.0 (October 24, 2024)

    global Ks
    dt = params.Dt;
    tFinal = params.Tend;
    tInitial = params.tInitial;
    result_once = cell(length(D_new_values), 2);
    
    % Initialize storage for full analysis if needed
    if ~only_once
        mat_final = cell(length(D_new_values), 4);
    end

    for j = 1:length(D_new_values)
        params.index = j;
        params.D_new_values = D_new_values(j);
        t_mat = 0:dt:tFinal;

        if only_once
            % Single simulation mode
            n_sim_local = 1;
            plot_once_u = zeros(length(t_mat), 1);
            plot_once_state = zeros(length(t_mat), 2);
        else
            % Full analysis mode
            n_sim_local = n_simulation;
            mat_stoc_u_ST = zeros(n_sim_local, 7);
            mat_stoc_u_det = zeros(n_sim_local, 7);
            mat_stoc_state_ST = zeros(length(t_mat), 2*n_sim_local);
            mat_stoc_state_u_det = zeros(length(t_mat), 2*n_sim_local);
        end

        for i = 1:n_sim_local
            % Run stochastic simulation
            params.control = 'Stochastic';
            [state_stochastic, uu_stochastic_ST_n] = sde_hh_model_solver(x_initial, params, D_new_values(j));
            
            % Run deterministic simulation
            params.control = 'deterministic';
            [state_stochastic_u_det, uu_deterministic_ST_n] = sde_hh_model_solver(x_initial, params, D_new_values(j));

            if only_once
                % Store single simulation results
                if strcmp(params.feedback, 'on')
                    plot_once_u = uu_stochastic_ST_n;
                else
                    plot_once_u = uu_deterministic_ST_n;
                end
                plot_once_state(:, 1) = state_stochastic(:, 1);
                plot_once_state(:, 2) = state_stochastic(:, 2);
            else
                % Compute and store full analysis results
                % Interpolate control signals
                F_1 = griddedInterpolant(t_mat, uu_stochastic_ST_n);
                F_2 = griddedInterpolant(t_mat, uu_deterministic_ST_n);
                
                % Calculate integrals
                q_1 = integral(@(x) F_1(x).^2, 0, tFinal, 'RelTol', 1e-8, 'AbsTol', 1e-13);
                q_2 = integral(@(x) F_2(x).^2, 0, tFinal, 'RelTol', 1e-8, 'AbsTol', 1e-13);
                
                % Compute metrics for stochastic control
                J_u = q_1;
                J_end_euclidean = EuclidianNorm(state_stochastic(end, 1) * Ks, state_stochastic(end, 2), 1);
                J_end = terminal_penalty(state_stochastic(end, 1), state_stochastic(end, 2));
                J_total_euclid = 1000 * J_end_euclidean + J_u;
                J_total = J_end + J_u;
                
                % Compute metrics for deterministic control
                J_u_2 = q_2;
                J_end_euclidean_2 = EuclidianNorm(state_stochastic_u_det(end, 1) * Ks, state_stochastic_u_det(end, 2), 1);
                J_end_2 = terminal_penalty(state_stochastic_u_det(end, 1), state_stochastic_u_det(end, 2));
                J_total_euclid_2 = 1000 * J_end_euclidean_2 + J_u_2;
                J_total_2 = J_end_2 + J_u_2;
                
                % Store results
                mat_stoc_u_ST(i, :) = [J_u, J_end_euclidean, J_end, J_total_euclid, J_total, ...
                                     state_stochastic(end, 1) * Ks, state_stochastic(end, 2)];
                mat_stoc_u_det(i, :) = [J_u_2, J_end_euclidean_2, J_end_2, J_total_euclid_2, J_total_2, ...
                                      state_stochastic_u_det(end, 1) * Ks, state_stochastic_u_det(end, 2)];
                
                mat_stoc_state_ST(:, 2*(i-1)+1 : 2*i) = state_stochastic;
                mat_stoc_state_u_det(:, 2*(i-1)+1 : 2*i) = state_stochastic_u_det;
            end
        end

        if only_once
            % Save single simulation results
            result_once{j, 1} = plot_once_u;
            result_once{j, 2} = plot_once_state;
            mat_final = 0;
        else
            % Save full analysis results
            mat_final{j, 1} = mat_stoc_u_ST;
            mat_final{j, 2} = mat_stoc_u_det;
            mat_final{j, 3} = mat_stoc_state_ST;
            mat_final{j, 4} = mat_stoc_state_u_det;
            result_once = 0;
        end
    end
end