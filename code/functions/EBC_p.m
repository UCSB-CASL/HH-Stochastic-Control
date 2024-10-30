function [Vp_wcwn, np_wcwn, u_integral_plot] = EBC_p(param, IC, uu, tt)
%%% EBC_P Performs event-based control for a population of coupled neurons
%
% Usage:
%   [Vp_wcwn, np_wcwn, u_integral_plot] = EBC_p(param, IC, uu, tt)
%
% Inputs:
%   param      - Structure containing simulation parameters
%       .M          - Number of simulation paths
%       .nNeurons   - Number of neurons in the population
%       .Tsim       - Total simulation time steps
%       .Dt         - Time step size
%       .R          - Ratio of simulation time step to control time step
%       .D_noise    - Noise intensity
%       .Vth        - Voltage threshold for control application
%       .Tth        - Time threshold for control application
%       .alpha      - Coupling strength matrix
%       .Ibvec      - Bias current vector
%       .umax       - Maximum control input amplitude
%   IC         - Initial conditions [V0; n0]
%   uu         - Control input sequence
%   tt         - Time vector
%
% Outputs:
%   Vp_wcwn         - Cell array containing voltage traces with control and noise
%   np_wcwn         - Cell array containing gating variable traces with control and noise
%   u_integral_plot - Array containing time and control integral values
%
% Description:
%   This function performs event-based control for a population of coupled neurons
%   described by the Hodgkin-Huxley model. It simulates the system for multiple paths
%   under three scenarios: without control and without noise, without control and with
%   noise, and with control and with noise.
%
%   The function uses the second-order Runge-Kutta method for numerical integration
%   and incorporates noise in the form of Brownian increments. It applies control
%   input when the average voltage of the neurons exceeds a specified threshold and
%   a certain time has elapsed since the last control application.
%
%   The function also generates plots for visualizing the voltage traces, control
%   input, and control integral. It calculates spike times and generates histograms
%   and raster plots for the different scenarios.
%
% Dependencies:
%   - hh_f_2d_p function
%   - interpolate_general function
%
% Note:
%   The function assumes the existence of the hh_f_2d_p function, which defines the
%   Hodgkin-Huxley model equations for a population of coupled neurons.
%
% Author: Faranak Rajabi
% Version: 1.0 (October 28, 2024)

% Modify the input.output as needed:
% [Vp_wcwn np_wcwn Vp_wcwn_cell]
% Event-Based Control for population of coupled neurons
% method   = 'linear'; 
% [uu, tt] = interpolate_general(tt_ode, uu_ode, param.Tsim, method); 
% creating an interpolated version of u(t) and time vector from ode solver to have a uniform grid values

param.Dt = tt(2) - tt(1); 
dt       = param.Dt / param.R; 

Vp_wcwn  = cell(param.M,1); % Voltage matrix when the control and noise are both applied
np_wcwn  = cell(param.M,1); % each row corresponds to one of the neurons in the population
Vp_wocwn = cell(param.M,1); % without control, with noise
np_wocwn = cell(param.M,1);
Vp_wocwon= cell(param.M,1); % without control, without noise
np_wocwon= cell(param.M,1);

Vavg_wcwn  = cell(param.M,1);
Vavg_wocwn = cell(param.M,1);
Vavg_wocwon= cell(param.M,1);

Tspike_wocwn  = cell(param.M,1);
Tspike_wcwn   = cell(param.M,1);
Tspike_wocwon = cell(param.M,1);

Neuronnumber_wocwn = cell(param.M,1);
Neuronnumber_wcwn  = cell(param.M,1);
Neuronnumber_wocwon= cell(param.M,1);

spikecount   = zeros(param.M,1); % total # of spikes at each path m
nThist       = 2*param.Dt:10*param.Dt:param.Tsim*param.Dt; % bins for the historgrams, not including 0 to avoid initial spike (as IC)
nShist_wocwn = zeros(length(nThist),param.M); 
nShist_wcwn  = zeros(length(nThist),param.M);
nShist_wocwon= zeros(length(nThist),param.M);

tmax_for_plot = 0;

for m = 1 : param.M
    Vp_wcwn{m} = zeros(param.nNeurons,param.Tsim); % Voltage matrix when the control and noise are both applied
    np_wcwn{m} = zeros(param.nNeurons,param.Tsim); % each row corresponds to one of the neurons in the population
    Vp_wocwn{m}= zeros(param.nNeurons,param.Tsim); % without control, with noise
    np_wocwn{m}= zeros(param.nNeurons,param.Tsim);
    Vavg_wcwn{m}  = zeros(1,param.Tsim);
    Vavg_wocwn{m} = zeros(1,param.Tsim);
    Tspike_wocwn{m}  = zeros(param.nNeurons,1);
    Tspike_wcwn{m}   = zeros(param.nNeurons,1);
    Tspike_wocwon{m} = zeros(param.nNeurons,1);
    
    randn('state',150*m);
    dW = sqrt(2*param.D_noise*dt)*randn(param.nNeurons,param.N); % Brownian increments
%     randn('state',11*m);
%     dW2 = sqrt(2*param.D_noise*dt)*randn(param.nNeurons/2,param.N); % because of memory contiguousy issues
%     W = cumsum(dW,2); % discretized Brownian path or Wiener trajectory
%                       % cumulative sum over 2nd dimension
    
    % without control, with noise
    Vp_wocwn{m}(:,1) = ones(param.nNeurons,1)*IC(1); 
    np_wocwn{m}(:,1) = ones(param.nNeurons,1)*IC(2); 
    Vavg_wocwn{m}(1) = 1/param.nNeurons * sum(Vp_wocwn{m}(:,1));
    for i = 1:param.Tsim
        Winc = sum(dW(:,param.R*(i-1)+1:param.R*i),2); % summing over 2nd dimension
%         Winc2 = sum(dW2(:,param.R*(i-1)+1:param.R*i),2); % summing over 2nd dimension
        [Fv1 Fn1] = hh_f_2d_p(Vp_wocwn{m}(:,i),np_wocwn{m}(:,i),param.alpha,param.Ibvec);

%         first order Euler-Maruyama integration (eqiv to Milstein integration for our additive noise)
%         Vp_wocwn{m}(:,i+1) = Vp_wocwn{m}(:,i) + Fv1*param.Dt + [Winc1;Winc2];
%         np_wocwn{m}(:,i+1) = np_wocwn{m}(:,i) + Fn1*param.Dt;

%         second order RK integration
        [Fv2 Fn2] = hh_f_2d_p(Vp_wocwn{m}(:,i) + Fv1*param.Dt + Winc , np_wocwn{m}(:,i) + Fn1*param.Dt , param.alpha,param.Ibvec);
        Vp_wocwn{m}(:,i+1) = Vp_wocwn{m}(:,i) + 0.5*param.Dt*(Fv1+Fv2) + Winc;
        np_wocwn{m}(:,i+1) = np_wocwn{m}(:,i) + 0.5*param.Dt*(Fn1+Fn2);
        
        Vavg_wocwn{m}(i+1) = 1/param.nNeurons * sum(Vp_wocwn{m}(:,i+1));
    end
    
    if param.nNeurons > 1
        figure(12)
        subplot(511)
        hold on
        plot(0:param.Dt:param.Dt*i,Vp_wocwn{m},'b');
        subplot(512)
        hold on
        plot(0:param.Dt:param.Dt*i,Vavg_wocwn{m},'b');
        hold on
        plot([0 param.Dt*i],[param.Vth param.Vth],':')
        
        figure(13)
        set(gcf,'Position',[100 0 650 900]);
        subplot(611)
        hold on
        plot(0:param.Dt:param.Dt*i,Vp_wocwn{m},'k','LineWidth',1.5);
        hold on
        plot(0:param.Dt:param.Dt*i,Vavg_wocwn{m},':','color',[0.8 0.8 0.8],'LineWidth',1.5);
        hold on
        plot([0 param.Dt*i],[param.Vth param.Vth],':','color',[0.5 0.5 0.5],'LineWidth',1.5);
        % set(gca,'FontName','Times','FontSize',16,'FontWeight','Bold','Box','on','XLim',[0 round(param.Dt*i)]);
        set(gca,'FontName','Times','FontSize',16,'FontWeight','Bold','Box','on','XLim',[0 round(param.Dt*i)], 'YLim', [-100 100]);
        ylabel('V_{i,woc}','FontName','Times','FontSize',18,'FontWeight','Bold');
        ylim([-100 100]); 
    end    
    
    % with control, with noise
    Vp_wcwn{m}(:,1) = ones(param.nNeurons,1)*IC(1); 
    np_wcwn{m}(:,1) = ones(param.nNeurons,1)*IC(2); 
    Vavg_wcwn{m}(1) = 1/param.nNeurons * sum(Vp_wcwn{m}(:,1));
    u = zeros(1,param.Tsim);
    flag = 0; count = 0;
    for i = 1:param.Tsim
        Winc = sum(dW(:,param.R*(i-1)+1:param.R*i),2); % summing over 2nd dimension
        
        if ~flag
            flag = Vavg_wcwn{m}(i)>param.Vth && param.Dt*i>param.Tth ; 
            if param.nNeurons==1 && count == length(uu)
                flag = 0;
            else
                count = 0;
            end
        end    
        if flag
            count = count + 1;
            u(i) = uu(count);
        end
        if count == length(uu)
            flag = 0;
        end
        clear Fv1 Fn1 Fv2 Fn2
        [Fv1 Fn1] = hh_f_2d_p(Vp_wcwn{m}(:,i),np_wcwn{m}(:,i),param.alpha,param.Ibvec);
        Fv1 = Fv1 + ones(param.nNeurons,1)*u(i);

%         first order Euler-Maruyama integration (eqiv to Milstein integration for our additive noise)
%         Vp_wcwn{m}(:,i+1) = Vp_wcwn{m}(:,i) + Fv1*param.Dt + [Winc1;Winc2];
%         np_wcwn{m}(:,i+1) = np_wcwn{m}(:,i) + Fn1*param.Dt;

%         second order RK integration
        [Fv2 Fn2] = hh_f_2d_p(Vp_wcwn{m}(:,i) + Fv1*param.Dt + Winc , np_wcwn{m}(:,i) + Fn1*param.Dt , param.alpha,param.Ibvec);
        Fv2 = Fv2 + ones(param.nNeurons,1)*u(i);
        Vp_wcwn{m}(:,i+1) = Vp_wcwn{m}(:,i) + 0.5*param.Dt*(Fv1+Fv2) + Winc;
        np_wcwn{m}(:,i+1) = np_wcwn{m}(:,i) + 0.5*param.Dt*(Fn1+Fn2);

        Vavg_wcwn{m}(i+1) = 1/param.nNeurons * sum(Vp_wcwn{m}(:,i+1));
    end
	
    % For exporting v_wcwn values for alpha robustness
    Vp_wcwn_cell = cell(3, 1); 
    if param.nNeurons > 1
        u_integral = cumsum(u.^2) * param.Dt;
        total_u_integral = u_integral(end);

        figure(12)
        subplot(513)
        hold on
        plot(0:param.Dt:param.Dt*i,Vp_wcwn{m},'b');
        subplot(514)
        hold on
        plot(0:param.Dt:param.Dt*i-param.Dt,u,'r');
        subplot(515)
        hold on
        plot(0:param.Dt:param.Dt*i,Vavg_wcwn{m},'b');
        hold on
        plot([0 param.Dt*i],[param.Vth param.Vth],':')
        
        %%%
        Vp_wcwn_cell{1} = 0:param.Dt:param.Dt*i;
        Vp_wcwn_cell{2} = Vp_wcwn{m};
        Vp_wcwn_cell{3} = Vavg_wcwn{m}; 
        %%%
        figure(13)
        subplot(612)
        plot(0:param.Dt:param.Dt*i,Vp_wcwn{m},'k','LineWidth',1.5);
        hold on
        plot(0:param.Dt:param.Dt*i,Vavg_wcwn{m},':','color',[0.8 0.8 0.8],'LineWidth',1.5);
        hold on
        plot([0 param.Dt*i],[param.Vth param.Vth],':','color',[0.5 0.5 0.5],'LineWidth',1.5);
        set(gca,'FontName','Times','FontSize',16,'FontWeight','Bold','Box','on','XLim',[0 round(param.Dt*i)]);
        ylabel('V_{i,wc}','FontName','Times','FontSize',18,'FontWeight','Bold');
        ylim([-100 100])
        
        subplot(613)
        hold on
        plot(0:param.Dt:param.Dt*i-param.Dt,u,'k','LineWidth',1.5);
        % set(gca,'FontName','Times','FontSize',16,'FontWeight','Bold',...
        %     'YLim',[-(param.umax+2) (param.umax+2)],'Box','on','XLim',[0 round(param.Dt*i)]);
        set(gca,'FontName','Times','FontSize',16,'FontWeight','Bold',...
            'YLim',[-(param.umax+2) (param.umax+2)],'Box','on','XLim',[0 round(param.Dt*i)]);
        ylabel('u (\muA/\muF)','FontName','Times','FontSize',18,'FontWeight','Bold');  

        u_integral_plot = zeros(2, length(0:param.Dt:param.Dt*i-param.Dt)); 
        u_integral_plot(1, :) = 0:param.Dt:param.Dt*i-param.Dt; 
        u_integral_plot(2, :) = u_integral; 
        figure(17)  % New figure for integral
        plot(0:param.Dt:param.Dt*i-param.Dt, u_integral, 'k', 'LineWidth', 1.5);
        set(gca, 'FontName', 'Times', 'FontSize', 16, 'FontWeight', 'Bold', ...
            'Box', 'on', 'XLim', [0 round(param.Dt*i)]);
        ylabel('∫u^2 dt', 'FontName', 'Times', 'FontSize', 18, 'FontWeight', 'Bold');
        xlabel('t (ms)', 'FontName', 'Times', 'FontSize', 18, 'FontWeight', 'Bold');
        title(sprintf('Integral of ∫u^2 dt, Final Value: %.2f', total_u_integral), ...
        'FontName', 'Times', 'FontSize', 18, 'FontWeight', 'Bold');
    end
    
    % without control, without noise
    Vp_wocwon{m}(:,1) = ones(param.nNeurons,1)*IC(1); 
    np_wocwon{m}(:,1) = ones(param.nNeurons,1)*IC(2); 
    Vavg_wocwon{m}(1) = 1/param.nNeurons * sum(Vp_wocwon{m}(:,1));
    for i = 1:param.Tsim
        [Fv1 Fn1] = hh_f_2d_p(Vp_wocwon{m}(:,i),np_wocwon{m}(:,i),param.alpha,param.Ibvec);
%         Fv1 = Fv1 + ones(param.nNeurons,1)*u(i);
%         first order Euler integration 
%         Vp_wocwon{m}(:,i+1) = Vp_wocwon{m}(:,i) + Fv1*param.Dt;
%         np_wocwon{m}(:,i+1) = np_wocwon{m}(:,i) + Fn1*param.Dt;

%         second order RK integration
        [Fv2 Fn2] = hh_f_2d_p(Vp_wocwon{m}(:,i) + Fv1*param.Dt, np_wocwon{m}(:,i) + Fn1*param.Dt , param.alpha,param.Ibvec);
%         Fv2 = Fv2 + ones(param.nNeurons,1)*u(i);
        Vp_wocwon{m}(:,i+1) = Vp_wocwon{m}(:,i) + 0.5*param.Dt*(Fv1+Fv2);
        np_wocwon{m}(:,i+1) = np_wocwon{m}(:,i) + 0.5*param.Dt*(Fn1+Fn2);
        
        Vavg_wocwon{m}(i+1) = 1/param.nNeurons * sum(Vp_wocwon{m}(:,i+1));
    end
	
    if param.nNeurons ==1
        % finding the index of first spiking instance after IC
        i_1stS_wocwon= find(Vp_wocwon{m}(21:length(Vp_wocwon{m}))>40,1)+20; 
        i_1stS_wocwn = find(Vp_wocwn{m}(21:length(Vp_wocwn{m}))>40,1)+20; 
        i_1stS_wcwn  = find(Vp_wcwn{m}(21:length(Vp_wcwn{m}))>40,1)+20; 
        % cutting the array down to a little after the first spike
        temp = Vp_wocwon{m}(1:i_1stS_wocwon+150); clear Vp_wocwon{m}; Vp_wocwon{m}=temp; clear temp;
        temp = np_wocwon{m}(1:i_1stS_wocwon+150); clear np_wocwon{m}; np_wocwon{m}=temp; clear temp;
        temp = Vp_wocwn{m}(1:i_1stS_wocwn+150);   clear Vp_wocwn{m};  Vp_wocwn{m}=temp;  clear temp;
        temp = np_wocwn{m}(1:i_1stS_wocwn+150);   clear np_wocwn{m};  np_wocwn{m}=temp;  clear temp;
        temp = Vp_wcwn{m}(1:i_1stS_wcwn+150);     clear Vp_wcwn{m};   Vp_wcwn{m}=temp;   clear temp;
        temp = np_wcwn{m}(1:i_1stS_wcwn+150);     clear np_wcwn{m};   np_wcwn{m}=temp;   clear temp;
        % Now plot figures 
        figure(14)
        hold on
        line(Vp_wocwn{m},np_wocwn{m},'Color',[0.75 0.75 0.75]);
        hold on
        line(Vp_wcwn{m},np_wcwn{m},'Color',[0.3 0.3 0.3]);
        hold on
        plot(-59.6,0.4026,'k*')
        hold on
        plot(Vp_wocwon{m},np_wocwon{m},'k--','LineWidth',2);
        
        figure(15)
        subplot(421)
        hold on
        plot(0:param.Dt:param.Dt*(length(Vp_wocwon{m})-1),Vp_wocwon{m},'k-','LineWidth',1.5);
        subplot(423)
        hold on
        plot(0:param.Dt:param.Dt*(length(Vp_wocwn{m})-1),Vp_wocwn{m},'k-','LineWidth',1.5);
        subplot(425)
        hold on
        plot(0:param.Dt:param.Dt*(length(Vp_wcwn{m})-1),Vp_wcwn{m},'k-','LineWidth',1.5);
        subplot(4,2,7:8)
        hold on
        plot(0:param.Dt:param.Dt*i-param.Dt,u,'k-','LineWidth',1.5);
        
        tmax_for_plot = max([tmax_for_plot,param.Dt*(length(Vp_wocwon{m})-1),param.Dt*(length(Vp_wocwn{m})-1),param.Dt*(length(Vp_wcwn{m})-1)]);
        
        figure(16)
        subplot(321)
        hold on
        plot(0:param.Dt:param.Dt*(length(Vp_wocwon{m})-1),Vp_wocwon{m},'k-','LineWidth',1.5);
        subplot(323)
        hold on
        plot(0:param.Dt:param.Dt*(length(Vp_wocwn{m})-1),Vp_wocwn{m},'k-','LineWidth',1.5);
        subplot(325)
        hold on
        plot(0:param.Dt:param.Dt*(length(Vp_wcwn{m})-1),Vp_wcwn{m},'k-','LineWidth',1.5);
        
    end
    
    % Data for the histograms and raster plots
    %without control, without noise
    clear temp r1 r2 row col
%     temp = Vp_wocwon{m}(:,1:param.Tsim); % making sure the possible initial spike is out.
    r1 = Vp_wocwon{m}>40; % recognizing spike instances
    r2 = r1==[0.5*ones(size(r1,1),1) r1(:,1:end-1)]; % to find multiple spikes, the number 0.5 is arbitrary, just not 1
    r1(r2) = 0; % eliminating multiple counts for one spike
    [row, col] = find(r1); % col+10 gives the time index of spikes
    spikecount(m) = sum(sum(r1)); % total # of spikes for the m-th path, all neurons together
    Tspike_wocwon{m} = (col)*param.Dt; % the times at which the neurons have spiked
    Neuronnumber_wocwon{m} = row; % the neuron number corresponding to each of the spike times above
    nShist_wocwon(:,m) = histc(Tspike_wocwon{m},nThist); % total number of spikes in each bin nThist for path m
    
    %without control, with noise
    clear temp r1 r2 row col
%     temp = Vp_wocwn{m}(:,1:param.Tsim); % making sure the possible initial spike is out.
    r1 = Vp_wocwn{m}>40; % recognizing spike instances
    r2 = r1==[0.5*ones(size(r1,1),1) r1(:,1:end-1)]; % to find multiple spikes, the number 0.5 is arbitrary, just not 1
    r1(r2) = 0; % eliminating multiple counts for one spike
    [row, col] = find(r1); % col+10 gives the time index of spikes
    spikecount(m) = sum(sum(r1)); % total # of spikes for the m-th path, all neurons together
    Tspike_wocwn{m} = (col)*param.Dt;
    Neuronnumber_wocwn{m} = row;
    nShist_wocwn(:,m) = histc(Tspike_wocwn{m},nThist);
    
    %with control, with noise
    clear temp r1 r2 row col
%     temp = Vp_wcwn{m}(:,1:param.Tsim); % making sure the possible initial spike is out.
    r1 = Vp_wcwn{m}>40; % recognizing spike instances
    r2 = r1==[0.5*ones(size(r1,1),1) r1(:,1:end-1)]; % to find multiple spikes, the number 0.5 is arbitrary, just not 1
    r1(r2) = 0; % eliminating multiple counts for one spike
    [row, col] = find(r1); % col+10 gives the time index of spikes
    spikecount(m) = sum(sum(r1)); % total # of spikes for the m-th path, all neurons together
    Tspike_wcwn{m} = (col)*param.Dt;
    % Tspike_wcwn{m} = 2.5*(col)*param.Dt;
    Neuronnumber_wcwn{m} = row;
    nShist_wcwn(:,m) = histc(Tspike_wcwn{m},nThist);
end
avg_nShist_wocwn  = mean(nShist_wocwn,2);
avg_nShist_wcwn   = mean(nShist_wcwn,2);
avg_nShist_wocwon = mean(nShist_wocwon,2);

if param.nNeurons == 1
    figure(15)
    set(gcf,'Position',[100 0 600 800]);
    hold on
    subplot(422)
    bar(nThist,param.M*avg_nShist_wocwon,'k','EdgeColor','k');
    set(gca,'YLim',[0 ceil(1.1*max(param.M*avg_nShist_wocwon))]);
    subplot(424)
    bar(nThist,param.M*avg_nShist_wocwn,'k','EdgeColor','k');
    set(gca,'YLim',[0 ceil(1.1*max(param.M*avg_nShist_wocwn))]);
    subplot(426)
    bar(nThist,param.M*avg_nShist_wcwn,'k','EdgeColor','k');
    set(gca,'YLim',[0 ceil(1.1*max(param.M*avg_nShist_wcwn))]);
    
    tmax_for_plot = tmax_for_plot-mod(tmax_for_plot,10) + 10;
    for kk = 1:2:5
        subplot(4,2,kk);
        set(gca,'FontName','Times','FontSize',16,'FontWeight','Bold',...
            'XLim',[0 tmax_for_plot],...
            'XTick',0:10:tmax_for_plot,...
            'YLim',[-100,50],'YTickLabel',[-100,-50,0,50],'Box','on');
        if kk==1; ylabel('V_{wocwon}','FontName','Times','FontSize',18,'FontWeight','Bold'); end
        if kk==3; ylabel('V_{wocwn}' ,'FontName','Times','FontSize',18,'FontWeight','Bold'); end
        if kk==5; ylabel('V_{wcwn}'  ,'FontName','Times','FontSize',18,'FontWeight','Bold'); end
    end
    for kk = 2:2:6
        subplot(4,2,kk);
        set(gca,'FontName','Times','FontSize',16,'FontWeight','Bold',...
            'XLim',[0 tmax_for_plot],'Box','on');
        ylabel('Count','FontName','Times','FontSize',18,'FontWeight','Bold');
    end
    subplot(4,2,5); xlabel('t (ms)','FontName','Times','FontSize',18,'FontWeight','Bold');
    subplot(4,2,6); xlabel('t (ms)','FontName','Times','FontSize',18,'FontWeight','Bold');
    
    
    figure(16)
    set(gcf,'Position',[100 0 600 500]);
    hold on
    subplot(322)
    bar(nThist,param.M*avg_nShist_wocwon,'k','EdgeColor','k');
    set(gca,'YLim',[0 ceil(1.1*max(param.M*avg_nShist_wocwon))]);
    subplot(324)
    bar(nThist,param.M*avg_nShist_wocwn,'k','EdgeColor','k');
    set(gca,'YLim',[0 ceil(1.1*max(param.M*avg_nShist_wocwn))]);
    subplot(326)
    bar(nThist,param.M*avg_nShist_wcwn,'k','EdgeColor','k');
    set(gca,'YLim',[0 ceil(1.1*max(param.M*avg_nShist_wcwn))]);
    
    for kk = 1:2:5
        subplot(3,2,kk);
        set(gca,'FontName','Times','FontSize',16,'FontWeight','Bold',...
            'XLim',[0 tmax_for_plot],...
            'XTick',0:10:tmax_for_plot,...
            'YLim',[-100,50],'YTickLabel',[-100,-50,0,50],'Box','on');
        if kk==1; ylabel('V_{wocwon}','FontName','Times','FontSize',18,'FontWeight','Bold'); end
        if kk==3; ylabel('V_{wocwn}','FontName','Times','FontSize',18,'FontWeight','Bold'); end
        if kk==5; ylabel('V_{wcwn}','FontName','Times','FontSize',18,'FontWeight','Bold'); end
    end
    for kk = 2:2:6
        subplot(3,2,kk);
        set(gca,'FontName','Times','FontSize',16,'FontWeight','Bold',...
            'XLim',[0 tmax_for_plot],'Box','on');
        ylabel('Count','FontName','Times','FontSize',18,'FontWeight','Bold');
    end
    subplot(3,2,5); xlabel('t (ms)','FontName','Times','FontSize',18,'FontWeight','Bold');
    subplot(3,2,6); xlabel('t (ms)','FontName','Times','FontSize',18,'FontWeight','Bold');
    
else
    figure
    subplot(611)
    bar(nThist,avg_nShist_wocwon);
    subplot(612)
    bar(nThist,avg_nShist_wocwn);
    subplot(613)
    bar(nThist,avg_nShist_wcwn);
    subplot(6,1,4:6)
    plot(Tspike_wcwn{m},Neuronnumber_wcwn{m},'.')
    
    figure(13)
    subplot(6,1,4:6)
    plot(Tspike_wcwn{m},Neuronnumber_wcwn{m},'k.','MarkerSize',15);
    set(gca,'FontName','Times','FontSize',16,'FontWeight','Bold','Box','on','XLim',[0 round(param.Dt*i)]);
    xlabel('t (ms)','FontName','Times','FontSize',18,'FontWeight','Bold');
    ylabel('Neuron Number','FontName','Times','FontSize',18,'FontWeight','Bold');
    save_str = sprintf('D_HJB_%.1f_D_p_%.1f_alpha_%.2f.jpg', param.D_HJB, param.D_noise, param.alpha(1, 2));
    title_str = sprintf('Results for u Integrated for population level. \n D_{HJB} = %.2f, D_{ODE} = %.2f, \\alpha = %.2f', param.D_HJB, param.D_noise, param.alpha(1, 2));
    sgtitle(title_str, 'FontName', 'Times', 'FontSize', 18, 'FontWeight', 'Bold');

    saveas(figure(13), save_str);
    disp(['Saving figure as: ' save_str]);
end

% Proof of Concept figure
clear r1 r2 row col Tspike_wocwn Neuronnumber_wocwn nShist_wocwn
if IC == [-59.6 0.403]
    for m = 1:param.M
        % finding the index of first spiking instance after IC
        [r c] = find(Vp_wocwn{m}>40); %row and column index of all V values above 40, each row is for one neuron
        for k = 1 : param.nNeurons
            i_1stS_wocwn(k,1) = c(find(r==k,1)); % finding the first time for which V goes above 40 for each neuron
         
            % cutting the array down to a little after the first spike
            tempVp_wocwn = Vp_wocwn{m}(k,1:i_1stS_wocwn(k)+150);   
            Tspike_wocwn{m}(k,1) = param.Dt*find(tempVp_wocwn==max(tempVp_wocwn)); % 1st spike time of neuron k in the mth path
            
            figure(20)
            set(gcf,'Position',[400 200 560 450]);
            subplot(211)
            hold on
            plot(0:param.Dt:param.Dt*(length(tempVp_wocwn)-1),tempVp_wocwn,'k-','LineWidth',1.5);
            set(gca,'FontName','Times','FontSize',16,'FontWeight','Bold','YTick',[-100,-50,0,50],...
                'YTickLabel',[-100,-50,0,50],'YMinorTick','on','XMinorTick','on','Box','on');
            XLim = get(gca,'XLim');
            ylabel('V (mV)','FontName','Times','FontSize',18,'FontWeight','Bold');
        end
        if param.nNeurons > 1
            figure(20)
            subplot(212)
            hist(Tspike_wocwn{m},nThist);
            h = findobj(gca,'Type','patch');
            set(h,'FaceColor','k','EdgeColor','k');
            set(gca,'FontName','Times','FontSize',16,'FontWeight','Bold','XLim',XLim,'YLim',[0,max(hist(Tspike_wocwn{m},nThist))+2],...
                'YMinorTick','on','XMinorTick','on',...
                'Box','on');
            xlabel('t (ms)','FontName','Times','FontSize',18,'FontWeight','Bold');
            ylabel('Count','FontName','Times','FontSize',18,'FontWeight','Bold');
        end
    end
    if param.nNeurons == 1
        for m= 1:param.M
            Tspike_wocwnMAT(m) = Tspike_wocwn{m};
        end
        figure(20)
        subplot(212)
        hist(Tspike_wocwnMAT,nThist);
        h = findobj(gca,'Type','patch');
        set(h,'FaceColor','k','EdgeColor','k')
        set(gca,'FontName','Times','FontSize',16,'FontWeight','Bold','XLim',XLim,'YLim',[0,max(hist(Tspike_wocwn{m},nThist))+2],...
                'YMinorTick','on','XMinorTick','on',...
                'Box','on');
        xlabel('t (ms)','FontName','Times','FontSize',18,'FontWeight','Bold');
        ylabel('Count','FontName','Times','FontSize',18,'FontWeight','Bold');
    end
end
