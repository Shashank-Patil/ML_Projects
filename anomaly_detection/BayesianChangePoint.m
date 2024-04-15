%Bayesian Change Point Detection Algorithm
% References used: 
%1) "https://github.com/y-bar/bocd"
%2) "https://github.com/gwgundersen/bocd/blob/master/bocd.py"


function runlength_posterior = BayesianChangePoint(x, lamda, mu0, kappa0, alpha0, beta0)

    % Declaring the hyperparamteres: mu,kappa,alpha,beta, and posterior predictive and run-length posterior of appropriate dimensions
    %which are updated later
   
    mu = zeros(1, length(x));
    kappa = zeros(1, length(x));
    alpha = zeros(length(x)+1, length(x)+1);
    beta = zeros(length(x)+1, length(x)+1);
    upm_predictive = zeros(1,length(x));
    runlength_posterior = zeros(length(x)+1, length(x)+1);
    
    % Initializing the values of the hyperparameters as given in the question
    mu(1) = mu0;
    kappa(1) = kappa0;
    alpha(:,1) = alpha0;
    beta(:,1) = beta0;
    
    % Defining the hazard function value or CP prior
    H = 1 / lamda;

 
    for t = 1 : length(x)
        
        % predictive posteriors as per the t-distribution 
        upm_predictive = find_upm(x(t), upm_predictive, t, mu, alpha, beta, kappa);
        
       
        if t ~= 1
            % calculating the growth probability
            runlength_posterior(t,2:t+1) = runlength_posterior(t-1,1:t) .* upm_predictive(1:t) * (1-H);

            % calculating the changepoint probabilities
            runlength_posterior(t,1) = sum(runlength_posterior(t-1,1:t) .* upm_predictive(1:t) * H);

            % normalizing factor/evidence
            evidence = sum(runlength_posterior(t,:));
            
            %Calculating the run length posterior
            runlength_posterior(t,:) = runlength_posterior(t,:) / evidence;
        
        
        % Given:Initially p(r0=0) = 1    
        else
           runlength_posterior(1,1) = upm_predictive(1)*H;
           runlength_posterior(2,1) = upm_predictive(1)*(1-H);
           
        end
        
        % updating the hyperparameters 
        [mu, alpha, beta, kappa] = update_hyperparams(x(t), t, mu, alpha, beta, kappa);

    end

end


%Helper Function-1
%Finding the upm predictive 
function p_t = find_upm(xt, p_t, t, mu, alpha, beta, kappa)
 
    
    for i = 1: t
        d = (alpha(t,i)*kappa(i));
        scale = sqrt(beta(t,i)*(kappa(i)+1)/d);
        p_t(i) = pdf('tLocationScale', xt, mu(i), scale, 2*alpha(t,i));
    end

end

%Helper Function-2
%Updating the hyperparameters
function [mu, alpha, beta, kappa] = update_hyperparams(xt, t, mu, alpha, beta, kappa)

%Ref: K. P. Murphy, “Conjugate bayesian analysis of the gaussian distribution,” tech. rep., 2007

    alpha(t+1,2:t+1) = alpha(t,1:t) + 0.5;
    beta(t+1,2:t+1) = beta(t,1:t) + kappa(1:t).*(xt-mu(1:t)).^2./(2*(kappa(1:t)+1));
    mu(2:t+1) = (kappa(1:t).*mu(1:t)+xt)./(kappa(1:t)+1);
    kappa(2:t+1) = kappa(1:t)+1;
end


%Helper Function-3
%Plotting the graph: run_length vs Time
% Ref -- "https://github.com/gwgundersen/bocd/blob/master/bocd.py"

function [] = plot_graph_Q2(N, observations, RL_probabs)
% N --> Number of data points
% observations --> data for CP detection
% RL_probabs --> Run length probabilities
    figure;
    subplot(2,1,1);
    hold on;
    plot(linspace(1, N, N), observations)
    title('Residuals vs Time after 2OO observations')
    xlim([0 N])
    
    
    subplot(2,1,2);
    hold on;
    colormap(flipud(gray(300)));
    title('Run length Posterior of residuals')
    RL_probabs = fliplr(rot90(RL_probabs, 3));
    image(1e5*RL_probabs);
    xlim([0 N])
end


% Helper Function -4
function [] = plot_graph_Q1(N, observations, RL_probabs)
% N --> Number of data points
% observations --> data for CP detection
% RL_probabs --> Run length probabilities
    figure;
    subplot(2,1,1);
    hold on;
    plot(linspace(1, N, N), observations)
    title('NMRlogWell data')
    xlim([0 N])
    
    
    subplot(2,1,2);
    hold on;
    colormap(flipud(gray(300)));
    title('Run length Posterior of the data')
    RL_probabs = fliplr(rot90(RL_probabs, 3));
    image(1e8*RL_probabs);
    xlim([0 N])
end

