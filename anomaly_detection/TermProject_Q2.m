%% Question 2a) Finding the appropriate model
% Note - Attach the 'BayesianChangePoint.m' before running this 
 
load 'historic.mat'
yk = y';
N = length(yk);
figure(1)
plot(1:N,yk, 'LineWidth', 1)
ylabel('Y')
xlabel('Time')
title('Y vs Time')

acf_yk = autocorr(yk,'NumLags',12);
pacf_yk = parcorr(yk,'NumLags',12);

%  Plot sample ACF and sample PACF
figure(2);
subplot(211)
autocorr(yk,'NumLags',12);
ylabel('Sample ACF');
xlabel('')
set(gca,'fontsize',12,'fontweight','bold');
hca = gca;
set(hca.Children(4),'LineWidth',2,'Color','m')
hca.YLim(2) = 1.1;
box off
subplot(212)
parcorr(yk,'NumLags',12);
ylabel('Sample PACF');
set(gca,'fontsize',12,'fontweight','bold');
hca = gca;
set(hca.Children(4),'LineWidth',2)
box off

% From PACF graph proposing an AR(1) model

yk1 = yk(1:49, :);
yt  = yk(2:50, :);
mdl = fitlm(yk1, yt); % We get the constant value to be insignificant

% Residual Analysis

res_mdl= yt - predict(mdl, yk1);
figure(3);
autocorr(res_mdl,'NumLags',12)
ylabel('Sample ACF');
set(gca,'fontsize',12,'fontweight','bold');
title('ACF of residuals of the proposed AR(1) model')
hca = gca;
set(hca.Children(4),'LineWidth',2)
box off

% Whiteness test of residuals
[h_res_mdl,pval_res_mdl] = lbqtest(res_mdl);
% This approves that residuals are White Noise 

%Parameter_estimate of the AR(1) model
parameter = mdl.Coefficients.Estimate(2);

%% Question 2a) Change Point Detection

load 'new.mat'
ynew = y';

% Recursive LS till 200th observation
% Parameter value = -0.6117 from previous question
obj = recursiveLS(1,'InitialParameters',-0.6117, 'InitialParameterCovariance',1);
thetaest_vec1 = []; Ptheta_vec1 = [];
x = ynew(1:199);
y_out = ynew(2:200);

for i = 1:numel(x)
    H = x(i);
    [theta1,~] = obj(y_out(i),H);
    Ptheta_vec1(i) = obj.ParameterCovariance;
    thetaest_vec1(i) = theta1;
end

% Updated Parameter of the AR model until 200th observation
Updated_parameter = thetaest_vec1(end);

%Finding the residuals
y_bocd = ynew(201:500,:); 
y_reg = ynew(200:499, :);
model_pred = Updated_parameter .* y_reg;
residuals = y_bocd - model_pred;


%Running the BOCD algorithm on the residuals after 200 observations
%Specifying the parameters as defined in the question
RL_probabilities2 = BayesianChangePoint(residuals, 50, 0, 1, 10, 1);
plot_graph_Q2(length(residuals), residuals, RL_probabilities2)

% From the graph, we observe that the changepoint is somewhere in between 85-95
% probabs for t=80:90 and their RL probabilities
disp(RL_probabilities2(80:90, 1:10)); 
%Rows represent the run length probalities for ith time step
%Columns represent the run lengths (0,1,2.....,10)

% We see that successive RL probabilities for RL =1, RL =2, RL =3, RL = 4 for 87, 88, 89, 90 respectively is the highest.
% Therefore changept = 86 which is also coherent with the graph
fprintf('Changepoint after 200th observation = 86\n');



%% Question 2b)

obj_new = recursiveLS(1,'InitialParameters',1, 'InitialParameterCovariance',10);
thetaest_vec_new = []; Ptheta_vec_new = [];
x_new = ynew(286:499);
y_out_new = ynew(287:500);

for i = 1:numel(x_new)
    H_new = x_new(i);
    [theta_new,~] = obj_new(y_out_new(i),H_new);
    Ptheta_vec_new(i) = obj_new.ParameterCovariance;
    thetaest_vec_new(i) = theta_new;
end

% Updated Parameter of the AR model until 200th observation
Updated_parameter_new = thetaest_vec_new(end);
fprintf('Updated Paramter of the new model= %f\n', Updated_parameter_new)

% Calculating the variance of driving noise 
% Relation between the residual noise and driving noise discussed in class
% We know the parameter value with good confidence using the recursiveLS
residuals_new = ynew(287:500)- Updated_parameter_new.*ynew(286:499);
variance_new = 1/(length(residuals_new) - 1).* sum(residuals_new .* residuals_new);
fprintf('Variance of the driving noise = %f\n',variance_new)





