% Consider covariance matrix estimation in a multiclass setting, where
% there are multiple classes (populations). We propose to estimate each
% class covariance matrix as a linear combination of all of the class
% sample covariance matrices (SCM). This approach is shown to reduce the
% estimation error when the sample sizes are limited and the true class 
% covariance matrices share a similar structure. This script demonstrates
% an effective method for estimating the minimum mean squared error 
% coefficients for the linear combination when the samples are drawn from
% (unspecified) elliptically symmetric distributions with finite fourth 
% moments. For more information, see E. Raninen, D.E. Tyler, and E. Ollila
% "Linear pooling of sample covariance matrices".
%
% This script demonstrates the proposed linear pooling of SCMs.
%
% By E. Raninen 2020

clear; clc;
rng(123);

%% Define four classes (populations)
% number of classes
K = 4;
% number of samples
n = [50 75 100 125].';
% dimension
p = 200;

% function for generating AR1
AR1cov = @(r,p) r.^abs(repmat(1:p,p,1)-repmat(1:p,p,1)');
% function for generating CS
CScov  = @(r,p) r*(ones(p)-eye(p)) + eye(p);

% the population covariance matrices
rho = [0.3 0.4 0.5 0.6];
trueCovarianceMatrices{1} = AR1cov(rho(1),p);
trueCovarianceMatrices{2} = AR1cov(rho(2),p);
trueCovarianceMatrices{3} = CScov(rho(3),p);
trueCovarianceMatrices{4} = CScov(rho(4),p);

% the means of the classes
trueMeans{1} = randn(p,1);
trueMeans{2} = randn(p,1);
trueMeans{3} = randn(p,1);
trueMeans{4} = randn(p,1);

%% Monte Carlo loop
% number of Monte Carlos
nmc = 300;

NSE1 = nan(nmc,K); % normalized squared error (NSE) for LIN1
NSE2 = nan(nmc,K); % for LIN2

for mc=1:nmc
    %% Generate multivariate t samples from classes
    % degrees of freedom of multivariate t data
    dof = [8 8 8 8];
    % To generate multivariate t with covariance Sig, we need to divide
    % Sig with the variance dof/(dof-2).
    variance = dof./(dof-2);
    
    dataFromClasses = cell(K,1);
    for k=1:K
        Sig = trueCovarianceMatrices{k};
        % normal distributed N(0,Sig/variance)
        N = randn(n(k),p)*sqrtm(Sig/variance(k));
        % multivariate t with covariance matrix Sig
        X = N ./ repmat(sqrt(chi2rnd(dof(k), n(k), 1)/dof(k)), 1, p);
        % add the mean
        dataFromClasses{k} = X + repmat(trueMeans{k}.', n(k), 1);
    end
    
    %% Estimate covariance matrices from the data
    params = estimate_parameters(dataFromClasses);
    
    % without identity shrinkage
    C = params.trCiCj/p;
    D = diag(params.MSE_Sk)/p;
    [LIN_POOL_1, A_1] = linpoolQP(params.SCM,C,D,1:K);
    
    % with identity shrinkage
    C_tilde = [C params.eta; params.eta.' 1];
    D_tilde = [D zeros(K,1); zeros(1,K) 0];
    [LIN_POOL_2, A_2] = linpoolQP([params.SCM; eye(p)],C_tilde,D_tilde,1:K);
    
    %% Compute normalized squared error NSE
    NSE = @(A,k) norm(A-trueCovarianceMatrices{k},'fro')^2/norm(trueCovarianceMatrices{k},'fro')^2;
    for k=1:K
        NSE1(mc,k) = NSE(LIN_POOL_1{k},k);
        NSE2(mc,k) = NSE(LIN_POOL_2{k},k);
    end
    
    if mod(mc,20)==0; fprintf('.'); end
end
fprintf('\n');
%% NMSE

% Average results over Monte Carlos
NMSE1 = mean(NSE1); STD1 = std(NSE1);
NMSE2 = mean(NSE2); STD2 = std(NSE2);

disp('Normalized MSE and standard deviation for the four classes.');
fprintf('(averaged over %d Monte Carlo trials.)\n',nmc);
% Table for NMSE
T = splitvars(table(round([NMSE1;NMSE2],2)));
T.Properties.VariableNames = {'AR1(0.3)','AR1(0.4)','CS(0.5)','CS(0.6)'};
T.Properties.RowNames = {'LIN1 NMSE:','LIN2 NMSE:'};
disp(T);

% Table for standard deviation
Tstd = splitvars(table(round([STD1;STD2],3)));
Tstd.Properties.VariableNames = {'AR1(0.3)','AR1(0.4)','CS(0.5)','CS(0.6)'};
Tstd.Properties.RowNames = {'LIN1 STD:','LIN2 STD:'};
disp(Tstd)