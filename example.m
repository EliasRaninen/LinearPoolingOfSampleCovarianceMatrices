% Consider covariance matrix estimation in a multiclass setting, where there are
% multiple classes (populations). We propose to estimate each class covariance
% matrix as a linear combination of all of the class sample covariance matrices
% (SCM). This approach is shown to reduce the estimation error when the sample
% sizes are limited and the true class covariance matrices share a similar
% structure. This script demonstrates an effective method for estimating the
% minimum mean squared error coefficients for the linear combination when the
% samples are drawn from (unspecified) elliptically symmetric distributions with
% finite fourth moments. For more information, see the article: E. Raninen, D.
% E. Tyler and E. Ollila, "Linear pooling of sample covariance matrices," in
% IEEE Transactions on Signal Processing, Vol 70, pp. 659-672, 2021, doi:
% 10.1109/TSP.2021.3139207.

% This script demonstrates the proposed linear pooling of SCMs.
%
% By E. Raninen and E. Ollila (2021)

clear; clc;
rng(0);

%% Define four classes (populations)
% number of classes
K = 4;
% number of samples
n = [20 100 20 100].';
% dimension
p = 100;

% function for generating AR1
AR1cov = @(r,p) r.^abs(repmat(1:p,p,1)-repmat(1:p,p,1)');
% function for generating CS
CScov  = @(r,p) r*(ones(p)-eye(p)) + eye(p);

% the population covariance matrices
rho = [0.3 0.4 0.5 0.6];
trueCovarianceMatrices{1} = 1*AR1cov(rho(1),p);
trueCovarianceMatrices{2} = 2*AR1cov(rho(2),p);
trueCovarianceMatrices{3} = 3*CScov(rho(3),p);
trueCovarianceMatrices{4} = 4*CScov(rho(4),p);

% the means of the classes
trueMeans{1} = randn(p,1);
trueMeans{2} = randn(p,1);
trueMeans{3} = randn(p,1);
trueMeans{4} = randn(p,1);

%% Monte Carlo loop
% number of Monte Carlos
nmc = 1000;

NSE1 = nan(nmc,K); % for normalized squared error (NSE)
NSE2 = nan(nmc,K);

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

    % with identity shrinkage
    [LIN_POOL_1, A_1] = linpool(dataFromClasses,'linear');

    % convex with identity shrinkage
    [LIN_POOL_2, A_2] = linpool(dataFromClasses,'convex');

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
NMSE1 = [mean(NSE1) mean(sum(NSE1,2))]; STD1 = [std(NSE1) std(sum(NSE1,2))];
NMSE2 = [mean(NSE2) mean(sum(NSE2,2))]; STD2 = [std(NSE2) std(sum(NSE2,2))];

disp('Normalized MSE and standard deviation for the four classes.');
fprintf('(averaged over %d Monte Carlo trials.)\n',nmc);
% Table for NMSE
T = splitvars(table(round([NMSE1;NMSE2],2)));
T.Properties.VariableNames = {'AR1(0.3)','AR1(0.4)','CS(0.5)','CS(0.6)','Sum'};
T.Properties.RowNames = {'LINPOOL-Linear NMSE:','LINPOOL-Convex NMSE:'};
disp(T);

% Table for standard deviation
Tstd = splitvars(table(round([STD1;STD2],3)));
Tstd.Properties.VariableNames = {'AR1(0.3)','AR1(0.4)','CS(0.5)','CS(0.6)','Sum'};
Tstd.Properties.RowNames = {'LINPOOL-Linear STD:','LINPOOL-Convex STD:'};
disp(Tstd)
