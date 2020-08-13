function [Sigmas, A] = linpoolQP(SCM, C, D, G)
%LINPOOL computes regularized sample covariance matrix estimates by pooling
%the class SCMs via a non-negative linear combination. This function first
%computes the unconstrained solution for the optimal coefficients by using
%Matlab's '\' matrix inversion operator. If the optimal coefficients have
%negative values, the function imposes the nonnegativity constraint and
%uses Matlab's QP solver quadprog.m.
%
% Usage:
%  [Sigmas, A] = linpoolQP(SCM, C, D, G)
%
%   outputs:
%       Sigmas  - a Kx1 cell array of computed covariance matrix estimates.
%       A       - a KxK matrix of coefficients used in the linear 
%                 combination, i.e., Sigmas{k} = sum_{j=1}^K a(j,k)*SCM{j}.
%   inputs:
%       SCM     - the class sample covariance matrices (Kx1 cell array).
%       C       - matrix consisting of the estimates of the scaled inner
%                 products, trace(C_i * C_j)/p, where C_i and C_j are the
%                 population covariance matrices of classes i and j, and p
%                 is the dimension.
%       D       - is a diagonal matrix consisting of the estimates of the
%                 scaled mean squared error of the SCMs, i.e., D(i,i) is 
%                 the estimated MSE of the SCM of class i divided by p.
%       G       - a vector denoting the classes for which to compute the
%                 linear combination. For example, 
%                    G = 1 computes the covariance matrix only for the
%                    first class, k=1.
%                    G = 1:K computes the covariance matrix estimates for
%                    classes 1,2,...,K.
%
% Example:
%
%    params = estimate_parameters(dataFromClasses);
%
%    % without identity shrinkage
%    C = params.trCiCj/p;
%    D = diag(params.MSE_Sk)/p;
%    [LIN_POOL_1, A_1] = linpoolQP(SCM,C,D,1:K);
%
%    % with identity shrinkage
%    C_tilde = [C params.eta; params.eta.' 1];
%    D_tilde = [D zeros(K,1); zeros(1,K) 0];
%    [LIN_POOL_2, A_2] = linpoolQP([SCM; eye(p)],C_tilde,D_tilde,1:K);
%
% By E. Raninen (2020)

K = numel(SCM); % number of classes
p = size(SCM{1},1); % dimension

if ~exist('G','var')
    G = 1:K;
end

A = nan(K);
Sigmas = cell(K,1);

% The unconstrained solution. Used in case it is non-negative.
A0 = (C + D)\C;

% Solve the pooled estimators for each class using QP solver of MATLAB
opts = optimoptions('quadprog','Display','Off');
for k = G
    % solve coefficients of linear combination
    if any(A0(:,k) < 0)
        A(:,k) = quadprog(D + C,-C(:,k),[],[],[],[],zeros(K,1),[],[],opts);
    else
        A(:,k) = A0(:,k);
    end
    % compute covariance matrix estimate of class k
    Sigmas{k} = zeros(p);
    for j=1:K
        Sigmas{k} = Sigmas{k} + A(j,k)*SCM{j};
    end
end