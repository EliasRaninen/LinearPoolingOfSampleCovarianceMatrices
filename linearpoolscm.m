function [Sigmas, A, SigmasI, AI] = linearpoolscm(dataFromClasses)
%LINPOOLSCM computes regularized sample covariance matrix estimates by
%pooling the class SCMs via a nonnegative linear combination. This function
%is a wrapper for the function linpoolQP.m.
%
%Usage:
%  [Sigmas, A, SigmasI, AI] = linearpoolscm(dataFromClasses)
%
%outputs:
%       Sigmas  - a Kx1 cell array of computed covariance matrix estimates,
%                 K is the number of classes
%       A       - a KxK matrix of coefficients used in the linear
%                 combination, i.e., Sigmas{k} = sum_{j=1}^K a(j,k)*SCM{j}.
%
%       SigmasI - a (K+1)x1 cell array of computed covariance matrix
%                 estimates, where in addition to the class SCM a identity 
%                 matrix is used as shrinkage target.
%       AI      - a (K+1)xK matrix of coefficients used in the linear
%                 combination Sigmas{k} = sum_{j=1}^K a(j,k)*SCM{j} +
%                 a(K+1,k)*I, where I is the identity matrix.

%inputs:
% dataFromClasses - a Kx1 cell array of data of the classes, where each
%                   cell is a (nxp) matrix with observations are stacked as
%                   row-vectors, i,e., rows and columns correspond to
%                   samples and variables, respectively.
%
% By E. Raninen (2020)

%% validate input
K = numel(dataFromClasses); % number of classes
assert(all(size(dataFromClasses) == [K,1]))

%% estimate parameters
params = estimate_parameters(dataFromClasses);
SCM = params.SCM;
p = size(SCM{1},2);

%% compute estimators
% without identity shrinkage
C = params.trCiCj/p;
D = diag(params.MSE_Sk)/p;
[Sigmas, A] = linpoolQP(SCM,C,D);

% with identity shrinkage
C_tilde = [C params.eta(:); params.eta(:).' 1];
D_tilde = [D zeros(K,1); zeros(1,K) 0];
[SigmasI, AI] = linpoolQP([SCM; eye(p)],C_tilde,D_tilde);