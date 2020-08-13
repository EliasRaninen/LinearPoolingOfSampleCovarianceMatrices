function [SSCM, U] = compute_SSCM(X,n,p)
%%COMPUTE_SSCM computes the sample spatial sign covariance matrix.
% X is a nxp matrix, where n is the number of samples, p is the dimension.
% Centering of the samples is done with the spatial median.
%
% Usage: [SSCM,U] = compute_SSCM(X,n,p), where
% SSCM is the computed spatial sign covariance matrix and U is a nxp matrix
% of centered and normalized samples on the unit sphere.
%
% By E. Raninen (2020)

assert(isequal(size(X),[n,p]));

% compute spatial median
mu = spatialmedian(X,n,p);
assert(isequal(size(mu),[1 p]));
% center by the spatial median
Xc = X - repmat(mu,n,1);
% compute the norm of each sample
normXc = sqrt(sum(Xc.^2,2));
assert(all(normXc > 0))
% divide each sample by its norm (take them to the unit sphere)
U  = Xc./repmat(normXc,1,p);
% compute sample covariance of normalized samples
SSCM = U'*U/n; % normalization by n
